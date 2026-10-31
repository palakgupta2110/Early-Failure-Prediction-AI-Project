from pathlib import Path
import os
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "MetroPT3(AirCompressor).csv"
CACHE_PATH = ROOT / "data" / "aggregated_5min.parquet"


def get_dataset_metadata(path=DATA_PATH):
    """Return presence and metadata for the raw MetroPT-3 dataset."""
    path = Path(path)
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "size_mb": 0.0,
            "modified": None,
        }
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": pd.to_datetime(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_and_aggregate(path=DATA_PATH, window="5min", use_cache=True):
    """
    Load sensor readings, aggregate them into temporal windows, and generate
    deterministic multivariate proxy anomaly labels.

    Optimized for performance:
    - High-speed parquet cache when available.
    - C-engine CSV parsing with delimiter detection.
    - Deterministic proxy anomaly labeling at the 98th percentile.
    """
    path = Path(path)
    cache_file = ROOT / "data" / f"aggregated_{window}.parquet"

    # Fast-path: Load from parquet cache if present and newer than source CSV
    if use_cache and cache_file.exists():
        try:
            if not path.exists() or cache_file.stat().st_mtime >= path.stat().st_mtime:
                cached_df = pd.read_parquet(cache_file)
                if "__target_y__" in cached_df.columns and "__timestamp__" in cached_df.columns:
                    y = cached_df["__target_y__"].astype(int)
                    timestamps = pd.to_datetime(cached_df["__timestamp__"])
                    feature_cols = [c for c in cached_df.columns if c not in ("__target_y__", "__timestamp__")]
                    X = cached_df[feature_cols].copy()
                    X.index = timestamps
                    y.index = timestamps
                    return X.astype(float), y, timestamps
        except Exception:
            # Fall back to regenerating from CSV if cache reading fails
            pass

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python src/fetch_data.py"
        )

    # Detect delimiter using the first line to enable the fast C engine
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    try:
        df = pd.read_csv(path, sep=sep, engine="c")
    except Exception:
        # Fallback to python engine only if C engine encounters unusual format
        df = pd.read_csv(path, sep=sep, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    time_col = next(
        (c for c in df.columns if c.lower() in {"timestamp", "time", "date"}),
        df.columns[0]
    )

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric sensor columns found in the dataset.")

    # Aggregate sensor readings into stable windows (mean, std, min, max)
    agg = numeric.resample(window).agg(["mean", "std", "min", "max"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0)

    # Weak proxy label: mark extreme multivariate sensor windows as anomalies.
    # Explicitly documented as a proxy label, not authoritative failure ground truth.
    raw_mean = numeric.resample(window).mean().reindex(agg.index).fillna(0)
    median_val = raw_mean.median()
    mad_val = raw_mean.sub(median_val).abs().median().replace(0, np.nan).fillna(raw_mean.std().replace(0, 1))
    z = (raw_mean - median_val) / mad_val
    score = z.abs().replace([np.inf, -np.inf], np.nan).fillna(0).mean(axis=1)
    cutoff = score.quantile(0.98) if len(score) else 0
    y = (score >= cutoff).astype(int)

    # Drop timestamps with all-zero/invalid features
    keep = agg.notna().any(axis=1)
    X = agg.loc[keep].astype(float)
    y = y.loc[keep].astype(int)
    timestamps = agg.index[keep]

    # Persist cache for subsequent instantaneous page loads
    try:
        combined = X.copy()
        combined["__target_y__"] = y.values
        combined["__timestamp__"] = timestamps
        combined.to_parquet(cache_file, index=False)
    except Exception:
        pass

    return X, y, timestamps


def get_downsampled_telemetry(window="1h", max_points=1200):
    """
    Return downsampled multi-sensor telemetry for responsive visualization.
    Derives from the cached 5-minute features to avoid reprocessing raw rows.
    """
    X, y, timestamps = load_and_aggregate()
    if X.empty:
        return pd.DataFrame()

    # Core physical channels from mean features
    mean_cols = [c for c in X.columns if c.endswith("_mean")]
    telemetry = X[mean_cols].copy()
    telemetry.columns = [c[:-5] for c in telemetry.columns]  # remove '_mean'
    telemetry["proxy_anomaly"] = y.values

    # Resample or step-downsample to keep trace size responsive
    if len(telemetry) > max_points:
        step = max(1, len(telemetry) // max_points)
        telemetry_sampled = telemetry.iloc[::step].copy()
    else:
        telemetry_sampled = telemetry.copy()

    telemetry_sampled.index = pd.to_datetime(telemetry_sampled.index)
    return telemetry_sampled
