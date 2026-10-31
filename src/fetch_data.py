from pathlib import Path
import io
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TARGET = DATA_DIR / "MetroPT3(AirCompressor).csv"

URLS = [
    "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
    "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
]

def main():
    if TARGET.exists():
        print(f"Dataset already exists: {TARGET}")
        return
    print("Downloading MetroPT-3 from the UCI repository...")
    last_error = None
    for url in URLS:
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            content = response.content
            if content[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not candidates:
                        raise RuntimeError("No CSV found in downloaded archive.")
                    # Prefer the largest CSV member.
                    member = max(candidates, key=lambda n: zf.getinfo(n).file_size)
                    TARGET.write_bytes(zf.read(member))
                    print(f"Saved {member} to {TARGET}")
                    return
            if b"," in content[:2000] or b";" in content[:2000]:
                TARGET.write_bytes(content)
                print(f"Saved dataset to {TARGET}")
                return
            raise RuntimeError("Downloaded content was not a recognized CSV/ZIP.")
        except Exception as exc:
            last_error = exc
    print("Automatic download did not succeed.")
    print(f"Reason: {last_error}")
    print("Download the dataset manually from https://archive.ics.uci.edu/dataset/791/metropt+3+dataset")
    print(f"Place the CSV at: {TARGET}")

if __name__ == "__main__":
    main()
