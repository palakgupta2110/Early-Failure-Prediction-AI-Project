# Early Failure Prediction AI — Professional Edition

A research prototype for predicting abnormal/failure-related windows from the UCI MetroPT-3 Air Compressor dataset.

## Models
This version compares exactly five classifiers (no Random Forest):
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes
- Logistic Regression
- Decision Tree

## Important limitations
Failure windows are rare and labels are approximations derived from multivariate sensor deviations (top 2% cutoff). Scores are not calibrated real-world probabilities. Treat this as an educational research prototype, not a safety-critical monitoring system. Model performance is evaluated using stratified train/validation/test splits (to ensure rare proxy anomalies appear reliably in each split), with precision/recall and PR-AUC alongside accuracy.

## Windows setup
Open CMD in this folder and run:

```cmd
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\fetch_data.py
python src\train_model.py
python -m streamlit run app.py
```

If `py` is unavailable, use `python` instead. Dataset download requires internet access and may take time.

## Data source
MetroPT-3 Air Compressor dataset, UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/791/metropt+3+dataset

The fetch script downloads the original archive from UCI when available and attempts to locate the CSV. If download layout changes, download the dataset manually from the UCI page and place the CSV in `data/` as `MetroPT3(AirCompressor).csv`.

## Project structure
- `app.py`: Streamlit dashboard
- `src/fetch_data.py`: dataset downloader
- `src/pipeline.py`: loading and aggregation
- `src/train_model.py`: stratified training and model comparison
- `models/`: trained artifacts created after training
- `reports/`: evaluation outputs
- `docs/PROJECT_EXPLANATION.md`: project overview
- `docs/VIVA_QUESTIONS.md`: viva practice
