# Netflix Maturity Predictor (Streamlit)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

`model.joblib` is already trained. To retrain: `python train_model.py` (needs netflix_titles.csv in this folder).
Files: app.py (UI) · common.py (features + pipeline, identical to the notebook) · train_model.py · requirements.txt
Free hosting: push this folder to GitHub and deploy on Streamlit Community Cloud (main file: app.py).
