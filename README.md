# Nigerian Social Media Sentiment Analysis — Streamlit App

Real-time deployment of the classifier trained in
`nigeria_sentiment_analysis_kaggle.ipynb`.

## 1. Get the model artifacts from Kaggle

1. Run the notebook on Kaggle (upload your annotated dataset first — see the
   notebook's Section 1A).
2. The last cells save four files to `/kaggle/working/`:
   `sentiment_model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`,
   `model_metadata.json`.
3. Open the notebook's **Output** tab (right sidebar) and download those four
   files.
4. Place them in this project's `model/` folder, replacing the placeholder
   `PUT_MODEL_FILES_HERE.txt`.

## 2. Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 3. Deploy for free — Streamlit Community Cloud

1. Push this folder (`app.py`, `requirements.txt`, `model/*.joblib`,
   `model/*.json`) to a **public or private GitHub repo**.
2. Go to https://share.streamlit.io → **New app** → pick the repo/branch →
   set the main file path to `app.py` → **Deploy**.
3. Streamlit Cloud installs `requirements.txt` and starts the app; NLTK data
   is downloaded automatically on first run (`ensure_nltk_data()` in `app.py`).
4. You'll get a public `https://<name>.streamlit.app` link.

### Alternative deployment options
- **Hugging Face Spaces** (Streamlit SDK): same repo contents, add a
  `README.md` header block HF requires, push to a new Space.
- **Docker / any cloud VM**: `docker run` a small image with
  `pip install -r requirements.txt` then `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
- **Render / Railway**: point the start command to
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## 4. What the app does

- **Single Post tab** — type or pick an example post, get an instant
  positive/negative/neutral prediction from the classical model with a confidence
  bar chart.
- **Batch (CSV) tab** — upload a CSV of posts, get predictions for every row,
  a sentiment-distribution chart, and a downloadable results CSV.
- **Model Performance tab** — shows classical-model accuracy/precision/recall/F1/AUC-ROC
  results produced during training (from `model_metadata.json`); transformer results are excluded.
- **About tab** — summarises the methodology (preprocessing, features,
  models compared).

## 5. Project structure

```
streamlit_app/
├── app.py               # Streamlit application
├── requirements.txt
├── README.md
└── model/
    ├── sentiment_model.joblib     # <- from Kaggle notebook output
    ├── vectorizer.joblib          # <- from Kaggle notebook output
    ├── label_encoder.joblib       # <- from Kaggle notebook output
    ├── model_metadata.json        # <- from Kaggle notebook output
    └── transformer_model/         # <- OPTIONAL: fine-tuned transformer (see section 6)
```

## Notes

- The preprocessing function in `app.py` is an exact copy of the one in the
  Kaggle notebook (Section 2), so predictions at inference time match what
  the model saw during training/evaluation.
- If you retrain with a different winning model/feature combination, just
  re-run Section 9 of the notebook and replace the four files — `app.py`
  needs no changes, since it reads `model_metadata.json` to know which model
  was deployed.
