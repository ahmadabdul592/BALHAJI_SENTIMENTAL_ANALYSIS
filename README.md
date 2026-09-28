# Nigerian Social Media Sentiment Analysis — Streamlit App

Real-time deployment of the classifier trained in
`b-Sentimental.ipynb`.

## 1. Get the model artifacts

1. Run the notebook on Jupyter, Collab or Kaggle  Notebook(upload your annotated dataset first — see the
   notebook).
2. The last cells save four files to `../working/`:
   `sentiment_model.pkl`, `vectorizer.pkl`, `label_encoder.pkl`,
   `model_metadata.json`.
3. Open the notebook's **Output** tab (right sidebar) and download those four
   files.
4. Place them in this project's `model/` folder, replacing the placeholder.

## 2. Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## 3. What the app does

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
b-sentimental.ipynb
streamlit_app/
├── app.py               # Streamlit application
├── requirements.txt
├── README.md
└── model/
    ├── sentiment_model.pkl     
    ├── vectorizer.pkl         
    ├── label_encoder.pkl      
    ├── model_metadata.json    
    └── transformer_model/         # <- OPTIONAL: fine-tuned transformer
```
