# IMDb Sentiment Classifier using Hugging Face Transformers

A modular NLP pipeline and interactive web app for sentiment analysis on IMDb movie reviews using Hugging Face Transformers. This project demonstrates scalable inference, multilingual compatibility, model evaluation, and clean engineering practices.

## Project Overview

- **Goal**: Predict whether a movie review expresses positive or negative sentiment
- **Input Options**: Upload your own CSV or use built-in sample data
- **Models**:
  - [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) (binary)
  - [`tabularisai/multilingual-sentiment-analysis`](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) (multiclass)
- **Interface**: Built using [Streamlit](https://streamlit.io/) for a deployable, interactive UI
- **Features**:
  - Batch inference with tokenizer-aware chunking
  - Model selection and label mapping
  - Per-label filtering, explanation outputs, and downloadable results

## Directory Structure
```bash
imdb-sentiment-classifier/
├── app.py # Streamlit app
├── scripts/
│ └── run_pipeline.py # CLI workflow
├── src/
│ ├── load_data.py # Dataset loading
│ ├── preprocess.py # Tokenization and decoding
│ ├── inference.py # Hugging Face model inference
│ └── evaluate.py # Model evaluation
├── outputs/ # Evaluation results and logs
├── metrics_log.md # Logged metrics
├── requirements.txt # Project dependencies
└── README.md # You are here
```

## App Demo

You can interact with the web app to upload review data, select the model type, and view predictions directly. Results can be filtered by label and downloaded as CSV.

> Link to live app: **https://sentiment-classifier-owen-wienczkowski.streamlit.app/**

## How to Run Locally

> Make sure you're in the project root directory and have Python 3.8+.

1. **Install dependencies**
 ```bash
 pip install -r requirements.txt
 ```
2. **Run interactive web app**
  ```bash
  streamlit run app.py
  ```

## Results for Web App (Example)
![sentiment-metrics-1k](https://github.com/user-attachments/assets/165aab9c-a7a7-43e7-b962-0b31a2689733)

## Results for Pipeline

### DistilBERT

| Metric     | Positive | Negative | Accuracy |
|------------|----------|----------|----------|
| Precision  | 0.89     | 0.95     | 0.92     |
| Recall     | 0.93     | 0.90     |          |
| F1-Score   | 0.91     | 0.92     |          |

The DistilBERT model achieved strong performance, with balanced precision and recall across both sentiment classes. Evaluation was based on a stratified sample of 500 reviews.

### Tabularisai

When evaluated using label bucketing (e.g., grouping "Very Positive" and "Positive" together), the multilingual model achieved 100% accuracy on a reduced subset after removing neutral entries:

Classes	Accuracy
Positive vs Negative	1.00

Neutral predictions (not present in the original dataset) were excluded dynamically to enable fair binary classification.

## Skills Demonstrated

Hugging Face Transformers and inference pipelines

Streamlit interface design for real-time inference

Modular, production-style pipeline design

Tokenization, decoding, label mapping

Custom evaluation logic for multiclass-to-binary transitions

Multilingual model handling and class consolidation

Usable, downloadable web interface for non-technical users
