import streamlit as st
import pandas as pd
from pathlib import Path
from src.preprocess import tokenize, decode
from src.inference import analyze

st.title("HuggingFace Sentiment Analyzer")
st.caption("This tool allows users to upload a CSV file of text data and apply a pre-trained Hugging Face model for binary or multi-class sentiment analysis. Results are displayed, filterable, and downloadable.")

st.header("Upload and Configure Data")

def upload_data():
    input = st.file_uploader("Upload a CSV file", type="csv")
    if input is not None:
        if Path(input.name).suffix.lower() != ".csv":
            st.error("File must be CSV")
        else:
            df = pd.read_csv(input)
            return df
    return None

def batch_texts(texts, batch_size=32):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

model_map = {
    "binary": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "multi-class": "tabularisai/multilingual-sentiment-analysis"
}

# Load or upload data
if "data_with_predictions" in st.session_state:
    data = st.session_state["data_with_predictions"]
    reviews = st.session_state["selected_column"]
    classification_type = st.session_state["classification_type"]

if "data_with_predictions" in st.session_state:
    data = st.session_state["data_with_predictions"]
    reviews = st.session_state["selected_column"]
    classification_type = st.session_state["classification_type"]

    if "Prediction" not in data.columns:
        st.dataframe(data, use_container_width=True)

        string_columns = list(data.select_dtypes(include=['object']).columns)
        if string_columns:
            st.info(f"Using column: `{reviews}`")
        else:
            st.warning("No text columns found")

        st.info(f"Using model: `{model_map[classification_type]}`")

        if st.button("Run Sentiment Analysis"):
            texts = data[reviews].tolist()
            all_predictions = []

            with st.spinner("Analyzing..."):
                for batch in batch_texts(texts, batch_size=32):
                    tokenized, tokenizer = tokenize(batch, model_map[classification_type])
                    preprocessed = decode(tokenized, tokenizer)
                    predictions = analyze("sentiment-analysis", model_map[classification_type], preprocessed, len(preprocessed))
                    all_predictions.extend(predictions)

            st.success("Done!")

            data["Prediction"] = [pred['label'] for pred in all_predictions]
            data["Confidence"] = [pred['score'] for pred in all_predictions]
            data["Explanation"] = [f"Predicted as {p['label']} with {round(p['score'] * 100, 2)}% confidence" for p in all_predictions]

            st.session_state["data_with_predictions"] = data.copy()


else:
    data = upload_data()

    if data is None:
        if st.checkbox("Try example data"):
            sample_data = pd.DataFrame({
                "review": ["I loved this!", "It was terrible", "Just okay."]
            })
            st.session_state["data_with_predictions"] = sample_data.copy()
            st.session_state["selected_column"] = "review"
            st.session_state["classification_type"] = "binary"
            st.rerun()  # Trigger a full rerun to enter the top-level if-block
    else:
        if len(data) > 1000:
            st.warning("Only first 1000 rows will be processed.")
            data = data.head(1000)

        st.dataframe(data, use_container_width=True)

        string_columns = list(data.select_dtypes(include=['object']).columns)
        if string_columns:
            reviews = st.selectbox("Select the column to analyze.", string_columns, key="column_selector")
            st.session_state["selected_column"] = reviews
        else:
            st.warning("No text columns found")

        classification_type = st.selectbox("Select classification type", model_map.keys(), key="model_selector")
        st.session_state["classification_type"] = classification_type

        if st.button("Run Sentiment Analysis"):
            st.info(f"Using model: `{model_map[classification_type]}`")
            texts = data[reviews].tolist()
            all_predictions = []

            with st.spinner("Analyzing..."):
                for batch in batch_texts(texts, batch_size=32):
                    tokenized, tokenizer = tokenize(batch, model_map[classification_type])
                    preprocessed = decode(tokenized, tokenizer)
                    predictions = analyze("sentiment-analysis", model_map[classification_type], preprocessed, len(preprocessed))
                    all_predictions.extend(predictions)

            st.success("Done!")

            data["Prediction"] = [pred['label'] for pred in all_predictions]
            data["Confidence"] = [pred['score'] for pred in all_predictions]
            data["Explanation"] = [f"Predicted as {p['label']} with {round(p['score'] * 100, 2)}% confidence" for p in all_predictions]

            st.session_state["data_with_predictions"] = data.copy()

# Post-inference UI
if "data_with_predictions" in st.session_state and "Prediction" in st.session_state["data_with_predictions"].columns:
    data = st.session_state["data_with_predictions"]

    if "Prediction" in data.columns:
        st.header("Prediction Results")

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download with predicted labels",
            data=csv,
            file_name="sentiment_predictions.csv",
            mime="text/csv"
        )

        label_options = ["All"] + sorted(data["Prediction"].unique().tolist())
        selected_label = st.selectbox("Filter by prediction label", label_options, key="filter_label")

        if selected_label == "All":
            filtered = data
        else:
            filtered = data[data["Prediction"] == selected_label]

        st.dataframe(filtered, use_container_width=True)
        st.bar_chart(data["Prediction"].value_counts())

if st.button("Clear selections"):
    st.session_state.clear()
    st.rerun()

st.markdown("View the code on [GitHub](https://github.com/owenwienczkowski/imdb_sentiment)")
