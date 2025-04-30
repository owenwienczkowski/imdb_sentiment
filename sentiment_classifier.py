from src.load_data import get_dataset
from src.preprocess import tokenize, decode
from src.inference import analyze
from src.evaluate import evaluate_multiclass, evaluate_bucket

# import dataset
dataset = get_dataset("imdb", "train[:1%]+train[-1%:]")

# isolate text reviews
texts = dataset["text"]

# explore dataset
# print(dataset.features)   #{'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}
# print(len(texts)) # 500
# print(Counter(dataset["label"]))  #Counter({0: 250, 1: 250})

# tokenize (encode and decode)
distilbert_model_string = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenized, tokenizer = tokenize(texts, distilbert_model_string)
preprocessed = decode(tokenized, tokenizer)

# predict labels
distilbert = analyze("sentiment-analysis", distilbert_model_string, preprocessed, len(preprocessed))

# evaluate predictions
distilbert_results = evaluate_multiclass(dataset, distilbert, 'label', 'label', distilbert_model_string)
print(distilbert_results)


# Test another model
# msa = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
model_string_msa = "tabularisai/multilingual-sentiment-analysis"
tokenized_msa, tokenizer_msa = tokenize(texts, model_string_msa)
preprocessed_msa = decode(tokenized_msa, tokenizer_msa)
# tokenizer_msa = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")

msa =  analyze("sentiment-analysis", model_string_msa, preprocessed_msa, len(preprocessed_msa))

msa_results = evaluate_multiclass(dataset, msa, 'label', 'label', model_string_msa)
print(msa_results)

msa_bucket = evaluate_bucket( msa, 'label', model_string_msa)
print(msa_bucket)
