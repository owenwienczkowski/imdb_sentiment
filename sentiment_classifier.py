from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoConfig
from sklearn.metrics import classification_report
# from collections import Counter   # used for EDA


# import dataset
dataset = load_dataset("imdb", split="train[:1%]+train[-1%:]")

# explore dataset
# print(dataset.features)   #{'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}
texts = dataset["text"]
# print(len(texts)) # 500
# print(Counter(dataset["label"]))  #Counter({0: 250, 1: 250})

# instantiate a tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# function to format texts to be tokenized and analyzed
def format(tknizr, list):
    res = []
    for _ in list:
        res.append(tknizr(_, padding = 'max_length', truncation=True, max_length = tknizr.model_max_length))
    return res

tokenized = format(tokenizer, texts)

# decoded text
readable_texts = [input['input_ids'] for input in tokenized]

preprocessed = tokenizer.batch_decode(readable_texts, skip_special_tokens=True)

# predict labels
analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# store predictions in a new list
results = analyzer(preprocessed, batch_size=len(preprocessed))

# calculate scoring metrics
ytrue = list((_ for _ in dataset["label"]))
ypred = []
for r in results:
    if r['label'] == "NEGATIVE":
        ypred.append(0)
    elif r['label'] == "POSITIVE":
        ypred.append(1)
    else:
        raise ValueError("Label not recognized.")

metrics = classification_report(y_true=ytrue, y_pred=ypred)
print(metrics)


# Test another model
msa = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")
tokenizer_msa = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
config = AutoConfig.from_pretrained("tabularisai/multilingual-sentiment-analysis")
# print(config.id2label)  # {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}

tokenized_msa = format(tokenizer_msa, texts)
readable_texts_msa = [input['input_ids'] for input in tokenized_msa]
preprocessed_msa = tokenizer_msa.batch_decode(readable_texts_msa, skip_special_tokens=True)

msa_results = msa(preprocessed_msa, batch_size=64)
print(msa_results[0])
ypred_msa = []
for r in msa_results:
    if r['label'] == "Very Negative" or r['label'] == "Negative":
        ypred_msa.append(0)
    elif r['label'] == "Very Positive" or r['label'] == "Positive":
        ypred_msa.append(1)
    elif r['label'] == "Neutral":
        ypred_msa.append(2)
    else:
        raise ValueError(f"Unexpected label: {r['label']}")
    
# determine how many neutral
ytrue_bin = []
ypred_bin = []
for y_t, y_p in zip(ytrue, ypred_msa):
    if y_p in [0, 1]:
        ytrue_bin.append(y_t)
        ypred_bin.append(y_p)

# The multilingual model showed moderate performance but frequently predicted Neutral, 
# a label not present in the original binary-labeled IMDb set. 
# These were flagged and analyzed separately.
msa_metrics = classification_report(y_true=ytrue, y_pred=ypred_msa)
print(msa_metrics)

# metrics using non-neutrals
print(classification_report(ytrue_bin, ypred_bin))
