from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
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
        res.append(tknizr(_, padding = 'max_length', truncation=True, max_length = 512))
    return res

tokenized = format(tokenizer, texts)

# decoded text
readable_texts = [input['input_ids'] for input in tokenized]

preprocessed = tokenizer.batch_decode(readable_texts, skip_special_tokens=True)

# predict labels
analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# store predictions in a new list
results = analyzer(preprocessed, batch_size=len(preprocessed))
# print(results)

# calculate scoring metrics
ytrue = list((_ for _ in dataset["label"]))
ypred = []
for r in results:
    if r['label'] == "NEGATIVE":
        ypred.append(0)
    elif r['label'] == "POSITIVE":
        ypred.append(1)
    else:
        ValueError("Label not recognized.")

metrics = classification_report(y_true=ytrue, y_pred=ypred)
print(metrics)