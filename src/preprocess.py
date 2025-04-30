from transformers import AutoTokenizer

def tokenize(data, tokenizer):
    # initalize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    # format text for tokenization
    res = []
    for _ in data:
        res.append(tokenizer(_, padding='max_length', truncation=True, max_length=tokenizer.model_max_length))
    # return tokenizer and formatted text
    return res, tokenizer

def decode(data, tokenizer):
    # convert tokenized, encoded text to readable text
    readable_text = [input['input_ids'] for input in data]
    return tokenizer.batch_decode(readable_text, skip_special_tokens=True)