from transformers import pipeline

def analyze(task, model, data, batch_size=32):
    analyzer = pipeline(task=task, model=model, device=-1)  # CPU inference

    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        preds = analyzer(batch)
        if isinstance(preds, dict):
            results.append(preds)
        elif isinstance(preds, list):
            results.extend(preds)
        else:
            raise TypeError(f"Unexpected prediction type: {type(preds)}")

    return results
