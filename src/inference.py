from transformers import pipeline

def analyze(name, model, data, batch_size):
    # form predictions based on HuggingFace model
    analyzer = pipeline(name, model)
    return analyzer(data, batch_size=batch_size)
    