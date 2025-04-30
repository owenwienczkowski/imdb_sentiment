from datasets import load_dataset

def get_dataset(name, split):
    # establish pipeline
    return load_dataset(name, split=split)

