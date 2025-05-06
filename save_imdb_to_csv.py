from datasets import load_dataset
import pandas as pd

# Load the unsupervised split (unlabeled reviews)
dataset = load_dataset("stanfordnlp/imdb", split="unsupervised")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Save to CSV
df.to_csv("imdb_unsupervised.csv", index=False)

print("Saved to imdb_unsupervised.csv")
