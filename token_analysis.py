import pandas as pd
from transformers import AutoTokenizer

# Load the Parquet file into a Pandas DataFrame
df = pd.read_parquet('datasets/ill_caselaw.parquet')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl')

# Tokenize the text column and count the number of tokens
df['num_tokens'] = df['text'].apply(lambda x: len(tokenizer.tokenize(x)))

# Calculate the minimum, average, and maximum number of tokens
min_tokens = df['num_tokens'].min()
avg_tokens = df['num_tokens'].mean()
max_tokens = df['num_tokens'].max()

# Print the results
print(f"Minimum number of tokens: {min_tokens}")
print(f"Average number of tokens: {avg_tokens}")
print(f"Maximum number of tokens: {max_tokens}")
