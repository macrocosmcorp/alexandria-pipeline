import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-base')

# Create a list to store articles that exceed the token limit
exceed_limit_records = []
columns = []

def extract_exceed_limit_data(parquet_path):
    global exceed_limit_records
    global columns

    table = pq.read_table(parquet_path)
    data = table.to_pandas()

    columns = data.columns

    for _, row in data.iterrows():
        abstract_tokens = len(tokenizer.tokenize(row['abstract']))

        if abstract_tokens > 512:
            exceed_limit_records.append(row.to_dict())

def save_to_parquet():
    global exceed_limit_records
    global columns
    
    df_exceed_limit = pd.DataFrame(exceed_limit_records, columns=columns)
    df_exceed_limit.to_parquet('datasets/arxiv_abstract_exceed_512.parquet', engine='pyarrow')

if __name__ == "__main__":
    extract_exceed_limit_data('datasets/arxiv2M.parquet')
    save_to_parquet()