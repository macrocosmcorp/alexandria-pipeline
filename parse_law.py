import pandas as pd
import jsonlines

input_path = './ill_text_20210921/data/data.jsonl'
output_path = './datasets/ill_caselaw.parquet'

# Read in the JSONL file and convert to a Pandas DataFrame
with jsonlines.open(input_path) as reader:
    data = []
    for i, obj in enumerate(reader):
        id = obj['id']
        opinions = obj['casebody']['data']['opinions']
        for opinion in opinions:
            text = opinion['text']
            # Maybe introduce a minimum length requirement?
            if text is None or text == '':
                continue
            data.append({'id': id, 'text': text})
        if i % 1000 == 0:
            print(f'Processed {i} cases')
    df = pd.DataFrame(data)

df.dropna(inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)
print(f'len(data): {len(df)}')

# Write the DataFrame to a Parquet file
df.to_parquet(output_path)
