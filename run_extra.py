import time
import os
import pickle
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer
import pyarrow.parquet as pq
import numpy as np
import nltk
nltk.download('punkt')

model = INSTRUCTOR('hkunlp/instructor-xl')
tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-xl')


class ParquetDataset(Dataset):
    def __init__(self, parquet_path, start_line, type, crop=False, batch_size=None):
        self.parquet_path = parquet_path
        self.start_line = start_line
        self.crop = crop
        self.type = type
        self.batch_size = batch_size
        self.table = pq.read_table(self.parquet_path)
        self.data = self.table.to_pandas()

        if self.crop and self.batch_size is not None:
            self.data = self.data[:self.batch_size * 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        content = row[self.type]
        id = row.id

        return {"id": id, "content": content}


def handle_long_sentence(sentence, text_chunks, max_tokens):
    # recursively split long sentences into smaller chunks until they are all under max_tokens in a very dumb way
    sentence_len = len(tokenizer.tokenize(sentence))
    if sentence_len > max_tokens:
        # split sentence in half
        half_len = len(sentence) // 2
        first_half = sentence[:half_len]
        second_half = sentence[half_len:]

        # recursively split each half
        handle_long_sentence(first_half, text_chunks, max_tokens)
        handle_long_sentence(second_half, text_chunks, max_tokens)
    else:
        text_chunks.append([sentence])

# calculate the weighted average of embeddings


def weighted_average_embedding(chunk_embeddings, text_chunks):
    weights = [len(chunk[1]) for chunk in text_chunks]
    return np.average(chunk_embeddings, axis=0, weights=weights)


def process_batch(batch, max_tokens=512):
    start_time = time.time()
    # Initialize list for final embeddings
    final_embeddings = []

    # Iterate over each text in the batch, text[0] is prompt, text[1] is content
    for text in batch:
        if text == "":
            continue
        # Split text into sentences
        sentences = nltk.sent_tokenize(text[1])

        # Split text into chunks of up to max_tokens tokens by sentences
        text_chunks = []
        text_chunk = []
        chunk_len = 0
        for sentence in sentences:
            sentence_len = len(tokenizer.tokenize(sentence))

            # Sentences over max_tokens are split into their own chunks
            if sentence_len > max_tokens:
                if text_chunk:
                    text_chunks.append(text_chunk)
                    text_chunk = []
                    chunk_len = 0
                handle_long_sentence(sentence, text_chunks, max_tokens)
                continue

            if chunk_len + sentence_len <= max_tokens:
                text_chunk.append(sentence)
                chunk_len += sentence_len
            else:
                text_chunks.append(text_chunk)
                text_chunk = [sentence]
                chunk_len = sentence_len

        if text_chunk:
            text_chunks.append(text_chunk)

        # Convert chunks back into text
        text_chunks = [[text[0], ' '.join(chunk)] for chunk in text_chunks]

        # Embed each chunk and calculate their weighted average
        chunk_embeddings = model.encode(text_chunks)
        final_embedding = weighted_average_embedding(
            chunk_embeddings, text_chunks)
        final_embeddings.append(final_embedding)

    end_time = time.time()
    elapsed_time = end_time - start_time
    speed = len(batch) / elapsed_time
    print(
        f"Processed {len(batch)} entries in {elapsed_time:.2f} seconds ({speed:.2f} entries/second)")
    return final_embeddings  # Return the list of final embeddings


def load_checkpoint(output_path):
    if os.path.exists(f'{output_path}checkpoint.pkl'):
        with open(f'{output_path}checkpoint.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return 0, 0  # default values


def save_checkpoint(output_path, batch_id, line_num):
    with open(f'{output_path}checkpoint.pkl', 'wb') as f:
        pickle.dump((batch_id, line_num), f)


def save_embeddings(output_path, batch_id, all_data):
    with open(f'{output_path}embeddings_{batch_id}.pkl', 'wb') as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return []


def get_output_path(id):
    path = 'embeddings/instance_' + str(id) + '/'
    if os.path.exists(path):
        return get_output_path(id + 1)
    else:
        os.makedirs(path)
        return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint_size', type=int, default=100)
    parser.add_argument('--start_checkpoint', type=int, default=0)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    CHECKPOINT_SIZE = args.checkpoint_size
    CHECKPOINT_START = args.start_checkpoint
    PROMPT = ''
    TYPE = args.type
    if TYPE == 'title':
        PROMPT = "Represent the Research Paper title for retrieval; Input:"
    elif TYPE == 'abstract':
        PROMPT = "Represent the Research Paper abstract for retrieval; Input:"
    elif TYPE == 'text':
        PROMPT = "Represent the US Judicial Opinion document for retrieval: "
    else:
        print("Invalid embed type")
        exit(1)

    DATASET_PATH = args.dataset_path
    TEST = args.test

    output_path = get_output_path(0)
    print("Using output path:", output_path)

    print("Using test option?", TEST)

    with open(f'{output_path}params.txt', 'w') as f:
        f.write(f'batch_size: {BATCH_SIZE}\n')
        f.write(f'checkpoint_size: { CHECKPOINT_SIZE}\n')
        f.write(f'prompt: {PROMPT}\n')
        f.write(f'type: {TYPE}\n')
        f.write(f'time string: {time.strftime("%Y%m%d-%H%M%S")}\n')

    print("Loading and processing parquet file...")
    print("Batch size: {}".format(BATCH_SIZE))
    print("Checkpoint size: {}".format(CHECKPOINT_SIZE))

    # Load and process parquet
    batch_id, start_line = load_checkpoint(output_path)

    dataset = ParquetDataset(DATASET_PATH, start_line,
                             TYPE, crop=TEST, batch_size=BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    all_data = []

    # Each data entry is a batch of size BATCH_SIZE
    for line_num, data in enumerate(dataloader, start_line):
        content_batch = data['content']
        id_batch = data['id']
        print(f'Processing batch {line_num}')
        content_prompt_batch = [[PROMPT, content]
                                for content in content_batch]
        content_embeddings = process_batch(content_prompt_batch)

        for i in range(len(content_batch)):
            all_data.append(
                (content_batch[i], content_embeddings[i], id_batch[i]))

        if line_num % CHECKPOINT_SIZE == 0 and line_num != 0:
            print(f'Saving checkpoint {batch_id}, progress {line_num}')
            save_checkpoint(output_path, batch_id, line_num)
            all_data = save_embeddings(output_path, batch_id, all_data)
            batch_id += 1

    # Save the final checkpoint
    print(f'Saving final checkpoint {batch_id}, progress {len(dataloader)}')
    all_data = save_embeddings(output_path, batch_id, all_data)
    save_checkpoint(output_path, batch_id, len(dataloader))
