import glob
import os
import pickle
import re

import numpy as np


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def print_pickle_data(data):
    for _, _, id in data:
        print("ID:", id)
    print()

def merge_data(titles, abstracts, ratio):
    assert len(titles) == len(abstracts), "Data sizes do not match"

    print(f"Merging {len(titles)} pairs of data...")

    merged_data = []
    for (title, title_embedding, id1), (_, abstract_embedding, id2) in zip(titles, abstracts):
        assert id1 == id2, f"Mismatched ids between title and abstract: {id1} != {id2}"
        
        combined_embedding = np.concatenate([title_embedding*ratio, abstract_embedding*(1-ratio)])
        merged_data.append((title, combined_embedding, id1))
    
    return merged_data

def extract_number(filepath):
    # Extract the number from the filename
    number = re.search(r'(\d+)', filepath)
    return int(number.group(1))

if __name__ == "__main__":
    directory_path_abstracts = './embeddings/arxiv_abstracts/'
    directory_path_titles = './embeddings/arxiv_titles/'
    ratio = 0.2

    abstract_files = sorted(glob.glob(os.path.join(directory_path_abstracts, 'embeddings_*.pkl')), key=extract_number)
    title_files = sorted(glob.glob(os.path.join(directory_path_titles, 'embeddings_*.pkl')), key=extract_number)

    print("Abstract files count: ", len(abstract_files))
    print("Title files count: ", len(title_files))

    # Iterate over the larger files
    for i, title_file in enumerate(title_files):
        print("Processing title file: ", title_file)
        title_data = load_pickle(title_file)

        print("Title data length: ", len(title_data))
        # Make sure each larger file has the right number of documents
        assert len(title_data) == 1010000 if i == 0 else (1000000 if i != len(title_files) - 1 else 400000), "Unexpected number of documents in larger file"
        
        # Determine which smaller files we need
        start_index = i * 10
        end_index = (i + 1) * 10 if i != len(title_files) - 1 else len(abstract_files)  # Handle the last file with only 400,000 documents
        print("Processing abstract files from index {} to {}".format(start_index, end_index))

        title_start = 0
        # Merge with corresponding smaller files
        for j in range(start_index, end_index):
            print("Processing abstract file: ", abstract_files[j])
            abstract_data = load_pickle(abstract_files[j])

            print("Abstract data length: ", len(abstract_data))
            # Make sure each smaller file has the right number of documents
            assert len(abstract_data) == 101000 if j == 0 else 100000, "Unexpected number of documents in smaller file"

            print("Merging data...")
            # For each smaller file, take the corresponding part from the larger file
            end = title_start + len(abstract_data)
            merged_data = merge_data(title_data[title_start:end], abstract_data, ratio)
            title_start = end

            # Check merged data
            print("Merged data:")
            print_pickle_data(merged_data)
            print("Merge completed for this pair.")
