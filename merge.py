import glob
import os
import pickle
import re

import numpy as np


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

def print_pickle_data(data):
    for title, embedding, id in data[:1]:
        print("Title:", title)
        print("Embedding:", embedding)
        print("ID:", id)
        print()

def merge_data(data1, data2, ratio):
    assert len(data1) == len(data2), "Data sizes do not match"

    merged_data = []
    for (title1, embedding1, id1), (title2, embedding2, id2) in zip(data1, data2):
        assert id1 == id2, "Mismatched ids between datasets"
        
        combined_embedding = np.concatenate([embedding1*ratio, embedding2*(1-ratio)])
        merged_data.append((title1, combined_embedding, id1))
    
    return merged_data

def extract_number(filepath):
    # Extract the number from the filename
    number = re.search(r'(\d+)', filepath)
    return int(number.group(1))

if __name__ == "__main__":
    directory_path_abstracts = './embeddings/arxiv_abstracts/'
    directory_path_titles = './embeddings/arxiv_titles/'
    output_directory = './merged_embeddings/'
    os.makedirs(output_directory, exist_ok=True)  # Make sure the output directory exists
    ratio = 0.2

    abstract_files = sorted(glob.glob(os.path.join(directory_path_abstracts, 'embeddings_*.pkl')), key=extract_number)
    title_files = sorted(glob.glob(os.path.join(directory_path_titles, 'embeddings_*.pkl')), key=extract_number)

    print("Abstract files count: ", len(abstract_files))
    print("Title files count: ", len(title_files))

    # Iterate over the larger files
    sum = 0
    for i, title_file in enumerate(title_files):
        print("Processing title file: ", title_file)
        title_data = load_pickle(title_file)

        print("Title data length: ", len(title_data))
        sum += len(title_data)

    # print("Sum: ", sum)
    #     # Make sure each larger file has the right number of documents
    #     assert len(title_data) == 1010000 if i == 0 else (1000000 if i != len(title_files) - 1 else 400000), "Unexpected number of documents in larger file"
        
    #     # Determine which smaller files we need
    #     start_index = i * 10
    #     end_index = (i + 1) * 10 if i != len(title_files) - 1 else len(abstract_files)  # Handle the last file with only 400,000 documents
    #     print("Processing abstract files from index {} to {}".format(start_index, end_index))

    #     title_start = 0
    #     # Merge with corresponding smaller files
    #     for j in range(start_index, end_index):
    #         print("Processing abstract file: ", abstract_files[j])
    #         abstract_data = load_pickle(abstract_files[j])

    #         print("Abstract data length: ", len(abstract_data))
    #         # Make sure each smaller file has the right number of documents
    #         assert len(abstract_data) == 101000 if j == 0 else 100000, "Unexpected number of documents in smaller file"

    #         print("Merging data...")
    #         # For each smaller file, take the corresponding part from the larger file
    #         end = title_start + len(abstract_data)
    #         merged_data = merge_data(abstract_data, title_data[title_start:end], ratio)
    #         title_start = end

    #         # Save merged data
    #         output_path = os.path.join(output_directory, 'merged_embeddings_{}.pkl'.format(j))
    #         save_pickle(merged_data, output_path)

    #         # Check merged data
    #         print("Merged data:")
    #         print_pickle_data(merged_data)
    #         print("Merge completed for this pair.")

    sum = 0
    # iterate over the smaller files
    for i, abstract_file in enumerate(abstract_files):

        abstract_data = load_pickle(abstract_file)

        print("Abstract data length: ", len(abstract_data))
        sum += len(abstract_data)

    print(sum)