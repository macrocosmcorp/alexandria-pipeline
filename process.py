import os
import pickle

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-base')

def print_pickle(data):
    for item in data:
        print(item)
        print('\n')

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def verify_pickles(titles, abstracts):
    for item in zip(titles, abstracts):
        _, _, doi = item[0]
        _, _, doi = item[1]
        assert doi == doi, "Mismatched doi between datasets"
        # print("ID:", doi)

def load_and_append_pkl_files(directory_path):
    data = []
    print(f'Start loading pkl files from {directory_path}...')
    # Traverse the directory
    files = []
    for root, dirs, files_in_dir in os.walk(directory_path):
        files.extend([os.path.join(root, file) for file in files_in_dir if file.startswith('embeddings_') and file.endswith('.pkl')])
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort files based on id

    for file in files:
        print(f'Loading file: {file}')
        with open(file, 'rb') as f:
            # Load the data from the .pkl file
            loaded_data = pickle.load(f)
            # Append the data to the list
            data.extend(loaded_data)
                    
    print('Finished loading data.')
    return data

def save_to_pkl(data, output_dir, output_file):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created directory: {output_dir}')

    print('len(data):', len(data))

    # Divide data into chunks of 100,000
    chunks = [data[x:x+100000] for x in range(0, len(data), 100000)]
    print(f'Divided data into {len(chunks)} chunks.')

    for i, chunk in enumerate(chunks):
        with open(os.path.join(output_dir, f'{output_file}_{i+1}.pkl'), 'wb') as f:
            # Write the list to a new .pkl file
            pickle.dump(chunk, f)
        print(f'Wrote chunk {i+1} to file.')

def load_parquet(parquet_path):
    table = pq.read_table(parquet_path)
    data = table.to_pandas()
    return data

def find_and_save_missing_dois():
    raw_parquet = load_parquet('./datasets/arxiv2M.parquet')
    raw_dois = raw_parquet['id'].tolist()
    print("Raw parquet count:", len(raw_dois))
    del raw_parquet

    title_dois, abstract_dois = [], []
    for i in range(1,23):
        print("Processing file: ", i)
        abstracts = load_pickle('./processed/abstract_chunks/abstracts_{}.pkl'.format(i))
        titles = load_pickle('./processed/title_chunks/ordered_titles_{}.pkl'.format(i))

        for title in titles:
            _, _, doi = title
            title_dois.append(doi)

        for abstract in abstracts:
            _, _, doi = abstract
            abstract_dois.append(doi)

    print("Title DOI count: ", len(title_dois))
    print("Abstract DOI count: ", len(abstract_dois))

    missing_title_dois, missing_abstract_dois = [], []
    for doi1, doi2 in zip(raw_dois, title_dois):
        if doi1 != doi2:
            missing_title_dois.append(doi1)
    if len(raw_dois) > len(title_dois):
        print("Mismatched title dois count:", len(raw_dois) - len(title_dois))
        missing_title_dois.extend(raw_dois[len(title_dois):])
    print("Missing title dois count:", len(missing_title_dois))

    for doi1, doi2 in zip(raw_dois, abstract_dois):
        if doi1 != doi2:
            missing_abstract_dois.append(doi1)
    if len(raw_dois) > len(abstract_dois):
        print("Mismatched abstract dois count:", len(raw_dois) - len(abstract_dois))
        missing_abstract_dois.extend(raw_dois[len(abstract_dois):])
    print("Missing abstract dois count:", len(missing_abstract_dois))

    # Save missing title dois to file
    with open('./processed/missing_title_dois.pkl', 'wb') as f:
        pickle.dump(missing_title_dois, f)

    # Save missing abstract dois to file
    with open('./processed/missing_abstract_dois.pkl', 'wb') as f:
        pickle.dump(missing_abstract_dois, f)    

    return missing_title_dois, missing_abstract_dois

def merge_data(titles, abstracts, title_weight=0.2):
    assert len(titles) == len(abstracts), "Data sizes do not match"

    merged_data = []
    for (title, title_embedding, title_doi), (abstract, abstract_embedding, abstract_doi) in zip(titles, abstracts):
        assert title_doi == abstract_doi, "Mismatched ids between datasets"
        
        combined_embedding = np.concatenate([title_embedding*title_weight, abstract_embedding*(1-title_weight)])
        merged_data.append((title, abstract, combined_embedding, title_doi))
    
    return merged_data

if __name__ == "__main__":
    # title_count, abstract_count = 0, 0
    # for i in range(1,23):
    #     print("Processing file: ", i)
    #     abstracts = load_pickle('./processed/abstract_chunks/abstracts_{}.pkl'.format(i))
    #     titles = load_pickle('./processed/title_chunks/ordered_titles_{}.pkl'.format(i))
        
    #     # Verify that the two pickles have the same dois
    #     verify_pickles(titles, abstracts)

    #     # Count the number of embeddings
    #     title_count += len(titles)
    #     abstract_count += len(abstracts)

    # print("Title count: ", title_count)
    # print("Abstract count: ", abstract_count)
    # Processed title count:  2200000
    # Processed abstract count:  2200000
    # I intentionally cut off at 2.2 million, even though there are 2.3 and 23 pkls, because the last have the cutoff issue. This way there's no alignment issues and we can just handle the last part later.

    # # Count raw pickle files
    # raw_abstracts =load_and_append_pkl_files('./embeddings/arxiv_abstracts/')
    # print("Raw abstract count: ", len(raw_abstracts))
    # del raw_abstracts
    # # Raw abstract count: 2254000.

    # raw_titles = load_and_append_pkl_files('./embeddings/arxiv_titles/')
    # print("Raw title count: ", len(raw_titles))
    # del raw_titles
    # # Raw title count: 2250000.

    # # Count lines from raw parquet file
    # raw_lines = len(load_parquet('./datasets/arxiv2M.parquet'))
    # print("Raw parquet count:", raw_lines)
    # # Raw parquet count: 2254198

    # # Find all missing dois from the processed dataset (not the raw) to the raw parquet file
    # missing_title_dois, missing_abstract_dois = find_and_save_missing_dois()
    # print("Missing title dois count:", len(missing_title_dois))
    # print("Missing abstract dois count:", len(missing_abstract_dois))
    # # Missing title dois count: 54198
    # # Missing abstract dois count: 54198


    # # Merge the two processed datasets
    # for i in range(1,23):
    #     print("Processing file: ", i)
    #     abstracts = load_pickle('./processed/abstract_chunks/abstracts_{}.pkl'.format(i))
    #     titles = load_pickle('./processed/title_chunks/ordered_titles_{}.pkl'.format(i))

    #     # Merge the two lists
    #     merged = merge_data(titles, abstracts, 0.2)

    #     # Save the merged list to file
    #     with open('./processed/merged_chunks/merged_{}.pkl'.format(i), 'wb') as f:
    #         pickle.dump(merged, f)

    # Verify data was merged correctly
    total_count = 0
    for i in range(1,23):
        print("Processing file: ", i)
        merged = load_pickle('./processed/merged_chunks/merged_{}.pkl'.format(i))
        
        print_pickle(merged[:3])

        # Count the number of embeddings
        print("Merged count: ", len(merged))
        total_count += len(merged)

    print("Total count: ", total_count)

 