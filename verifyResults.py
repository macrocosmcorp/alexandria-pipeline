import os
import pickle
import glob

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def print_pickle_data(data):
    for batch in data:
        abstract, embedding, doi = batch
        print("Abstract:", abstract)
        print("Embedding:", embedding)
        print("DOI:", doi)
        print()

def load_and_print_pickle(pickle_path):
    data = load_pickle(pickle_path)
    print_pickle_data(data)

if __name__ == "__main__":
    print("Starting...")
    directory_path = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(os.path.join(directory_path, 'embeddings_*.pkl')):
        print("Loading and printing pickle file:", file_path)
        load_and_print_pickle(file_path)