import os
import pickle


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

# Directory path containing .pkl files
directory_path = './embeddings/arxiv_abstracts/'

# Output directory
output_dir = './abstract_chunks/'

# Output file
output_file = 'abstracts'

# Load and append pkl files
data = load_and_append_pkl_files(directory_path)

# Save the data to multiple .pkl files each containing up to 100k items
save_to_pkl(data, output_dir, output_file)

print('Finished processing.')
