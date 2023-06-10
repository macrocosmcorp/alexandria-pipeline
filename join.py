import os
import pickle
import pandas as pd


def load_and_append_pkl_files(directory_path):
    data = []
    print(f'Start loading pkl files from {directory_path}...')
    # Traverse the directory
    files = []
    for root, dirs, files_in_dir in os.walk(directory_path):
        files.extend([os.path.join(root, file) for file in files_in_dir if file.startswith(
            'embeddings_') and file.endswith('.pkl')])
    # Sort files based on id
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

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

    # Save the data to a single .pkl file
    with open(os.path.join(output_dir, f'{output_file}.pkl'), 'wb') as f:
        # Write the list to a new .pkl file
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# Directory path containing .pkl files
directory_path = './embeddings/'

# Output directory
output_dir = directory_path

# Output file
output_file = 'ill_opinions'

# Load and append pkl files
data = load_and_append_pkl_files(directory_path)

# Turn the data into pd.DataFrame
df = pd.DataFrame(data, columns=['opinion', 'embedding', 'id'])

# Turn id column into int from tensor
df['id'] = df['id'].apply(lambda x: int(x.item()))

# Drop duplicates and null values
df.drop_duplicates(subset=['opinion'], inplace=True)
df.dropna(inplace=True)

# Sort the dataframe by id
df.sort_values(by=['id'], inplace=True)

# Print the first 10 rows
print(df.head(10))

# Turn dataframe back to list
data = df.values.tolist()
print(data[:1])

# Save the data to a single .pkl file
save_to_pkl(data, output_dir, output_file)

print('Finished processing.')
