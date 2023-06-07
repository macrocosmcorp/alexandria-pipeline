import pickle


def load_and_print_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        print(f'Loaded {pickle_path}, printing data...')
        for item in data:
            print(item)
            print('\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    instance_index = parser.add_argument('--instance_index', type=int, required=True)
    run_index = parser.add_argument('--run_index', type=int, required=True)
    args = parser.parse_args()

    instance_index = args.instance_index
    run_index = args.run_index

    load_and_print_pickle(f'embeddings/instance_{instance_index}/embeddings_{run_index}.pkl')
