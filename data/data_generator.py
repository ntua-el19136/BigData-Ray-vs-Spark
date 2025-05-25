import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from faker import Faker
import sys

faker = Faker()

def generate_mixed_data_chunk(chunk_size):
    numeric_features, labels = make_classification(
        n_samples=chunk_size,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        random_state=123
    )

    numeric_df = pd.DataFrame(numeric_features, columns=[f'feature_{i+1}' for i in range(4)])

    categorical_df = pd.DataFrame({
        'categorical_feature_1': np.random.randint(1, chunk_size, size=chunk_size),
        'categorical_feature_2': np.random.randint(1, chunk_size // 2, size=chunk_size),
        'word': [faker.word() for _ in range(chunk_size)]
    })

    df = pd.concat([numeric_df, categorical_df], axis=1)
    df['label'] = labels
    return df

def create_mixed_large_csv(num_samples, chunk_size=1000):
    header = [f'feature_{i+1}' for i in range(4)] + ['categorical_feature_1', 'categorical_feature_2', 'word', 'label']
    print(','.join(header))

    for _ in range(0, num_samples, chunk_size):
        chunk_size_actual = min(chunk_size, num_samples)
        df = generate_mixed_data_chunk(chunk_size_actual)
        df.to_csv(sys.stdout, index=False, header=False)
        sys.stdout.flush()
        num_samples -= chunk_size_actual

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_dataset.py <num_samples>")
        sys.exit(1)

    num_samples = int(sys.argv[1])
    create_mixed_large_csv(num_samples)
