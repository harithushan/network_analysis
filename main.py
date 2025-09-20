import pandas as pd

chunk_size = 50000
for i, chunk in enumerate(pd.read_csv('data/Final_Augmented_dataset_Diseases_and_Symptoms.csv', chunksize=chunk_size), start= 1):
    print(f"Processing chunk with {len(chunk)} rows.")
    chunk.to_csv(f'data/processed_data{i}.csv', mode='a', header=True, index=False)