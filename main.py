import pandas as pd

chunk_size = 50000  # Number of rows per chunk
for i, chunk in enumerate(pd.read_csv('data/Final_Augmented_dataset_Diseases_and_Symptoms.csv', chunksize=chunk_size)):
    # Process each chunk here (e.g., filter rows, aggregate data)
    print(f"Processing chunk with {len(chunk)} rows.")
    # Example: Save processed chunks to a new CSV
    chunk.to_csv(f'data/processed_data{i+1}.csv', mode='a', header=False, index=False)