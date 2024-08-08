import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pinecone.grpc import PineconeGRPC as Pinecone
from retry import retry
import time
import uuid

# Define a function to retry with exponential backoff
@retry(ValueError, tries=5, delay=1, backoff=2)
def embed (strings_to_embed):
    return pc.inference.embed(
        model="multilingual-e5-large",  # dimensionality=1024
        inputs=strings_to_embed,
        parameters={
            "input_type": "passage",  # or "query"
            "truncate": "END"
        }
    )

# Init pinecone client
pc = Pinecone(api_key="a089d378-c48b-4197-93f8-466310cdf910")  # This is gRPC client aliased as "Pinecone"

# Load the dataset 
df = pd.read_csv('companies-large/data/companies.csv')
#print(df.describe())

total_rows = df.shape[0]
batch_size = 96  # Number of strings to embed in a single batch
total_batches = (total_rows + batch_size - 1) // batch_size  # Calculate total number of batches

file_name_template = 'companies-large/data/embeddings/output_batch_{}.parquet'
batches_per_file = 1200  # Number of batches to write to a single Parquet file
file_count = 0

# Keep track of the total number of tokens consumed
total_tokens_used = 0

# Record the start time for the entire process
total_start_time = time.time()

# Initialize an empty DataFrame to accumulate batches
accumulated_df = pd.DataFrame()

for current_batch in range(total_batches):
    start_idx = current_batch * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    batch_df = df.iloc[start_idx:end_idx].copy() 

    # Build strings to embed using list comprehension
    strings_to_embed = [
        ' '.join(
            str(value) for value in [
                row['name'], row['industry'], row['city'], row['state'], row['country_code']
            ] if value is not None
        )
        for row in batch_df.to_dict(orient='records')
    ]

    # Record the start time for the current batch
    batch_start_time = time.time()
    
    # Embed the batch of strings
    try:
        embedding_result =  embed(strings_to_embed)
    except Exception as e:
        print(f"Failed to get embedding for {strings_to_embed} after retries: {e}")
        continue
        
    # Convert EmbeddingsList to a list of lists
    vectors = [embedding['values'] for embedding in embedding_result.data]
    tokens_used = embedding_result.usage['total_tokens']

    if vectors is None:
        raise ValueError("Embedding process failed, vectors is None")
    
    # Check for None values in vectors and handle them
    for i, v in enumerate(vectors):
        if v is None:
            print(f"Failed to embed string: {strings_to_embed[i]}")
            vectors[i] = [0.0] * 1024  # Replace None with a zero vector of appropriate length

    # Ensure vectors is a list of lists and has the correct length
    if not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
        raise ValueError("Vectors is not a list of lists")

    if len(vectors) != len(batch_df):
        raise ValueError("Length of vectors does not match length of batch_df")

    # Add the vectors and unique identifier as new columns to batch_df
    batch_df.loc[:, 'embedding'] = vectors
    batch_df.loc[:, 'id'] = [str(uuid.uuid4()) for _ in range(len(batch_df))]

    # Accumulate the current batch DataFrame
    accumulated_df = pd.concat([accumulated_df, batch_df])

    # Determine the file name based on the record count
    if (current_batch + 1) % batches_per_file == 0:
        file_count += 1
        file_name = file_name_template.format(file_count)
        # Convert the DataFrame to a PyArrow table
        table = pa.Table.from_pandas(accumulated_df)
        # Write the table to a new Parquet file
        pq.write_table(table, file_name, compression='snappy')
        print(f"Wrote {table.num_rows} embeddings & original data to '{file_name}'")
        # Reset the accumulated DataFrame
        accumulated_df = pd.DataFrame()

    # Record the end time for the current batch and calculate the elapsed time
    batch_end_time = time.time()
    batch_elapsed_time = batch_end_time - batch_start_time

    # Update and print total tokens used
    total_tokens_used += tokens_used
    print(f"Processed batch {current_batch + 1}/{total_batches}, Tokens Used: {tokens_used}, Total Tokens Used: {total_tokens_used}, Batch Time: {batch_elapsed_time:.2f} seconds")

# Write any remaining records to a new Parquet file
if not accumulated_df.empty:
    file_count += 1
    file_name = file_name_template.format(file_count)
    table = pa.Table.from_pandas(accumulated_df)
    pq.write_table(table, file_name, compression='snappy')

# Record the end time for the entire process and calculate the total elapsed time
total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

print(f"Embeddings and original data saved to '{file_name}'")
print(f"Total Time Elapsed: {total_elapsed_time:.2f} seconds")