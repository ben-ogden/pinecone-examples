import glob
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Define the directory containing the Parquet files
directory = 'companies-large/data/embeddings/'

# Find all Parquet files in the directory
parquet_files = glob.glob(f'{directory}/*.parquet')

# Initialize an empty list to hold DataFrames
dataframes = []

# Load each Parquet file into a DataFrame and append to the list
for file_name in parquet_files:
    table = pq.read_table(file_name)
    df = table.to_pandas()
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Display basic information about the DataFrame
print("DataFrame Information:")
print(combined_df.info())

# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(combined_df.head())

# Display summary statistics of the DataFrame
print("\nSummary statistics of the DataFrame:")
print(combined_df.describe())

# Display the columns of the DataFrame
print("\nColumns in the DataFrame:")
print(combined_df.columns)

# Exclude the 'embedding' column for unique and duplicate calculations
columns_to_check = combined_df.columns.difference(['embedding'])

# Display the number of unique values in each column
print("\nNumber of unique values in each column:")
unique_counts = combined_df[columns_to_check].nunique()
print(unique_counts)

# Display the number of duplicate values in each column
print("\nNumber of duplicate values in each column:")
duplicate_counts = combined_df[columns_to_check].apply(lambda x: x.duplicated().sum())
print(duplicate_counts)

# Extract the embeddings
embeddings = np.vstack(combined_df['embedding'].values)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne.fit_transform(embeddings)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(embeddings)

# Plot the 3D embeddings with cluster coloring
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=clusters, cmap='viridis', s=5, alpha=0.7)
ax.set_title('3D Visualization of Embeddings with Clusters using t-SNE')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
plt.colorbar(sc)
plt.show()