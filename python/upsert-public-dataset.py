from pinecone import Pinecone
from pinecone_datasets import list_datasets, load_dataset

dataset = load_dataset("ANN_Fashion-MNIST_d784_euclidean")

pc = Pinecone(api_key="CHANGEME")
index = pc.Index("my-index")

for batch in dataset.iter_documents(batch_size=100):
    index.upsert(vectors=batch)