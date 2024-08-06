#
# Upsert data and inspect headers to ensure that data is indexed before querying it.
#
import requests
import time

# Set up Pinecone API key and index name
API_KEY = 'CHANGE_ME'
HOST = 'https://fresh-test-e3vcj9z.svc.aped-4627-b74a.pinecone.io'

# Example vector data for upsert
upsert_data = {
    'vectors': [
        {
            'id': 'example-vector-1',
            'values': [0.1, 0.2, 0.3, 0.4]
        },
        {
            'id': 'example-vector-2',
            'values': [0.5, 0.6, 0.7, 0.8]
        },
        {
            'id': 'example-vector-3',
            'values': [0.9, 1.0, 1.1, 1.2]
        }
    ]
}

# Upsert and note the request lsn
upsert_response = requests.post(f"{HOST}/vectors/upsert", json=upsert_data, headers={
    'Accept': 'application/json',
    'Api-Key': API_KEY,
    'Content-Type': 'application/json',
    'X-Pinecone-API-Version': '2024-07'
})
request_lsn = int(upsert_response.headers.get('x-pinecone-request-lsn', 0))
print(f'Upsert Request LSN: {request_lsn}')


query_data = {
    'vector': [0.3, 0.3, 0.3, 0.3],
    'topK': 2,
    'includeValues': True
}

max_indexed_lsn = 0
while request_lsn > max_indexed_lsn:
    # Query and note the max lsn
    query_response = requests.post(f"{HOST}/query", json=query_data, headers={
        'Api-Key': API_KEY,
        'Content-Type': 'application/json',
        'X-Pinecone-API-Version': '2024-07'
    })
    max_indexed_lsn = int(query_response.headers.get(
        'x-pinecone-max-indexed-lsn', 0))
    print(f'Max Indexed LSN: {max_indexed_lsn}')
    time.sleep(1)

print('Upserted data is now searchable.')
