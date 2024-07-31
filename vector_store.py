import faiss
import numpy as np
import csv
import ast

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []

    def add_vectors(self, vectors, metadata):
        self.index.add(np.array(vectors).astype('float32'))
        self.data.extend(metadata)

    def search(self, query_vector, k):
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k)
        return [(self.data[i], distances[0][j]) for j, i in enumerate(indices[0])]

def load_embeddings(csv_file):
    vectors = []
    metadata = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            embedding = ast.literal_eval(row['embedding'])
            vectors.append(embedding)
            # 保留所有原始字段
            metadata.append({k: v for k, v in row.items() if k != 'embedding'})
    return vectors, metadata

# 使用示例
capability_vectors, capability_metadata = load_embeddings('capability_statements_embedded.csv')
bid_vectors, bid_metadata = load_embeddings('bids_embedded.csv')

capability_store = VectorStore(len(capability_vectors[0]))
capability_store.add_vectors(capability_vectors, capability_metadata)

bid_store = VectorStore(len(bid_vectors[0]))
bid_store.add_vectors(bid_vectors, bid_metadata)