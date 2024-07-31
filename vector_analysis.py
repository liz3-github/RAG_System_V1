import matplotlib.pyplot as plt
import numpy as np
from vector_store import VectorStore, load_embeddings
from sklearn.decomposition import PCA

# 加载数据
bid_vectors, bid_metadata = load_embeddings('bids_embedded.csv')

bid_store = VectorStore(len(bid_vectors[0]))
bid_store.add_vectors(bid_vectors, bid_metadata)

# 1. 向量在库中的分布
def plot_vector_distribution():
    # 使用PCA将向量降至2维以便可视化
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(bid_vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
    plt.title("向量在二维空间的分布")
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")

    # 随机选择几个点进行标注
    for i in np.random.choice(len(vectors_2d), 5, replace=False):
        plt.annotate(f"Bid {i}", (vectors_2d[i, 0], vectors_2d[i, 1]))

    plt.tight_layout()
    plt.show()

# 2. 搜索过程可视化
def plot_search_process():
    # 创建一个模拟的查询向量
    query_vector = np.random.rand(len(bid_vectors[0]))
    
    # 使用PCA将所有向量（包括查询向量）降至2维
    all_vectors = np.vstack([bid_vectors, [query_vector]])
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(all_vectors)

    # 分离bid向量和查询向量
    bid_vectors_2d = vectors_2d[:-1]
    query_vector_2d = vectors_2d[-1]

    # 计算与查询向量的距离
    distances = np.linalg.norm(bid_vectors_2d - query_vector_2d, axis=1)
    
    # 找出最近的k个向量
    k = 5
    nearest_indices = np.argsort(distances)[:k]

    plt.figure(figsize=(12, 8))
    
    # 绘制所有bid向量
    plt.scatter(bid_vectors_2d[:, 0], bid_vectors_2d[:, 1], alpha=0.3, label='Bid Vectors')
    
    # 绘制查询向量
    plt.scatter(query_vector_2d[0], query_vector_2d[1], color='red', s=100, label='Query Vector')
    
    # 绘制最近的k个向量
    plt.scatter(bid_vectors_2d[nearest_indices, 0], bid_vectors_2d[nearest_indices, 1], 
                color='green', s=100, label=f'Top {k} Nearest')

    # 绘制从查询向量到最近向量的线
    for idx in nearest_indices:
        plt.plot([query_vector_2d[0], bid_vectors_2d[idx, 0]], 
                 [query_vector_2d[1], bid_vectors_2d[idx, 1]], 
                 'k--', alpha=0.3)

    plt.title("向量搜索过程可视化")
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 运行函数以显示图表
plot_vector_distribution()
plot_search_process()