import os
from capability_statement_preprocessing import process_pdfs
from Capability_statement_embedding import generate_embeddings as generate_capability_embeddings
from bids_embedding import process_bids
from vector_store import VectorStore, load_embeddings
from matcher import find_matches
import csv
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-proj-UufKQEk1HstT6bqe6XgJT3BlbkFJr57NCtjRRasuOIVYJrky")

def main():
    # 步骤1：处理PDF文件（如果还没处理）
    pdf_paths = ['HOH Company Firm Overview One Page.pdf']
    capability_processed_file = 'capability_statements_processed.csv'
    if not os.path.exists(capability_processed_file):
        process_pdfs(pdf_paths, capability_processed_file)
        print(f"Processing complete. Output saved to {capability_processed_file}")

    # 步骤2：生成capability statements的嵌入向量
    capability_embedded_file = 'capability_statements_embedded.csv'
    if not os.path.exists(capability_embedded_file):
        generate_capability_embeddings(capability_processed_file, capability_embedded_file)
        print(f"Embeddings generated and saved to {capability_embedded_file}")

    # 步骤3：处理和生成bids的嵌入向量
    bids_input_file = 'Scraping_demo_results.csv'
    bids_embedded_file = 'bids_embedded.csv'
    if not os.path.exists(bids_embedded_file):
        process_bids(bids_input_file, bids_embedded_file)
        print(f"Bid data processed and saved to {bids_embedded_file}")

    # 步骤4：创建向量存储
    capability_vectors, capability_metadata = load_embeddings(capability_embedded_file)
    bid_vectors, bid_metadata = load_embeddings(bids_embedded_file)

    capability_store = VectorStore(len(capability_vectors[0]))
    capability_store.add_vectors(capability_vectors, capability_metadata)

    bid_store = VectorStore(len(bid_vectors[0]))
    bid_store.add_vectors(bid_vectors, bid_metadata)

    # 步骤5：查找匹配
    matches = find_matches(capability_store, bid_store)

    # 步骤6：输出结果
    output_file = 'matches.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        fieldnames = ['Company', 'Bid_Number', 'Bid_Name', 'Bid_Description', 'Status', 'Category', 'Due_Date', 'Detail_Link', 'Euclidean_Distance', 'Similarity_Score', 'Analysis_Summary']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    print(f"Matching process complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()