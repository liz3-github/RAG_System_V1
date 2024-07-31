import csv
import re
from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-proj-UufKQEk1HstT6bqe6XgJT3BlbkFJr57NCtjRRasuOIVYJrky")

def clean_text(text):
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 保留基本标点，但移除其他特殊字符
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]', '', text)
    return text.strip()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def process_bids(input_csv, output_csv):
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['embedding']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            combined_text = f"{row['bid_name']}. {row['bid_description']}"
            cleaned_text = clean_text(combined_text)
            
            embedding = get_embedding(cleaned_text)
            
            row['embedding'] = embedding
            writer.writerow(row)

    print(f"Bid data processed and saved to {output_csv}")

# 使用示例
input_file = 'Scraping_demo_results.csv'
output_file = 'bids_embedded.csv'

process_bids(input_file, output_file)