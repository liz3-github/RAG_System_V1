import csv
from openai import OpenAI
import os

# 确保设置了 OPENAI_API_KEY 环境变量
client = OpenAI(api_key="sk-proj-UufKQEk1HstT6bqe6XgJT3BlbkFJr57NCtjRRasuOIVYJrky")

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def generate_embeddings(input_csv, output_csv):
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['embedding']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            embedding = get_embedding(row['Capability_Statement'])
            row['embedding'] = embedding
            writer.writerow(row)

    print(f"Embeddings generated and saved to {output_csv}")

# 使用示例
input_file = 'capability_statements_processed.csv'
output_file = 'capability_statements_embedded.csv'

generate_embeddings(input_file, output_file)