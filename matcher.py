from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(api_key="sk-proj-UufKQEk1HstT6bqe6XgJT3BlbkFJr57NCtjRRasuOIVYJrky")

def analyze_match(capability_statement, bid_description, euclidean_distance):
    prompt = f"""
    Company Capability Statement:
    {capability_statement}

    Potential Contract Bid:
    {bid_description}

    Euclidean Distance: {euclidean_distance}

    Analyze how well this contract bid matches the company's capabilities. 
    Provide:
    1. A similarity score from 0 to 100.
    2. A brief analysis summary (maximum 50 words) explaining the overall match and key reasons for the score.

    Format your response as follows:
    Similarity Score: [score]
    Analysis Summary: [brief summary]
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing matches between company capabilities and contract bids."},
            {"role": "user", "content": prompt}
            
        ],
        temperature=0
    )

    return response.choices[0].message.content

def find_matches(capability_store, bid_store, top_k=5):
    matches = []
    for i, capability in enumerate(capability_store.data):
        capability_vector = capability_store.index.reconstruct(i)
        similar_bids = bid_store.search(capability_vector, top_k)
        
        for bid, distance in similar_bids:
            analysis = analyze_match(capability['Capability_Statement'], bid['bid_description'], distance)
            
            # 解析GPT的输出
            similarity_score = ""
            analysis_summary = ""
            for line in analysis.split("\n"):
                if line.startswith("Similarity Score:"):
                    similarity_score = line.split(":")[1].strip()
                elif line.startswith("Analysis Summary:"):
                    analysis_summary = line.split(":", 1)[1].strip()
            
            matches.append({
                'Company': capability['Company'],
                'Bid_Number': bid['bid_number'],
                'Bid_Name': bid['bid_name'],
                'Bid_Description': bid['bid_description'],
                'Status': bid['status'],
                'Category': bid['category'],
                'Due_Date': bid['due_date'],
                'Detail_Link': bid['detail_link'],
                'Euclidean_Distance': distance,
                'Similarity_Score': similarity_score,
                'Analysis_Summary': analysis_summary
            })
    
    return matches

# 如果需要，可以添加一个主函数来测试
def main():
    # 这里可以添加测试代码
    pass

if __name__ == "__main__":
    main()