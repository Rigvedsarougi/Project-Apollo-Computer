# sample_queries.py
"""
Sample queries and responses for testing the RAG pipeline.
This can be exported to a JSON/CSV file as required.
"""

sample_data = [
    {
        "query": "What are the key features of supervised learning?",
        "response": "Supervised learning is characterized by training models on labeled datasets where each example is paired with a target output. Key features include: 1) Using labeled data for training, 2) The ability to make predictions based on input features, 3) Support for both classification and regression tasks, 4) Performance evaluation through metrics like accuracy and precision, and 5) The requirement for high-quality labeled data which can be resource-intensive to obtain."
    },
    {
        "query": "Summarize the history of AI in 5 bullet points",
        "response": "1. 1950s: Alan Turing proposes the Turing Test and foundational concepts of machine intelligence\n2. 1956: John McCarthy coins the term 'Artificial Intelligence' at the Dartmouth Conference\n3. 1980s: Expert systems become commercially successful in various industries\n4. 1990s-2000s: Machine learning algorithms and neural networks gain prominence\n5. 2010s-Present: Deep learning revolution with breakthroughs in computer vision, natural language processing, and reinforcement learning"
    },
    {
        "query": "What is the difference between supervised and unsupervised learning with examples?",
        "response": "Supervised learning uses labeled data to train predictive models, while unsupervised learning finds patterns in unlabeled data.\n\nExamples of supervised learning:\n- Spam detection (emails labeled as spam/not spam)\n- House price prediction (features labeled with prices)\n- Image classification (images labeled with categories)\n\nExamples of unsupervised learning:\n- Customer segmentation (grouping customers by purchasing behavior)\n- Anomaly detection (identifying unusual patterns in data)\n- Topic modeling (discovering themes in document collections)"
    },
    {
        "query": "Explain reinforcement learning in simple words",
        "response": "Reinforcement learning is like training a pet with rewards. The AI agent learns by interacting with an environment and receiving rewards for good actions or penalties for bad ones. It tries different actions to discover which ones yield the best outcomes over time. For example, a program learning to play a game would experiment with various moves, earning points for successful actions that lead toward winning. The goal is to learn an optimal strategy that maximizes cumulative rewards."
    },
    {
        "query": "What are the main types of machine learning algorithms?",
        "response": "The three primary types of machine learning algorithms are:\n\n1. Supervised Learning: Learns from labeled training data to make predictions or decisions. Examples include classification and regression algorithms.\n\n2. Unsupervised Learning: Discovers patterns and relationships in unlabeled data. Examples include clustering and dimensionality reduction techniques.\n\n3. Reinforcement Learning: Learns through interaction with an environment by receiving rewards or penalties. Examples include game-playing AI and robotic control systems.\n\nAdditionally, there are hybrid approaches like semi-supervised learning (using both labeled and unlabeled data) and self-supervised learning (where the data provides its own supervision signals)."
    }
]

# Function to export to JSON
def export_to_json(filename="sample_queries.json"):
    import json
    with open(filename, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Sample queries exported to {filename}")

if __name__ == "__main__":
    export_to_json()
