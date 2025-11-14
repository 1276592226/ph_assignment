import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document

def read_word_document(file_path):
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

import hashlib

def create_embeddings(text, client, dimension=128):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Summarize the following text in 10 key points, separated by semicolons:"},
            {"role": "user", "content": text}
        ],
        stream=False
    )
    summary = response.choices[0].message.content
    # Create a fixed-size embedding using SHA-256 hash
    hash_object = hashlib.sha256(summary.encode())
    hash_digest = hash_object.digest()
    return np.frombuffer(hash_digest, dtype=np.float32)[:dimension]

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_similar(query_embedding, index, k=1):
    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return I[0]

def main():
    load_dotenv()
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL"))

    # Read the Word document
    doc_path = "鲁滨逊漂流记第二部英文文本_265-292.docx"
    content = read_word_document(doc_path)

    # Create embeddings for the document content
    content_embedding = create_embeddings(content, client)

    # Create FAISS index
    index = create_faiss_index([content_embedding])

    # Generate a question about Robinson Crusoe
    question = "What challenges did Robinson Crusoe face on the island?"

    # Create embedding for the question
    question_embedding = create_embeddings(question, client)

    # Search for similar content
    similar_indices = search_similar(question_embedding, index)

    # Generate answer using the DeepSeek model
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about Robinson Crusoe."},
            {"role": "user", "content": f"Based on the following context from Robinson Crusoe, answer this question: {question}\n\nContext: {content}"},
        ],
        stream=False
    )

    print(f"Question: {question}")
    print(f"Answer: {response.choices[0].message.content}")

if __name__ == "__main__":
    main()
