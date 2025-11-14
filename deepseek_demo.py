import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import logging
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_word_document(file_path):
    doc = Document(file_path)
    return [paragraph.text for paragraph in doc.paragraphs]

def split_into_chunks(paragraphs, max_chunk_size=2000):
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_embeddings(text, client, dimension=128):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Summarize the following text concisely:"},
                {"role": "user", "content": text}
            ],
            stream=False
        )
        summary = response.choices[0].message.content
        embedding = np.array([ord(c) for c in summary]).astype(np.float32)
        if len(embedding) < dimension:
            embedding = np.pad(embedding, (0, dimension - len(embedding)), 'constant')
        return embedding[:dimension]
    except Exception as e:
        logging.error(f"Error creating embedding: {str(e)}")
        return np.zeros(dimension, dtype=np.float32)

def create_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_similar(query_embedding, index, k=3):
    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return I[0]

def save_faiss_index(index, chunks, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump({'index': faiss.serialize_index(index), 'chunks': chunks}, f)
    logging.info(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        index = faiss.deserialize_index(data['index'])
        chunks = data['chunks']
    logging.info(f"FAISS index loaded from {file_path}")
    return index, chunks

def main():
    load_dotenv()
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL"))

    doc_path = "鲁滨逊漂流记第二部英文文本_265-292.docx"
    index_path = "faiss_index.pkl"

    if os.path.exists(index_path):
        index, chunks = load_faiss_index(index_path)
    else:
        # Read the Word document
        paragraphs = read_word_document(doc_path)
        logging.info(f"Read {len(paragraphs)} paragraphs from the document.")

        # Split content into chunks
        chunks = split_into_chunks(paragraphs)
        logging.info(f"Split content into {len(chunks)} chunks.")

        # Create embeddings for each chunk
        logging.info("Creating embeddings for chunks...")
        chunk_embeddings = []
        for chunk in tqdm(chunks):
            embedding = create_embeddings(chunk, client)
            chunk_embeddings.append(embedding)

        # Create FAISS index
        index = create_faiss_index(chunk_embeddings)
        logging.info("FAISS index created successfully.")

        # Save the index and chunks
        save_faiss_index(index, chunks, index_path)

    # Generate a question about Robinson Crusoe
    question = "What attitude of Robinson Crusoe is depicted about China?"

    # Create embedding for the question
    question_embedding = create_embeddings(question, client)

    # Search for similar content
    similar_indices = search_similar(question_embedding, index)

    # Combine relevant chunks
    context = " ".join([chunks[i] for i in similar_indices])

    # Generate answer using the DeepSeek model
    logging.info("Generating answer...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about Robinson Crusoe."},
            {"role": "user", "content": f"Based on the following context from Robinson Crusoe, answer this question: {question}\n\nContext: {context}"},
        ],
        stream=False
    )

    print(f"Question: {question}")
    print(f"Answer: {response.choices[0].message.content}")

if __name__ == "__main__":
    main()
