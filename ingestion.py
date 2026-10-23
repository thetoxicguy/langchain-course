import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    # Embeddings are computed by Pinecone itself, so no extra provider key is needed.
    # multilingual-e5-large produces 1024-dim vectors: the index must match.
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")

    print("Ingesting into Pinecone...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Done")
