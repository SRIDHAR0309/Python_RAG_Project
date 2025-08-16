import os
import requests
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
import openai
import hashlib

load_dotenv()

# === Config ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

# === Step 1: Clone Private PDF from GitHub ===
def download_pdf_from_github():
    github_pat = "USe your own PAT key"
    owner = "Your Github name"
    repo = "rag-app-cloner"
    branch = "main"
    pdf_path_in_repo = "sample.pdf"
    save_path = os.path.join("docs", "sample.pdf")

    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{pdf_path_in_repo}"
    headers = {"Authorization": f"token {github_pat}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"📥 PDF downloaded to {save_path}")
    else:
        raise Exception(f"Failed to download PDF: {response.status_code} {response.text}")

# === Step 2: Extract and Chunk Text ===
def extract_chunks_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"📄 Extracted {len(chunks)} text chunks")
    return chunks

# === Step 3: Embed Text with Azure OpenAI ===
def embed_text(texts):
    openai.api_type = "azure"
    openai.api_key = AZURE_OPENAI_API_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-05-15"

    response = openai.Embedding.create(
        input=texts,
        engine=AZURE_OPENAI_EMBED_DEPLOYMENT
    )
    return [d["embedding"] for d in response["data"]]

# === Step 4: Store Embeddings in Qdrant ===
def upload_to_qdrant(embeddings, texts):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # Create collection if it doesn't exist
    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
        print(f"✅ Collection '{QDRANT_COLLECTION_NAME}' already exists")
    except:
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
        )
        print(f"✅ Created Qdrant collection: {QDRANT_COLLECTION_NAME}")

    # Upload points
    points = []
    for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
        points.append(
            PointStruct(
                id=int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 12),
                vector=embedding,
                payload={"text": text}
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
    print(f"🚀 Uploaded {len(points)} vectors to Qdrant")

# === Main Execution ===
if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    download_pdf_from_github()
    chunks = extract_chunks_from_pdf("docs/sample.pdf")
    embeddings = embed_text(chunks)
    upload_to_qdrant(embeddings, chunks)
