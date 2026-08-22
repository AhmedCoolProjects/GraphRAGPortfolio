import os
import sys
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Add parent directory to path to import zvec_store
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_PATH)

# Import our ZVec wrapper
from zvec_store import ZVecVectorStore

# 1. Configuration
DATA_PATH = os.path.join(BASE_PATH, "data")
ZVEC_DATA_PATH = os.path.join(BASE_PATH, "zvec_data")

# Load environment variables
load_dotenv()

# Configuration
HF_API_KEY = os.getenv("HF_API_KEY")
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "portfolio"

def create_vector_db():
    print(f"Loading data from {DATA_PATH}...")

    # 2. Load Documents
    documents = []
    files_to_ingest = []

    # Include .md files directly in DATA_PATH
    files_to_ingest.extend(glob.glob(os.path.join(DATA_PATH, "*.md"), recursive=False))

    # Include .md files in subdirectories of DATA_PATH, excluding 'docs'
    for root, dirs, files in os.walk(DATA_PATH):
        if "docs" in dirs:
            dirs.remove("docs")

        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                if file_path not in files_to_ingest:
                    files_to_ingest.append(file_path)
    
    # Filter out files within any 'docs' subdirectory
    filtered_files_to_ingest = []
    for file_path in files_to_ingest:
        if "/docs/" not in file_path:
            filtered_files_to_ingest.append(file_path)

    print(f"   Found {len(filtered_files_to_ingest)} .md files for ingestion.")
    
    for file_path in filtered_files_to_ingest:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            loaded_docs = loader.load()
            
            for doc in loaded_docs:
                doc.metadata["source"] = file_path
            
            documents.extend(loaded_docs)
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
    
    if not documents:
        print("❌ No documents found! Make sure your folders exist and contain .md files.")
        return
    
    # 3. Split Text (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=list(set([
            "\n\n",
            "\n",
            "!",
            ".",
            "?",
            ",",
            " ",
            "",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ]))
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split {len(documents)} files into {len(chunks)} chunks.")

    # 4. Initialize Embeddings Model
    print("🧠 Initializing embeddings (HuggingFace Inference API)...")
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not found in environment variables. Please set it.")
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model=HF_EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=HF_API_KEY
    )

    # 5. Create ZVec Collection
    print(f"💾 Creating ZVec collection '{COLLECTION_NAME}' at {ZVEC_DATA_PATH}...")
    
    vector_db = ZVecVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=ZVEC_DATA_PATH,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Success! Your knowledge base is ready in ZVec.")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Location: {ZVEC_DATA_PATH}")
    print(f"   Documents: {len(chunks)} chunks")


if __name__ == "__main__":
    create_vector_db()