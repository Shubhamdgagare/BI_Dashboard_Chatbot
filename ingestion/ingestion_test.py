import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# === Load environment variables ===
env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GOOGLE_API_KEY")

# === Set paths ===
STRUCTURED_DIR = "data/structured"
UNSTRUCTURED_DIR = "data/unstructured"
SQLITE_DB = "db/sqlite/structured.db"
VECTOR_DIR = "db/vectorstore"

# === Connect to SQLite ===
conn = sqlite3.connect(SQLITE_DB)
cursor = conn.cursor()

# === Create metadata table if not exists ===
cursor.execute("""
CREATE TABLE IF NOT EXISTS file_metadata (
    table_name TEXT,
    file_name TEXT,
    ai_description TEXT
)
""")
conn.commit()

# === Prepare metadata docs for Chroma ===
metadata_docs = []

# === Ingest Structured Files ===
for fname in os.listdir(STRUCTURED_DIR):
    fpath = os.path.join(STRUCTURED_DIR, fname)
    table_name = os.path.splitext(fname)[0].lower().replace(" ", "_")

    try:
        if fname.endswith(".csv"):
            df = pd.read_csv(fpath)
        elif fname.endswith(".xlsx"):
            df = pd.read_excel(fpath)
        else:
            continue

        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        ai_description = f"This table '{table_name}' comes from file '{fname}' and contains columns: {', '.join(df.columns)}."
        cursor.execute("INSERT INTO file_metadata VALUES (?, ?, ?)", (table_name, fname, ai_description))
        conn.commit()

        metadata_docs.append(Document(
            page_content=ai_description,
            metadata={
                "source_type": "structured",
                "table_name": table_name,
                "file_name": fname,
                "description": ai_description
            }
        ))

        print(f"✅ Indexed structured file: {fname} -> table: {table_name}")
    except Exception as e:
        print(f"❌ Failed to ingest {fname}: {e}")

# === Ingest Unstructured Files ===
documents = []
for fname in os.listdir(UNSTRUCTURED_DIR):
    fpath = os.path.join(UNSTRUCTURED_DIR, fname)

    try:
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(fpath)
        elif fname.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(fpath)
        elif fname.endswith(".txt"):
            loader = TextLoader(fpath)
        else:
            continue

        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "filename": fname,
                "chunk_index": i,
                "source_type": "unstructured"
            })
        documents.extend(docs)

        preview = docs[0].page_content[:300].strip().replace("\n", " ") if docs else "unavailable"
        unstructured_summary = f"Unstructured document '{fname}' likely contains information such as: {preview}..."

        metadata_docs.append(Document(
            page_content=unstructured_summary,
            metadata={
                "source_type": "unstructured",
                "file_name": fname,
                "description": unstructured_summary
            }
        ))

        print(f"✅ Indexed unstructured file: {fname}")
    except Exception as e:
        print(f"❌ Failed to load {fname}: {e}")

# === Embedding model ===
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# === Store document chunks in ChromaDB ===
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DIR,
        collection_name="unstructured_data"
    )
    print(f"✅ Stored {len(chunks)} unstructured chunks in ChromaDB")

# === Store file-level metadata index in ChromaDB ===
if metadata_docs:
    Chroma.from_documents(
        documents=metadata_docs,
        embedding=embedding_model,
        persist_directory=VECTOR_DIR,
        collection_name="metadata_index"
    )
    print(f"✅ Stored {len(metadata_docs)} metadata entries in ChromaDB")

print("\n🌟 Ingestion complete. All data stored in SQLite + ChromaDB.")
conn.close()
