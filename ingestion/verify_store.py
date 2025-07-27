import sqlite3
import pandas as pd
import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

# === Load environment ===
env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is not set. Check your .env file in /config.")

# Set environment variable for Chroma Gemini embedding
os.environ["CHROMA_GOOGLE_GENAI_API_KEY"] = api_key

# === Paths ===
SQLITE_DB = "db/sqlite/structured.db"      # ✅ Matches your file_ingestor target
VECTOR_DIR = "db/vectorstore"              # ChromaDB directory

# === Verify SQLite ===
print("\n🔍 Verifying Structured Data (SQLite)...\n")
if not os.path.exists(SQLITE_DB):
    print("❌ Structured DB not found.")
else:
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    print("📁 Tables:", tables)

    # Print metadata table
    if "file_metadata" in tables:
        print("\n📌 File Metadata:")
        df = pd.read_sql("SELECT * FROM file_metadata", conn)
        print(df.to_markdown(index=False))

    # Preview contents of each structured table
    for tbl in tables:
        if tbl != "file_metadata":
            print(f"\n📊 Preview of `{tbl}`:")
            try:
                df = pd.read_sql(f"SELECT * FROM {tbl} LIMIT 5", conn)
                print(df.to_markdown(index=False))
            except Exception as e:
                print(f"⚠️ Error reading {tbl}: {e}")

    conn.close()

# === Verify Vector Store ===
print("\n🔍 Verifying Unstructured Data (ChromaDB)...\n")
if not os.path.exists(VECTOR_DIR):
    print("❌ ChromaDB directory not found.")
else:
    try:
        # Initialize Chroma client with embedding
        embedding_func = GoogleGenerativeAiEmbeddingFunction(
            model_name="models/embedding-001",
            api_key=api_key
        )
        chroma_client = chromadb.PersistentClient(path=VECTOR_DIR)

        # === Check unstructured data ===
        collection = chroma_client.get_or_create_collection(
            name="unstructured_data",
            embedding_function=embedding_func
        )

        count = collection.count()
        print(f"📦 ChromaDB contains {count} embedded chunks.")

        if count:
            print("\n🧠 Sample Semantic Query: 'Give summary'")
            results = collection.query(
                query_texts=["Give summary"],
                n_results=3
            )
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                print(f"\n🔹 Match {i+1}:\n{doc[:300]}...\n")

        # === Check metadata index ===
        meta_collection = chroma_client.get_or_create_collection(
            name="metadata_index",
            embedding_function=embedding_func
        )

        meta_count = meta_collection.count()
        print(f"\n📎 Metadata Index contains {meta_count} entries.")
        if meta_count:
            meta_results = meta_collection.get(include=["documents", "metadatas"])
            for i in range(min(3, meta_count)):
                print(f"\n🔹 Metadata Entry {i+1}:")
                print("  Document:", meta_results["documents"][i])
                print("  Metadata:", meta_results["metadatas"][i])

    except Exception as e:
        print("❌ Error accessing ChromaDB:", e)

def get_metadata_table(sqlite_path):
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != "file_metadata"]

    metadata = {}

    for table in tables:
        # Get column names
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]

        # Get description from file_metadata
        cursor.execute("SELECT ai_description FROM file_metadata WHERE table_name = ? LIMIT 1", (table,))
        result = cursor.fetchone()
        description = result[0] if result else ""

        metadata[table] = {
            "columns": columns,
            "ai_description": description
        }

    conn.close()
    return metadata

if __name__ == "__main__":
    metadata = get_metadata_table(SQLITE_DB)
    print("\n📌 Parsed Metadata:")
    for table, meta in metadata.items():
        print(f"\n📊 {table}")
        print(f"  Description: {meta['ai_description']}")
        print(f"  Columns: {meta['columns']}")

print("\n✅ Verification complete.")
