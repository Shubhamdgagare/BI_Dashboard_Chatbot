import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from ingestion.verify_store import get_metadata_table

# --- Load environment ---
env_path = os.path.join(os.path.dirname(__file__), "config", ".env")
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set.")

# --- Paths ---
SQLITE_DB = "db/sqlite/structured.db"
VECTOR_DB_DIR = "db/vectorstore/chroma_db"

# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# --- Initialize Embeddings and VectorStore ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)
vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name="unstructured_data"
)
# --- Metadata from SQL ---
metadata_cache = get_metadata_table(SQLITE_DB)

# --- Functions ---

def classify_query(user_input: str) -> str:
    prompt = (
        "Decide if the user query is about tabular data (structured) "
        "or textual documents (unstructured).\n\n"
        f"Query: {user_input}\n\nReturn only 'structured' or 'unstructured'."
    )
    resp = llm.invoke(prompt)
    return resp.content.strip().lower()

def paraphrase(user_input: str) -> str:
    resp = llm.invoke(f"Rephrase this query more clearly: {user_input}")
    return resp.content.strip()

def identify_table(user_input: str):
    tables = list(metadata_cache.keys())
    prompt = f"Which table best matches this query? {tables}\n\nQuery: {user_input}\nReturn only table name."
    resp = llm.invoke(prompt).content.strip()
    return resp if resp in tables else None

def generate_sql(user_input: str, table: str) -> str:
    meta = metadata_cache[table]
    sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", sqlite3.connect(SQLITE_DB))
    prompt = (
        f"Table: {table}\nDescription: {meta['ai_description']}\n"
        f"Columns: {meta['columns']}\nPreview:\n{sample.to_string(index=False)}\n\n"
        f"Write an SQLite query to answer: '{user_input}'. Return only SQL."
    )
    return llm.invoke(prompt).content.strip()

def run_sql(sql: str) -> str:
    try:
        df = pd.read_sql(sql, sqlite3.connect(SQLITE_DB))
        if df.empty:
            return "No data found."
        return tabulate(df, headers="keys", tablefmt="grid")
    except Exception as e:
        return f"SQL error: {e}"

def search_documents(user_input: str, file_name: str = None) -> str:
    filter_arg = {"filename": file_name} if file_name else None
    docs = vectorstore.similarity_search(query=user_input, k=3, filter=filter_arg)
    if not docs:
        return None
    return "\n\n".join([d.page_content for d in docs])

# --- Main Loop ---
def main():
    print("🤖 Chatbot ready! Type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break
        
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        qtype = classify_query(user_input)

        if qtype == "structured":
            table = identify_table(user_input)
            if not table:
                print("⚠️ Could not identify a table—switching to document search...\n")
                qtype = "unstructured"
            else:
                sql = generate_sql(user_input, table)
                result = run_sql(sql)
                if result.startswith("SQL error") or result == "No data found.":
                    retry = paraphrase(user_input)
                    sql = generate_sql(retry, table)
                    result = run_sql(sql)
                print(f"\n📄 SQL: {sql}\n\n📊 Result:\n{result}\n")
                continue

        if qtype == "unstructured":
            # detect file mention
            mentioned = None
            for table_meta in vectorstore._collection.get()["metadatas"]:
                fn = table_meta.get("filename")
                if fn and fn.lower() in user_input.lower():
                    mentioned = fn
                    break

            content = search_documents(user_input, file_name=mentioned)
            if not content:
                print("⚠️ No relevant documents found.\n")
                continue

            final_prompt = f"Based only on these snippets, answer: '{user_input}'\n\n{content}"
            resp = llm.invoke(final_prompt)
            print(f"\n📄 Answer:\n{resp.content.strip()}\n")

if __name__ == "__main__":
    main()
