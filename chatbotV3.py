import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from ingestion.verify_store import get_metadata_table

# --- Load environment ---
env_path = os.path.join(os.path.dirname(__file__), "config", ".env")
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set.")

# --- Paths ---
SQLITE_DB = "db/sqlite/structured.db"
VECTOR_DB_DIR = "db/vectorstore"

# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# --- Add memory to LLM for unstructured queries ---
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# --- Embeddings & VectorStores ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name="unstructured_data"
)

metadata_index = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name="metadata_index"
)

# --- Metadata from SQL ---
metadata_cache = get_metadata_table(SQLITE_DB)

# --- Functions ---
def classify_query_scores(user_input: str) -> tuple[str, float]:
    meta_results = metadata_index.similarity_search_with_score(user_input, k=1)
    unstruct_results = vectorstore.similarity_search_with_score(user_input, k=1)

    meta_score = meta_results[0][1] if meta_results else 1.0
    unstruct_score = unstruct_results[0][1] if unstruct_results else 1.0

    if meta_score < unstruct_score:
        meta_type = meta_results[0][0].metadata.get("source_type", "structured")
        return (meta_type, meta_score)
    else:
        return ("unstructured", unstruct_score)

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
        f"You are writing SQLite for table joins and subqueries if needed.\n"
        f"Table: {table}\nDescription: {meta['ai_description']}\n"
        f"Columns: {meta['columns']}\nPreview:\n{sample.to_string(index=False)}\n\n"
        f"Write an SQLite query to answer: '{user_input}'. Return only SQL."
    )
    return llm.invoke(prompt).content.strip()

def clean_sql(sql: str) -> str:
    return (
        sql.replace("```sql", "")
           .replace("```", "")
           .lstrip("sql\n")
           .strip("` ")
    )

def run_sql(sql: str) -> str:
    try:
        sql = clean_sql(sql)
        df = pd.read_sql(sql, sqlite3.connect(SQLITE_DB))
        if df.empty:
            return "No data found."
        return tabulate(df, headers="keys", tablefmt="grid")
    except Exception as e:
        return f"SQL error: {e}"

def search_documents(user_input: str, file_name: str = None) -> tuple[str, str]:
    filter_arg = {"filename": file_name} if file_name else None
    docs = vectorstore.similarity_search(query=user_input, k=5, filter=filter_arg)
    if not docs:
        docs = vectorstore.similarity_search(query=user_input, k=5)
    if not docs:
        return None, None
    content = "\n\n".join([d.page_content for d in docs])
    file_name_used = docs[0].metadata.get("filename")
    return content, file_name_used

# --- Main Loop ---
def main():
    RELEVANCE_THRESHOLD = 0.7

    print("🤖 Hello! I’m your Data Assistant.\n\nI can help you with:\n• Sales & product analysis (structured data)\n• Resume & document search (unstructured files)\nJust ask your question, and I’ll do my best to help!\n(Type 'exit' to quit.)\n")

    while True:
        try:
            raw_input = input("You: ").strip()
            lower_input = raw_input.lower()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            memory.clear()
            break

        if lower_input in ["exit", "quit"]:
            print("👋 Okay, see you next time!")
            memory.clear()
            break

        if not raw_input:
            continue

        if lower_input in ["hi", "hello", "hey"]:
            print("\n🤖 Hello! How can I assist you today?\n")
            continue
        elif lower_input in ["how are you", "how are you doing"]:
            print("\n🤖 I'm just a bot, but I'm ready to help! 😊 What can I do for you today?\n")
            continue
        elif lower_input in ["who are you", "what can you do", "help"]:
            print("\n🤖 I’m your AI assistant!\nI can:\n• Analyze sales/orders data\n• Extract answers from resumes or docs\nJust ask!\n")
            continue

        qtype, score = classify_query_scores(raw_input)

        if score > RELEVANCE_THRESHOLD:
            print("\n🤔 I'm not sure how to answer that. I can only answer questions about sales data or search my document library. Please try rephrasing your question.\n")
            continue

        if qtype == "structured":
            table = identify_table(raw_input)
            if not table:
                print("⚠️ Could not identify a table—switching to document search...\n")
                qtype = "unstructured"
            else:
                sql = generate_sql(raw_input, table)
                result = run_sql(sql)
                if result.startswith("SQL error") or result == "No data found.":
                    print("Initial query failed, attempting to rephrase and retry...")
                    retry_input = paraphrase(raw_input)
                    sql = generate_sql(retry_input, table)
                    result = run_sql(sql)
                print(f"\n📄 SQL: {sql}\n\n📊 Result:\n{result}\n")
                continue

        if qtype == "unstructured":
            mentioned = None
            all_metas = vectorstore._collection.get()["metadatas"]
            for meta in all_metas:
                fn = meta.get("filename")
                if fn and fn.lower() in lower_input:
                    mentioned = fn
                    break

            content, file_name_used = search_documents(raw_input, file_name=mentioned)
            if not content:
                print("⚠️ No relevant documents found.\n")
                continue

            final_prompt = (
                f"You are answering based on a document titled '{file_name_used}'.\n"
                f"Assume the document belongs to that person if the name is in the title.\n"
                f"Answer the question in a short and precise way using only the information in the context.\n"
                f"If the answer is not found, reply: 'I don't know.'\n\n"
                f"Context:\n{content}\n\nQuestion: {raw_input}"
            )
            resp = conversation.predict(input=final_prompt)
            print(f"\n📄 Answer:\n{resp.strip()}\n")

if __name__ == "__main__":
    main()