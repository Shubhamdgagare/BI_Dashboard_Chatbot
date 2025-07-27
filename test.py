import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from tabulate import tabulate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from ingestion.verify_store import get_metadata_table

# Load environment variables
load_dotenv("config/.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# Vectorstores
vectorstore = Chroma(
    persist_directory="db/vectorstore",
    embedding_function=embeddings,
    collection_name="unstructured_data"
)
metadata_index = Chroma(
    persist_directory="db/vectorstore",
    embedding_function=embeddings,
    collection_name="metadata_index"
)

# SQL Metadata
SQLITE_DB = "db/sqlite/structured.db"
metadata_cache = get_metadata_table(SQLITE_DB)

# Functions
def classify_query_scores(user_input):
    meta_results = metadata_index.similarity_search_with_score(user_input, k=1)
    unstruct_results = vectorstore.similarity_search_with_score(user_input, k=1)
    meta_score = meta_results[0][1] if meta_results else 1.0
    unstruct_score = unstruct_results[0][1] if unstruct_results else 1.0
    if meta_score < unstruct_score:
        return (meta_results[0][0].metadata.get("source_type", "structured"), meta_score)
    return ("unstructured", unstruct_score)

def identify_table(user_input):
    tables = list(metadata_cache.keys())
    prompt = f"Which table best matches this query? {tables}\n\nQuery: {user_input}\nReturn only table name."
    resp = llm.invoke(prompt).content.strip()
    return resp if resp in tables else None

def paraphrase(user_input):
    return llm.invoke(f"Rephrase this query more clearly: {user_input}").content.strip()

def generate_sql(user_input, table):
    meta = metadata_cache[table]
    sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 5", sqlite3.connect(SQLITE_DB))
    prompt = (
        f"You are writing SQLite for table joins and subqueries if needed.\n"
        f"Table: {table}\nDescription: {meta['ai_description']}\nColumns: {meta['columns']}\n"
        f"Preview:\n{sample.to_string(index=False)}\n\n"
        f"Write an SQLite query to answer: '{user_input}'. Return only SQL."
    )
    return llm.invoke(prompt).content.strip()

def run_sql(sql):
    try:
        sql = sql.replace("```sql", "").replace("```", "").lstrip("sql\n").strip("` ")
        df = pd.read_sql(sql, sqlite3.connect(SQLITE_DB))
        if df.empty:
            return "No data found."
        return "\n\n📄 SQL Query Result :\n\n" + tabulate(df, headers="keys", tablefmt="grid")
    except Exception as e:
        return f"SQL error: {e}"

def search_documents(user_input, file_name=None):
    filter_arg = {"filename": file_name} if file_name else None
    docs = vectorstore.similarity_search(query=user_input, k=5, filter=filter_arg)
    if not docs:
        docs = vectorstore.similarity_search(query=user_input, k=5)
    if not docs:
        return None, None
    content = "\n\n".join([d.page_content for d in docs])
    file_name_used = docs[0].metadata.get("filename")
    return content, file_name_used

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🧠 AI Chatbot")
st.markdown("""
Hello! I’m your Data Assistant

📊 Sales & product analysis  
📄 Resume & document search

Just ask your question below 👇
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Type your question...")
if query:
    st.session_state.chat_history.append(("user", query))

    if query.lower().strip() in ["hi", "hello", "hey"]:
        response = "Hello there! 👋 How can I assist you?"
    else:
        qtype, score = classify_query_scores(query)
        if score > 0.7:
            response = "🤔 I'm not sure how to answer that. Try rephrasing."
        elif qtype == "structured":
            table = identify_table(query)
            if not table:
                qtype = "unstructured"
            else:
                sql = generate_sql(query, table)
                result = run_sql(sql)
                if result.startswith("SQL error") or result == "No data found.":
                    retry_input = paraphrase(query)
                    sql = generate_sql(retry_input, table)
                    result = run_sql(sql)
                response = f"{result}"

        if qtype == "unstructured":
            mentioned = None
            all_metas = vectorstore._collection.get()["metadatas"]
            for meta in all_metas:
                fn = meta.get("filename")
                if fn and fn.lower() in query.lower():
                    mentioned = fn
                    break
            content, fname = search_documents(query, mentioned)
            if not content:
                response = "⚠️ No relevant documents found."
            else:
                prompt = (
                    f"You are answering based on document titled '{fname}'.\n"
                    f"Answer short and precisely. Say 'I don't know' if unsure.\n"
                    f"\nContext:\n{content}\n\nQuestion: {query}"
                )
                response = st.session_state.conversation.predict(input=prompt)

    st.session_state.chat_history.append(("assistant", response))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
