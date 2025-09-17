import streamlit as st

# Set page configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv


#load env variables
load_dotenv()
groq_api_key=os.getenv("GROQ_KEY")
os.environ["HF_API_KEY"]=os.getenv("HF_API_KEY")


#load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("rag_dataset.csv")
    df.drop(columns=['_id', 'id', 'ups', 'subreddit', 'created_utc', 'num_comments', 'url', 'response'], inplace=True, errors='ignore')
    df["combined_text"] = df["title"] + " " + df["selftext"] + " " + df["comments"]
    return df
df=load_data()

#convert to documents
docs=[Document(page_content=text) for text in df['combined_text'].tolist() ]

#text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

#load embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings,)
retreiver=vectorstore.as_retriever()


#Load LLM
llm = ChatGroq(model="gemma2-9b-it")

# Define prompts
system_prompt = (
    "You are an AI assistant. Use the retrieved context to answer the question.\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, reformulate it "
    "as a standalone question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create retrieval and answering chain
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or initialize chat history for a given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create conversational RAG chain with history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Streamlit UI

st.title("🔍 AI-Powered RAG Chatbot")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user
query = st.chat_input("Ask me anything about AI...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get AI response
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )["answer"]

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)