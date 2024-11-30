# conda create -p env python==11 -y
#env
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("RAG Model With Groq")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please Provide the most accurate response based on the question]
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Silent Embedding Initialization
if "vectors" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader = PyPDFDirectoryLoader("./data")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# User Input
prompt1 = st.text_input("Enter your Questions from Directory")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure Response Time
    import time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")

    # Display Answer
    st.write(response['answer'])

    # Display Document References
    with st.expander("Document Search References"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------")
