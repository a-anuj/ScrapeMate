#!/usr/bin/env python
# coding: utf-8


import os
from groq import Groq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
import trafilatura
load_dotenv()
import langchain



client = Groq(api_key=os.getenv("GROQ_API_KEY"))




class GroqLLM:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, query):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": str(query)}]
        )
        return response.choices[0].message.content

# Initialize like this
llm = GroqLLM(client, "llama-3.3-70b-versatile")

st.title("🕸️ ScrapeMate")
st.markdown("###### Paste a link. Ask anything. Get a smart answer — not a dumb search")
st.markdown("---")
st.markdown("#### ⚡ Zap your doubts")


with st.form("prompt-form"):
    user_prompt_webaddress = st.text_input("🔗 Enter the web address")
    user_prompt_question = st.text_input("💬 Enter the question")
    col1,_,col2 = st.columns([3,2,2])
    with col1:
        submitted = st.form_submit_button("🔍Search")
    with col2:
        summary = st.form_submit_button("Summarize the page")
    
if submitted:
    # 1. Load webpage
    loader = WebBaseLoader(web_paths=(user_prompt_webaddress,))
    documents = loader.load()

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(documents)

    # 3. Create embeddings + vector store
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    db = FAISS.from_documents(documents, embedding)

    # 4. Retrieve relevant chunks
    retriever = db.as_retriever()
    docs = retriever.invoke(user_prompt_question)

    # 5. Build context manually
    context = "\n\n".join(doc.page_content for doc in docs)

    # 6. Create prompt
    prompt = f"""
    Answer the following question based on the provided context.

    Context:
    {context}

    Question:
    {user_prompt_question}
    """

    # 7. Call LLM
    response = llm(prompt)

    # 8. Display answer
    st.write(response)


    if summary:
        url = user_prompt_webaddress
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            document = trafilatura.extract(downloaded)
        else:
            print("Failed to fetch content!")
        
        prompt = ChatPromptTemplate.from_template("""
        Summarize the whole text given below with bullet points wherever necessary  
        text : {document}""")

        formatted_prompt = prompt.format(document=document)

        print(document)
        response = llm(formatted_prompt)
        st.write(response)

