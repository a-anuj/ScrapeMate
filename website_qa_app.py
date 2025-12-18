#!/usr/bin/env python
# coding: utf-8


import os
from groq import Groq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from dotenv import load_dotenv
import trafilatura
load_dotenv()



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
        loader = WebBaseLoader(web_paths=(user_prompt_webaddress,))
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        document = text_splitter.split_documents(document)
        embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        db = FAISS.from_documents(document,embedding)

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context
        <context>
        {context}
        </context>
        Question : {input}""")
        document_chain = create_stuff_documents_chain(llm,prompt)

        retriever = db.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever,document_chain)
        response = retrieval_chain.invoke({"input" : user_prompt_question})
        st.write(response['answer'])

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

