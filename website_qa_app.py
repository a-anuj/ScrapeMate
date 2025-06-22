#!/usr/bin/env python
# coding: utf-8


import os
from groq import Groq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from dotenv import load_dotenv
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


with st.form("prompt-form"):
    user_prompt_webaddress = st.text_input("Enter the web address")
    user_prompt_question = st.text_input("Enter the question")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        loader = WebBaseLoader(web_paths=(user_prompt_webaddress,))
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        document = text_splitter.split_documents(document)
        embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

        db = Chroma.from_documents(document,embedding)

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context
        <context>
        {context}
        </context>
        Answer in **one or two short sentences**:
        Question : {input}""")
        document_chain = create_stuff_documents_chain(llm,prompt)

        retriever = db.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever,document_chain)
        response = retrieval_chain.invoke({"input" : user_prompt_question})
        st.write(response['answer'])








