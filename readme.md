# ⚡ ScrapeMate

**Paste a link. Ask anything. Get a smart answer — not a dumb search.**

ScrapeMate is your AI-powered website reader that answers questions *directly* from a web page using RAG (Retrieval-Augmented Generation). It's built with **LangChain** + **Groq**, and here's the kicker:

```text
🚫 This feature isn’t available for free in ChatGPT. But with ScrapeMate, you get it. Free. Local. Smart.
```

## 🌍 Try it Live

🔗 **Try ScrapeMate now** → [scrapematechatbot.streamlit.app](https://scrapematechatbot.streamlit.app)  
No signups. No fees. Just paste a URL and zap your doubts ⚡



## 🚀 Features

- 🔗 Paste any public webpage URL  
- 💬 Ask natural language questions about its content  
- ⚡  Uses Groq’s blazing-fast LLM for responses  
- 🧠 Powered by RAG — not generic pre-trained fluff  
- 🌈 Clean Streamlit UI with good vibes  

## 💡Real-World Use Cases
- Ask detailed questions about legal or privacy pages
- Extract product info from listings
- QA over technical documentation


## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** Groq  
- **Vector DB:** FAISS   
- **Web Loader:** LangChain's WebBaseLoader  
- **Embeddings:** HuggingFace  


## 📦 Setup Instructions

```bash
git clone https://github.com/a-anuj/scrapemate.git
cd scrapemate
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run website_qa_app.py
```