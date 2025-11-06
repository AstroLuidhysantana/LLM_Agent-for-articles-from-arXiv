# PDF Agent — Interactive PDF Reader with AI, Figures, and Web Search

This project is a **Streamlit app** that allows users to upload PDF files (such as research articles or contracts) and interact with them through a conversational AI agent.  
It integrates with **Ollama** to use local LLMs (like LLaMA 3), and supports **text + image extraction**, **semantic search**, and **optional web lookups** to enhance responses.

---

## Features

-  Extracts and cleans text from PDFs (using **PyMuPDF** or **pdfplumber**)  
-  Detects and renders figures or full PDF pages as images  
-  Embeds PDF text with Ollama embeddings and retrieves relevant sections  
-  Allows chatting with the PDF content through an LLM hosted locally via Ollama  
-  Optional web search via Google (for additional context)  
-  Maintains a memory of the conversation inside the Streamlit session  

---

## Installation

### 1. Clone the repository

git clone https://github.com/yourusername/pdf-agent.git
cd pdf-agent

### 2. Instal the packages necessary to run it
pip install streamlit pdfplumber pymupdf numpy requests httpx beautifulsoup4 faiss-cpu

### 3. Installing Ollama on Linux 
curl -fsSL https://ollama.com/install.sh | sh
test your instalation
ollama serve

Download some LLM models to your Local machine

ollama pull llama3.1
ollama pull nomic-embed-text


### 3.  Run the code 

Streamlit run agent_pdf.py
