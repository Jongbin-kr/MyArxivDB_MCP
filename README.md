# 🧠 BKMS: Organizing Research Papers with MCP Server

This repository provides a MCP server to **organize personal research papers** using an **MCP (Modular Command Platform)** server and **semantic search**. The platform enables efficient paper retrieval, automatic project assignment, and assistance in writing literature review sections using LLMs.

This server is mainly designed for Claude Desktop but may also work well with other MCP clients.


---
## ⚡️ Quickstart

> The following quickstart guide is based on an Apple Silicon MacBook.

1. Install uv
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  
2. clone the repository
  ```bash
  git clone https://github.com/Jongbin-kr/MyArxivDB_MCP.git
  ```
  
3. Intsall dependencies & activate the virtual environment
  ```bash
  uv sync
  source /.venv/bin/activate
  ```
  
4. set up environment variables at `.env` file.
  ```
  # .env
  PINECONE_API_KEY = "YOUR PINECONE_API_KEY"
  DB_NAME = "YOUR_DB_NAME"
  DB_USER = "YOUR_DB_USERNAME"
  DB_PASSWORD = "YOUR_DB_PASSWORD"
  DB_HOST = "localhost"
  DB_PORT = 4444
  ```

5. Intsall MCP server at Claude Desktop
   ```bash
   mcp install server.py
   ```

6. Done! The Claude desktop app will automatically detect the MCP server and you can start using i!




---

## 📌 Motivation

Researchers frequently accumulate large numbers of papers but lack tools to systematically organize them by topic or project. BKMS aims to:

- Automatically assign new papers to relevant projects using embeddings
- Allow semantic search for project-specific literature
- Assist in drafting sections like “Related Work” using LLMs

---

## 🛠️ Main functions

Our MCP server supports the following core capabilities:

- **Crawling** metadata and PDFs from arXiv using ID or URL using ArXiv API
- **Embedding** abstracts using Pinecone API(`llama-text-embed-v2`)
- **Storing** papers and projects in a PostgreSQL + pgvector DB
- **Generating** "Related Work" sections via LLM prompts

---

## 🎥 Workflow & Demo video
You can see our PPT and demo video in assets folder.

Brief overview of our project workflow and DB schema is as follows. 


---

## 👨‍👩‍👧‍👦 Team

- 박연진
- 원종빈
- 정예준