import asyncio
import os 
from typing import *
import pathlib

from dotenv import load_dotenv  # for loading environment variables from .env file
import wget, requests, feedparser         # for crawling arxiv papers

from mcp.server.fastmcp import FastMCP

from pinecone import Pinecone  ## import pinecone libraries for ermbedding passage and queries w/o resources

import psycopg2



##########################
## config for local env ##
##########################
PDF_DIRECTORY = "/Users/jongbin/MyArxivDB_MCP/data/pdfs"



## load env keys
load_dotenv()
PINECONCE_API_KEY = os.getenv("PINECONE_API_KEY") ## load environment variables from .env file

# Create an MCP server
server = FastMCP("MyArxivDB_MCP", dependencies=["wget", "requests", "feedparser",
                                              "pinecone",
                                              "psycopg2", 
                                              ])






## embed document and embed query functions using Pinecone's embedding service
pc = Pinecone(api_key=PINECONCE_API_KEY)

@server.tool()
def embed_document(document: str)->List[float]:
    """
    Embeds a document, NOT QUERY using Pinecone's embedding service. 
    For query embedding, we need to use embed_query function instead.

    
    Args:
        document (str): The text of the document to embed.
        
    Returns:
        List[float]: A list of embeddings for the document.
    """
    
    embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[document],
        parameters={"input_type": "passage", "truncate": "END"}
    )
        
    return embedding.data[0]['values']



@server.tool()
def embed_query(query: str) -> List[float]:
    """
    Embeds a query using Pinecone's embedding service.
    
    Args:
        query (str): The text of the query to embed.
        
    Returns:
        List[float]: A list of embeddings for the query.
    """
    
    embedding = pc.inference.embed(
        model="llama-text-embed-v2",   # embedding dimension: 1024
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"}
    )
    
    return embedding.data[0]['values']




## crawl arxiv papers from arxiv url using arxiv API
@server.tool()
async def crawl_arxiv_paper(arxiv_id_or_url: str) -> dict:
    
    ## clean arxiv URL into arxiv ID
    def _clean_id(raw: str) -> str:
        """If input was arxiv URL, not arxiv ID, pre-process the URL into arxiv ID."""
        raw = raw.strip()
        if raw.startswith("http"):
            raw = raw.rsplit("/", 1)[-1]

        return raw.replace("pdf", "").strip("/")
    
    arxiv_id = _clean_id(arxiv_id_or_url)
    
    
    ## Fetch metadata from arXiv Atom API using arxiv ID
    def _fetch_meta(arxiv_id: str) -> dict:
        """Crawl metadata with arXiv Atom API"""
        API = "http://export.arxiv.org/api/query?id_list={id}"
        feed = feedparser.parse(requests.get(API.format(id=arxiv_id), timeout=10).text)
        if not feed.entries:
            raise ValueError(f"Invalid arXiv ID: {arxiv_id}")

        e = feed.entries[0]
        return {
            "id":                arxiv_id,
            "title":             " ".join(e.title.split()),
            "abstract":          " ".join(e.summary.split()),
            "authors":          [a.name for a in e.authors],
            "published_date":     e.published[:10],
            "primary_category": e.arxiv_primary_category["term"],
            "categories":       [t["term"] for t in e.tags],
            "arxiv_url":        next(l.href for l in e.links if l.type == "application/pdf"),
            # "pdf_file_path":    None,    # it will be updated later
            # "embedding":        None
        }
    
    metadata = _fetch_meta(arxiv_id)
    
    
    ## Download the PDF file of the arxiv paper & add pdf filepath to metadata dict
    pdf_dir = pathlib.Path(PDF_DIRECTORY)
    pdf_dir.mkdir(parents=True, exist_ok=True)  # <-- 이 줄 추가

    pdf_file_path = pdf_dir / f"{arxiv_id.replace('.', '_')}.pdf"
    if not pdf_file_path.exists():
        wget.download(metadata["arxiv_url"], out=str(pdf_file_path))
    #     print(f"\nSaved to {pdf_file_path}")
    # else:
    #     print(f"PDF already exists: {pdf_file_path}")
    
    metadata["pdf_file_path"] = pdf_file_path
    
    
    
    ## get embedding & update metadata
    embedding = embed_document(metadata["abstract"])
    metadata["embedding"] = embedding

    return metadata





if __name__ == "__main__":
    server.run()