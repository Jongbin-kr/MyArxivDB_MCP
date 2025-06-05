import asyncio
import os 
from typing import *
import pathlib
from datetime import date

from dotenv import load_dotenv  # for loading environment variables from .env file
import wget, requests, feedparser         # for crawling arxiv papers

from mcp.server.fastmcp import FastMCP

from pinecone import Pinecone  ## import pinecone libraries for ermbedding passage and queries w/o resources

import psycopg2


##########################
## config for local env ##
##########################
PDF_DIRECTORY = "/tmp/bkms1/pdfs"



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

class MyArxivDBServer(FastMCP):
    def connect_db(self):
        """Connect to the PostgreSQL database."""
        try: 
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT")
            )
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise
        else:
            print("Database connection established successfully.")
        return conn

    def create_tables(self):
        """Create necessary tables in the database if they do not exist."""
        cur = self.db_conn.cursor()

        # Ensure pgvector extension is enabled
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create Projects table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Projects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                start_date DATE,
                end_date DATE
            );
            """
        )
        
        # Create Papers table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Papers (
                arxiv_id VARCHAR(255) PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT[],
                published_date DATE,
                primary_category VARCHAR(255),
                categories TEXT[],
                arxiv_url TEXT,
                pdf_file_path TEXT,
                embedding VECTOR(1024),
                user_added_date DATE
            );
            """
        )

        # Create relation_enum type for ProjectPapers table
        cur.execute(
            """
            DO $$ 
            BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'relation_enum') THEN
                CREATE TYPE relation_enum AS ENUM ('assigned', 'related');
            END IF;
            END $$;
            """
        )
        
        # Create ProjectPapers table for many-to-many relationship
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ProjectPapers (
                project_id INT REFERENCES Projects(id),
                paper_id VARCHAR(255) REFERENCES Papers(arxiv_id),
                relation_type relation_enum DEFAULT 'assigned',
                PRIMARY KEY (project_id, paper_id)
            );
            """
        )



        # Create ProjectEmbeddings view
        cur.execute(
            """
            DROP VIEW IF EXISTS ProjectEmbeddings;
            CREATE VIEW ProjectEmbeddings AS
            SELECT
                pr.id AS project_id,
                AVG(p.embedding) FILTER (WHERE pp.relation_type = 'assigned') AS embedding, 
                COUNT(p.arxiv_id) FILTER (WHERE pp.relation_type = 'assigned') AS n_assigned_papers
            FROM Projects pr
            LEFT JOIN ProjectPapers pp ON pr.id = pp.project_id
            LEFT JOIN Papers p ON pp.paper_id = p.arxiv_id
            GROUP BY pr.id;
            """
        )
        
        self.db_conn.commit()
        cur.close()

    def show_table_schema(self):
        """Print the schema of the created tables."""
        cur = self.db_conn.cursor()
        
        # Show Projects table schema
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'projects';")
        print("Projects Table Schema:")
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]}")
        
        # Show Papers table schema
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'papers';")
        print("\nPapers Table Schema:")
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]}")
        
        # Show ProjectPapers table schema
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'projectpapers';")
        print("\nProjectPapers Table Schema:")
        for row in cur.fetchall():
            print(f"{row[0]}: {row[1]}")

        cur.close()
        
    def create_view(self):
        cur = self.db_conn.cursor()
        
        cur.execute(
            """
            CREATE OR REPLACE VIEW ProjectEmbeddings AS
            SELECT
                pr.id AS project_id,
                COUNT(p.arxiv_id) FILTER (WHERE ppl.relation_type = 'assigned') AS num_assigned_papers,
                AVG(p.embedding) FILTER (WHERE ppl.relation_type = 'assigned') AS avg_embedding
            FROM Projects pr
            LEFT JOIN ProjectPapers ppl ON pr.id = ppl.project_id
            LEFT JOIN Papers p ON ppl.paper_id = p.arxiv_id
            GROUP BY pr.id;
            """
        )
        
        self.db_conn.commit()
        cur.close()
    
    def __init__(self, name: str, dependencies: List[str]):
        super().__init__(name, dependencies=dependencies)
        # init db connection
        self.db_conn = self.connect_db()
        # create tables if not exist
        self.create_tables()
        self.show_table_schema()
        # self.create_view()

# Replace the server initialization with the new class
server = MyArxivDBServer("MyArxivDB_MCP", 
                         dependencies=["wget", "requests", "feedparser",
                                       "pinecone", "psycopg2"])


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
            "arxiv_id":          arxiv_id,
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

    ## update user_added_date to metadata
    metadata["user_added_date"] = date.today().isoformat()

    return metadata


# Clear database tables
@server.tool() 
async def clear_database() -> str:
    cur = server.db_conn.cursor()

    # Clear ProjectPapers table
    cur.execute("DELETE FROM ProjectPapers;")
    
    # Clear Papers table
    cur.execute("DELETE FROM Papers;")
    
    # Clear Projects table
    cur.execute("DELETE FROM Projects;")

    cur.close()

    server.db_conn.commit()

    return "Database cleared successfully"


# Add new project to the database, get project ID
@server.tool()
async def add_new_project(project_name: str,
                          project_description: str,
                          project_start: date,
                          project_end: date) -> int:
    cur = server.db_conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO Projects (name, description, start_date, end_date)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (project_name, project_description, project_start, project_end)
        )

        project_id = cur.fetchone()[0]
        cur.close()
        server.db_conn.commit()
        return project_id
    except Exception as e:
        print(f"Error inserting tuple into Projects : {e}")
        raise
        
    

# Add a new paper to the database
@server.tool()
async def add_new_paper(arxiv_id: str) -> dict:
    cur = server.db_conn.cursor()

    # Check if the paper already exists
    cur.execute(
        """
        SELECT arxiv_id FROM Papers WHERE arxiv_id = %s;
        """,
        (arxiv_id,)
    )
    paper = cur.fetchone()

    if not paper:
        # If the paper does not exist, crawl it
        metadata = await crawl_arxiv_paper(arxiv_id)
        
        # Insert the new paper into the Papers table
        cur.execute(
            """
            INSERT INTO Papers (arxiv_id, title, abstract, authors, published_date, primary_category, categories, pdf_file_path, embedding, user_added_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING arxiv_id;
            """,
            (
                metadata["arxiv_id"],
                metadata["title"],
                metadata["abstract"],
                metadata["authors"],
                metadata["published_date"],
                metadata["primary_category"],
                metadata["categories"],
                str(metadata["pdf_file_path"]),
                metadata["embedding"],
                metadata["user_added_date"]
            )
        )
        
        arxiv_id = cur.fetchone()[0]
        cur.close()
        server.db_conn.commit()
        return {"message": "Paper added successfully", "arxiv_id": arxiv_id}
        
    else:
        arxiv_id = paper[0]
        cur.close()
        return {"message": "Paper already exists", "arxiv_id": arxiv_id}

# Get one project from the database
@server.tool()
async def get_project_by_id(project_id: int) -> dict:
    cur = server.db_conn.cursor()

    cur.execute(
        """
        SELECT * FROM Projects WHERE id = %s;
        """,
        (project_id,)
    )

    project = cur.fetchone()
    cur.close()
    if project:
        return {
            "id": project[0],
            "name": project[1],
            "description": project[2],
            "start_date": project[3],
            "end_date": project[4]
        }
    else:
        return {"error": "Project not found"}

# Get all projects from the database
@server.tool()
async def get_all_projects() -> List[dict]:
    cur = server.db_conn.cursor()

    cur.execute(
        """
        SELECT * FROM Projects;
        """
    )

    projects = cur.fetchall()
    cur.close()
    return [
        {
            "id": project[0],
            "name": project[1],
            "description": project[2],
            "start_date": project[3],
            "end_date": project[4]
        }
        for project in projects
    ]

# Get all papers from the database
@server.tool()
async def get_all_papers() -> List[dict]:
    cur = server.db_conn.cursor()
    try: 
        cur.execute(
            """
            SELECT * FROM Papers;
            """
        )

        papers = cur.fetchall()
        return [
            {
                "arxiv_id": paper[0],
                "title": paper[1],
                "abstract": paper[2],
                "authors": paper[3],
                "published_date": paper[4],
                "primary_category": paper[5],
                "categories": paper[6],
                "arxiv_url": paper[7],
                "pdf_file_path": paper[8],
                "embedding": paper[9],
                "user_added_date": paper[10]
            }
            for paper in papers
        ]
    except Exception as e:
        server.db_conn.rollback()
        return {"error": str(e)}
    finally:
        cur.close()

# Assign a paper to a project
@server.tool()
async def assign_paper_to_project(project_id: int, arxiv_id: str) -> dict:
    cur = server.db_conn.cursor()

    # Check if the project exists
    cur.execute(
        """
        SELECT id FROM Projects WHERE id = %s;
        """,
        (project_id,)
    )
    project = cur.fetchone()
    
    if not project:
        return {"error": "Project not found"}

    # Check if the paper already exists
    cur.execute(
        """
        SELECT arxiv_id FROM Papers WHERE arxiv_id = %s;
        """,
        (arxiv_id,)
    )
    paper = cur.fetchone()

    if not paper:
        # If the paper does not exist, add it
        metadata = await add_new_paper(arxiv_id)
        if "error" in metadata:
            return metadata
        paper_id = metadata["arxiv_id"]
        
    else:
        paper_id = paper[0]

    # Assign the paper to the project
    cur.execute(
        """
        INSERT INTO ProjectPapers (project_id, paper_id)
        VALUES (%s, %s);
        """,
        (project_id, paper_id)
    )

    cur.close()
    server.db_conn.commit()

    return {"message": "Paper assigned to project successfully", "paper_id": paper_id}


# Get papers by date
@server.tool()
async def get_papers_by_date(added_date: str) -> List[dict]:
    cur = server.db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT arxiv_id, title, abstract, authors, published_date,
                   primary_category, categories, arxiv_url, pdf_file_path,
                   user_added_date
            FROM Papers
            WHERE user_added_date = %s;
            """,
            (added_date,)
        )
        rows = cur.fetchall()
        return [
            {
                "arxiv_id":       r[0],
                "title":          r[1],
                "abstract":       r[2],
                "authors":        r[3],
                "published_date": r[4],
                "primary_category": r[5],
                "categories":     r[6],
                "arxiv_url":      r[7],
                "pdf_file_path":  r[8],
                "user_added_date": r[9]
            }
            for r in rows
        ]
    except Exception as e:
        server.db_conn.rollback()
        return {"error": str(e)}
    finally:
        cur.close()


# Get papers by category
@server.tool()
async def get_papers_by_category(category: str) -> List[dict]:
    cur = server.db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT arxiv_id, title, abstract, authors, published_date,
                   primary_category, categories, arxiv_url, pdf_file_path,
                   user_added_date
            FROM Papers
            WHERE %s = ANY(categories);
            """,
            (category,)
        )
        rows = cur.fetchall()
        return [
            {
                "arxiv_id":       r[0],
                "title":          r[1],
                "abstract":       r[2],
                "authors":        r[3],
                "published_date": r[4],
                "primary_category": r[5],
                "categories":     r[6],
                "arxiv_url":      r[7],
                "pdf_file_path":  r[8],
                "user_added_date": r[9]
            }
            for r in rows
        ]
    except Exception as e:
        server.db_conn.rollback()
        return {"error": str(e)}
    finally:
        cur.close()


# Find the closest project of a given paper
@server.tool()
async def assign_paper_to_closest_project(arxiv_id: str) -> dict:
    cur = server.db_conn.cursor()

    # Check if the paper already exists
    cur.execute(
        """
        SELECT arxiv_id FROM Papers WHERE arxiv_id = %s;
        """,
        (arxiv_id,)
    )
    paper = cur.fetchone()

    if not paper:
        # If the paper does not exist, add it
        metadata = await add_new_paper(arxiv_id)
        if "error" in metadata:
            return metadata
        paper_id = metadata["arxiv_id"]
        
    else:
        paper_id = paper[0]

    # Find the project with the closest embedding
    cur.execute(
        """
        SELECT pr.id, pr.name, pr.description, pr.start_date, pr.end_date,
               pe.embedding, pe.n_assigned_papers
        FROM Projects pr
        JOIN ProjectEmbeddings pe ON pr.id = pe.project_id
        ORDER BY pe.embedding <-> (SELECT embedding FROM Papers WHERE arxiv_id = %s)
        LIMIT 1;
        """,
        (arxiv_id,)
    )
    project = cur.fetchone()
    if not project:
        return {"error": "No projects found"}

    # Assign the paper to the project
    cur.execute(
        """
        INSERT INTO ProjectPapers (project_id, paper_id, relation_type)
        VALUES (%s, %s, 'related');
        """,
        (project[0], paper_id)
    )

    server.db_conn.commit()

    return {"message": "Paper assigned to project successfully", "project_id": project[0], "project_name": project[1], "paper_id": paper_id}


# Check ProjectEmbeddings ordered by project_id (default rows = 20)
@server.tool()
async def query_project_stats(limit: int = 20) -> List[dict]:
    cur = server.db_conn.cursor()
    cur.execute("""
        SELECT project_id,
               num_assigned_papers,
               avg_embedding::text
        FROM   ProjectEmbeddings
        ORDER  BY project_id
        LIMIT  %s;
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    return [
        {
            "project_id":            r[0],
            "num_assigned_papers":   r[1],
            "avg_embedding":         r[2]
        }
        for r in rows
    ]

# Search related papers based on query in a specific project
@server.tool()
async def search_related_papers(project_id: int, query: str, top_k: int = 5) -> List[dict]:
    query_vec = embed_query(query)

    cur = server.db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.published_date,
                   p.primary_category, p.categories, p.arxiv_url, p.pdf_file_path,
                   p.user_added_date,
                   (p.embedding <-> %s::vector) AS distance
            FROM Papers p
            JOIN ProjectPapers pp ON p.arxiv_id = pp.paper_id
            WHERE pp.project_id = %s
            ORDER BY p.embedding <-> %s::vector
            LIMIT %s;
            """,
            (query_vec, project_id, query_vec, top_k)
        )
        rows = cur.fetchall()
        return [
            {
                "arxiv_id":       r[0],
                "title":          r[1],
                "abstract":       r[2],
                "authors":        r[3],
                "published_date": r[4],
                "primary_category": r[5],
                "categories":     r[6],
                "arxiv_url":      r[7],
                "pdf_file_path":  r[8],
                "user_added_date": r[9],
                "distance":       r[10]
            }
            for r in rows
        ]
    except Exception as e:
        server.db_conn.rollback()
        return {"error": str(e)}
    finally:
        cur.close()

if __name__ == "__main__":
    server.run()