# PaperMap

> **Type a topic and see every connected paper on arXiv, mapped as a live knowledge graph**

## What it does

1. You type a topic (e.g. *"diffusion models"*) or upload a PDF
2. PaperMap fetches related papers from **arXiv**
3. An LLM extracts **key concepts** from each abstract
4. Papers sharing concepts get connected in a **Neo4j knowledge graph**
5. You explore the graph visually, clicking nodes shows neighbours


## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start Neo4j (Docker)
docker run --name papermap-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5

# 3. Run
streamlit run app.py
```

Then open http://localhost:8501, enter your Neo4j password and OpenAI key in the sidebar, and search!

## Neo4j vector index (run once)

```cypher
CREATE VECTOR INDEX `paper-abstracts`
FOR (p:Paper) ON (p.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 3072,
  `vector.similarity_function`: 'cosine'
}}
```

## Project layout

```
papermap/
├── app.py                  ← Streamlit entry point + sidebar
├── requirements.txt
├── pages/
│   ├── discover.py         ← Search arXiv / upload PDF + ingest
│   ├── graph.py            ← Interactive vis.js graph explorer
│   ├── chat.py             ← GraphRAG chat about the paper set
│   └── pipeline.py        ← How it works + code walkthrough
└── utils/
    └── engine.py           ← ArxivFetcher, PaperGraphBuilder,
                               ConceptExtractor, GraphRAGEngine, PDFParser
```

## Stack

| Layer | Library |
|-------|---------|
| UI | Streamlit |
| Graph DB + Vector | Neo4j 5 |
| Orchestration | LangChain 0.3 |
| LLM | GPT-4o (concept extraction + chat) |
| Embeddings | text-embedding-3-large (3072d) |
| Paper source | arXiv API |
| PDF parsing | PyMuPDF |
| Graph visualisation | vis.js |
