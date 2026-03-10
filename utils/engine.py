"""
utils/engine.py
─────────────────────────────────────────────────────────────────────────────
Core logic:
  ArxivFetcher          – search arXiv, return structured paper dicts
  PaperGraphBuilder     – write papers + edges into Neo4j
  ConceptExtractor      – use LLM to pull concepts from abstracts
  GraphRAGEngine        – LangChain GraphCypherQAChain + vector search
  PDFParser             – extract text & metadata from uploaded PDFs
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Callable
import os


# ─────────────────────────────────────────────────────────────────────────────
# Data models (plain dicts — no heavy deps required at import time)
# ─────────────────────────────────────────────────────────────────────────────
def empty_paper() -> Dict[str, Any]:
    return {
        "id":        "",
        "title":     "",
        "abstract":  "",
        "authors":   [],
        "year":      0,
        "venue":     "arXiv",
        "url":       "",
        "concepts":  [],     # filled by ConceptExtractor
        "embedding": [],     # filled after embed step
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. arXiv Fetcher
# ─────────────────────────────────────────────────────────────────────────────
class ArxivFetcher:
    """
    Wraps the `arxiv` library.
    Returns list of paper dicts ready for graph ingestion.
    """

    def search(
        self,
        query: str,
        max_results: int = 15,
        progress: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        import arxiv

        if progress:
            progress(0.1, f"Searching arXiv for **{query}**…")

        client = arxiv.Client(num_retries=3, page_size=min(max_results, 25))
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for i, result in enumerate(client.results(search)):
            p = empty_paper()
            p["id"]       = result.entry_id.split("/")[-1]   # e.g. "2005.14165v3" → strip "v3"
            p["id"]       = re.sub(r"v\d+$", "", p["id"])
            p["title"]    = result.title.strip()
            p["abstract"] = result.summary.strip()
            p["authors"]  = [a.name for a in result.authors]
            p["year"]     = result.published.year
            p["url"]      = result.entry_id
            papers.append(p)

            if progress:
                progress(0.1 + 0.4 * (i + 1) / max_results,
                         f"Fetched {i+1}/{max_results}: {p['title'][:55]}…")

        return papers

    def fetch_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        try:
            result = next(client.results(search))
            p = empty_paper()
            p["id"]       = arxiv_id
            p["title"]    = result.title.strip()
            p["abstract"] = result.summary.strip()
            p["authors"]  = [a.name for a in result.authors]
            p["year"]     = result.published.year
            p["url"]      = result.entry_id
            return p
        except StopIteration:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. PDF Parser
# ─────────────────────────────────────────────────────────────────────────────
class PDFParser:
    """Extract title, abstract, authors from an uploaded PDF using PyMuPDF."""

    def parse(self, pdf_bytes: bytes) -> Dict[str, Any]:
        import fitz  # PyMuPDF
        import io

        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        # Heuristic: first non-empty lines = title
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title = lines[0] if lines else "Untitled"

        # Abstract: look for "abstract" section
        abstract = ""
        lower = text.lower()
        abs_start = lower.find("abstract")
        if abs_start != -1:
            intro = lower.find("introduction", abs_start)
            end   = intro if intro != -1 else abs_start + 1500
            abstract = text[abs_start + 8 : end].strip()[:800]

        p = empty_paper()
        p["id"]       = f"pdf_{hash(title) & 0xFFFFFF}"
        p["title"]    = title
        p["abstract"] = abstract or text[:600]
        p["full_text"] = text
        return p


# ─────────────────────────────────────────────────────────────────────────────
# 3. Concept Extractor  (LLM-powered)
# ─────────────────────────────────────────────────────────────────────────────
class ConceptExtractor:
    """
    Uses a cheap LLM call (gpt-4o-mini) to extract 4–8 concepts per paper.
    Falls back to keyword extraction if no API key.
    """

    def __init__(self, openai_api_key: str = ""):
        self.api_key = openai_api_key

    def extract(self, papers: List[Dict], progress: Optional[Callable] = None) -> List[Dict]:
        if self.api_key:
            return self._llm_extract(papers, progress)
        return self._keyword_extract(papers)

    def _llm_extract(self, papers, progress):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json

        os.environ["OPENAI_API_KEY"] = self.api_key
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Return ONLY a JSON array of 4-8 short concept strings (e.g. [\"attention mechanism\", \"transformer\"]). No explanation."),
            ("human",  "Title: {title}\nAbstract: {abstract}"),
        ])
        chain = prompt | llm | StrOutputParser()

        for i, p in enumerate(papers):
            if progress:
                progress(0.55 + 0.2 * i / len(papers),
                         f"Extracting concepts from: {p['title'][:50]}…")
            try:
                raw  = chain.invoke({"title": p["title"], "abstract": p["abstract"][:600]})
                raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
                p["concepts"] = json.loads(raw)
            except Exception:
                p["concepts"] = self._keyword_extract_one(p["abstract"])

        return papers

    def _keyword_extract(self, papers):
        for p in papers:
            p["concepts"] = self._keyword_extract_one(p["abstract"])
        return papers

    def _keyword_extract_one(self, text: str) -> List[str]:
        """Naive keyword extraction when no LLM is available."""
        ML_KEYWORDS = [
            "transformer", "attention", "bert", "gpt", "neural network",
            "deep learning", "reinforcement learning", "graph neural",
            "diffusion", "generative", "fine-tuning", "pre-training",
            "contrastive", "self-supervised", "knowledge graph", "rag",
            "retrieval", "language model", "convolutional", "recurrent",
            "embedding", "vector", "classification", "segmentation",
        ]
        found = [kw for kw in ML_KEYWORDS if kw in text.lower()]
        return found[:6] if found else ["machine learning"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Neo4j Graph Builder
# ─────────────────────────────────────────────────────────────────────────────
class PaperGraphBuilder:
    """
    Writes papers, authors, concepts and their relationships into Neo4j.
    Also builds citation edges between papers that share references.
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()

    def _init_schema(self):
        with self.driver.session() as s:
            # Constraints
            for stmt in [
                "CREATE CONSTRAINT paper_id  IF NOT EXISTS FOR (p:Paper)   REQUIRE p.id   IS UNIQUE",
                "CREATE CONSTRAINT concept_n IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT author_n  IF NOT EXISTS FOR (a:Author)  REQUIRE a.name IS UNIQUE",
            ]:
                try:
                    s.run(stmt)
                except Exception:
                    pass

    def ingest_papers(
        self,
        papers: List[Dict],
        progress: Optional[Callable] = None,
    ) -> None:
        for i, p in enumerate(papers):
            if progress:
                progress(0.75 + 0.2 * i / len(papers),
                         f"Writing to Neo4j: {p['title'][:55]}…")
            with self.driver.session() as s:
                # Paper node
                s.run("""
MERGE (p:Paper {id: $id})
SET p.title    = $title,
    p.abstract = $abstract,
    p.year     = $year,
    p.venue    = $venue,
    p.url      = $url
""", p)

                # Author nodes + AUTHORED edges
                for author in p.get("authors", []):
                    s.run("""
MERGE (a:Author {name: $name})
WITH a
MATCH (p:Paper {id: $pid})
MERGE (a)-[:AUTHORED]->(p)
""", {"name": author, "pid": p["id"]})

                # Concept nodes + COVERS edges
                for concept in p.get("concepts", []):
                    s.run("""
MERGE (c:Concept {name: $name})
WITH c
MATCH (p:Paper {id: $pid})
MERGE (p)-[:COVERS]->(c)
""", {"name": concept.lower(), "pid": p["id"]})

        # Shared-concept citation: papers covering same concepts get RELATED_TO edges
        with self.driver.session() as s:
            s.run("""
MATCH (p1:Paper)-[:COVERS]->(c:Concept)<-[:COVERS]-(p2:Paper)
WHERE p1.id < p2.id
MERGE (p1)-[r:RELATED_TO]->(p2)
ON CREATE SET r.shared_concepts = 1
ON MATCH  SET r.shared_concepts = r.shared_concepts + 1
""")

    def get_graph_data(self) -> Dict[str, Any]:
        """Return all nodes + edges for visualisation."""
        with self.driver.session() as s:
            node_records = s.run("""
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:COVERS]->(c:Concept)
RETURN p.id    AS id,
       p.title AS title,
       p.year  AS year,
       p.url   AS url,
       p.abstract AS abstract,
       collect(DISTINCT c.name) AS concepts
""").data()

            edge_records = s.run("""
MATCH (p1:Paper)-[r:RELATED_TO]->(p2:Paper)
RETURN p1.id AS from, p2.id AS to,
       r.shared_concepts AS weight
""").data()

            concept_records = s.run("""
MATCH (c:Concept)
RETURN c.name AS name,
       size([(p:Paper)-[:COVERS]->(c) | p]) AS paper_count
ORDER BY paper_count DESC LIMIT 30
""").data()

        return {
            "nodes":    node_records,
            "edges":    edge_records,
            "concepts": concept_records,
        }

    def concept_cluster(self, concept: str) -> List[Dict]:
        """Get all papers covering a specific concept."""
        with self.driver.session() as s:
            return s.run("""
MATCH (p:Paper)-[:COVERS]->(c:Concept {name: $name})
RETURN p.id AS id, p.title AS title, p.year AS year, p.abstract AS abstract
ORDER BY p.year DESC
""", {"name": concept.lower()}).data()

    def paper_neighbours(self, paper_id: str) -> List[Dict]:
        """Get papers related to a given paper_id."""
        with self.driver.session() as s:
            return s.run("""
MATCH (p:Paper {id: $id})-[:RELATED_TO]-(n:Paper)
OPTIONAL MATCH (n)-[:COVERS]->(c:Concept)
RETURN n.id AS id, n.title AS title, n.year AS year,
       collect(DISTINCT c.name) AS concepts
ORDER BY n.year DESC
""", {"id": paper_id}).data()

    def get_stats(self) -> Dict[str, int]:
        with self.driver.session() as s:
            papers   = s.run("MATCH (p:Paper)   RETURN count(p) AS n").single()["n"]
            edges    = s.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS n").single()["n"]
            concepts = s.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"]
        return {"papers": papers, "edges": edges, "concepts": concepts}

    def close(self):
        self.driver.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. GraphRAG Engine  (LangChain)
# ─────────────────────────────────────────────────────────────────────────────
class GraphRAGEngine:
    """
    Hybrid retrieval:
      a) Neo4j vector index  → semantic similarity search on abstracts
      b) GraphCypherQAChain  → structured graph traversal
    Both contexts fed to the LLM.
    """

    def __init__(
        self,
        neo4j_uri:  str,
        neo4j_user: str,
        neo4j_pass: str,
        openai_key: str,
        model:      str = "gpt-4o",
        embed_model: str = "text-embedding-3-large",
        top_k:      int = 5,
    ):
        os.environ["OPENAI_API_KEY"] = openai_key
        self.top_k = top_k
        self._build(neo4j_uri, neo4j_user, neo4j_pass, model, embed_model)

    def _build(self, uri, user, pw, model, embed_model):
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.graphs import Neo4jGraph
        from langchain_community.vectorstores import Neo4jVector
        from langchain.chains import GraphCypherQAChain
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        self.graph      = Neo4jGraph(url=uri, username=user, password=pw)
        self.embeddings = OpenAIEmbeddings(model=embed_model)
        self.llm        = ChatOpenAI(model=model, temperature=0.15, streaming=True)

        # Vector store over Paper.abstract
        self.vector_store = Neo4jVector.from_existing_graph(
            self.embeddings,
            url=uri, username=user, password=pw,
            index_name="paper-abstracts",
            node_label="Paper",
            text_node_properties=["title", "abstract"],
            embedding_node_property="embedding",
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # Cypher QA chain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=False,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

        # RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are PaperMap, a research assistant with access to a 
knowledge graph of AI/ML papers discovered from arXiv.

Use the context below to answer the user's question accurately and helpfully.

=== Graph traversal context ===
{graph_ctx}

=== Semantically similar papers (vector search) ===
{vector_ctx}

Rules:
- Always cite paper titles and years when mentioning research.
- When comparing papers, explain what connects or distinguishes them.
- If you generate a Cypher query, wrap it in ```cypher ... ``` blocks.
- Be concise but thorough."""),
            ("human", "{question}"),
        ])
        self._str_out = StrOutputParser()

    def query(self, question: str) -> Dict[str, Any]:
        # Vector retrieval
        vec_docs = self.retriever.invoke(question)
        vector_ctx = "\n\n".join(
            f"[{i+1}] **{d.metadata.get('title','')}**\n{d.page_content[:350]}"
            for i, d in enumerate(vec_docs)
        )

        # Graph traversal
        try:
            result     = self.cypher_chain.invoke({"query": question})
            graph_ctx  = result.get("result", "")
            cypher_used = (result.get("intermediate_steps") or [{}])[0].get("query", "")
        except Exception as e:
            graph_ctx   = f"(Graph traversal failed: {e})"
            cypher_used = ""

        # LLM
        chain  = self.prompt | self.llm | self._str_out
        answer = chain.invoke({
            "question":   question,
            "graph_ctx":  graph_ctx,
            "vector_ctx": vector_ctx,
        })

        return {
            "answer":      answer,
            "cypher":      cypher_used,
            "vec_docs":    vec_docs,
            "graph_ctx":   graph_ctx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mock data  (UI works without Neo4j connection)
# ─────────────────────────────────────────────────────────────────────────────
MOCK_PAPERS = [
    {"id":"2017.03550","title":"Attention Is All You Need","abstract":"We propose the Transformer, a model based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.","authors":["Ashish Vaswani","Noam Shazeer"],"year":2017,"concepts":["transformer","attention mechanism","self-attention","encoder-decoder","positional encoding"],"url":"https://arxiv.org/abs/1706.03762"},
    {"id":"1810.04805","title":"BERT: Pre-training of Deep Bidirectional Transformers","abstract":"We introduce BERT, designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.","authors":["Jacob Devlin","Ming-Wei Chang"],"year":2018,"concepts":["bert","pre-training","bidirectional","masked language model","nlp"],"url":"https://arxiv.org/abs/1810.04805"},
    {"id":"2005.14165","title":"Language Models are Few-Shot Learners","abstract":"We train GPT-3, an autoregressive language model with 175 billion parameters, and test its few-shot performance.","authors":["Tom Brown","Benjamin Mann"],"year":2020,"concepts":["gpt-3","few-shot learning","language model","in-context learning","autoregressive"],"url":"https://arxiv.org/abs/2005.14165"},
    {"id":"2302.13971","title":"LLaMA: Open and Efficient Foundation Language Models","abstract":"We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters trained on publicly available data.","authors":["Hugo Touvron","Thibaut Lavril"],"year":2023,"concepts":["llama","open-source","foundation model","efficient training","language model"],"url":"https://arxiv.org/abs/2302.13971"},
    {"id":"2106.09685","title":"LoRA: Low-Rank Adaptation of Large Language Models","abstract":"We propose LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer.","authors":["Edward Hu","Yelong Shen"],"year":2021,"concepts":["lora","fine-tuning","low-rank","parameter-efficient","adaptation"],"url":"https://arxiv.org/abs/2106.09685"},
    {"id":"2005.11401","title":"Retrieval-Augmented Generation for Knowledge-Intensive NLP","abstract":"We combine parametric and non-parametric memory for language generation with a retrieval component providing non-parametric memory.","authors":["Patrick Lewis","Ethan Perez"],"year":2020,"concepts":["rag","retrieval","knowledge-intensive","generation","dense passage retrieval"],"url":"https://arxiv.org/abs/2005.11401"},
    {"id":"2010.11929","title":"An Image is Worth 16x16 Words: Vision Transformer","abstract":"We apply a pure transformer directly to sequences of image patches and find it works well on image classification.","authors":["Alexey Dosovitskiy","Lucas Beyer"],"year":2020,"concepts":["vision transformer","vit","image patches","self-attention","image classification"],"url":"https://arxiv.org/abs/2010.11929"},
    {"id":"2112.10752","title":"High-Resolution Image Synthesis with Latent Diffusion Models","abstract":"We apply diffusion models in the latent space of powerful pretrained autoencoders to achieve high-resolution image synthesis.","authors":["Robin Rombach","Andreas Blattmann"],"year":2021,"concepts":["diffusion model","latent space","image synthesis","autoencoder","stable diffusion"],"url":"https://arxiv.org/abs/2112.10752"},
]

MOCK_EDGES = [
    {"from":"2017.03550","to":"1810.04805","weight":4},
    {"from":"2017.03550","to":"2005.14165","weight":3},
    {"from":"2017.03550","to":"2010.11929","weight":3},
    {"from":"1810.04805","to":"2005.14165","weight":3},
    {"from":"1810.04805","to":"2106.09685","weight":2},
    {"from":"2005.14165","to":"2302.13971","weight":4},
    {"from":"2005.14165","to":"2005.11401","weight":2},
    {"from":"2302.13971","to":"2106.09685","weight":3},
    {"from":"2112.10752","to":"2010.11929","weight":2},
]

MOCK_CONCEPTS = [
    {"name":"transformer",         "paper_count":4},
    {"name":"attention mechanism", "paper_count":3},
    {"name":"language model",      "paper_count":4},
    {"name":"fine-tuning",         "paper_count":2},
    {"name":"pre-training",        "paper_count":3},
    {"name":"diffusion model",     "paper_count":1},
    {"name":"retrieval",           "paper_count":1},
    {"name":"image classification","paper_count":1},
]

MOCK_CHAT_RESPONSES = [
    {
        "answer": "The graph shows **8 papers** on this topic. The most connected is *Attention Is All You Need* (Vaswani et al., 2017), which shares concepts with 4 other papers including BERT and GPT-3.\n\n**Shared concept clusters:**\n- `transformer` → 4 papers\n- `language model` → 4 papers\n- `pre-training` → 3 papers",
        "cypher": "MATCH (p:Paper)-[:COVERS]->(c:Concept)\nRETURN c.name, count(p) AS papers\nORDER BY papers DESC LIMIT 10",
    },
    {
        "answer": "**BERT** (2018) and the **Transformer** (2017) are directly connected via 4 shared concepts:\n- `transformer`\n- `attention mechanism`\n- `self-attention`\n- `pre-training`\n\nBERT builds directly on the Transformer encoder architecture, adding bidirectional pre-training with Masked Language Modeling.",
        "cypher": "MATCH (p1:Paper {id:'2017.03550'})-[:RELATED_TO]-(p2:Paper {id:'1810.04805'})\nMATCH (p1)-[:COVERS]->(c:Concept)<-[:COVERS]-(p2)\nRETURN c.name AS shared_concept",
    },
    {
        "answer": "The **LoRA** paper (Hu et al., 2021) is most relevant for parameter-efficient fine-tuning. It's connected to:\n- **LLaMA** (shares: `language model`, `efficient training`)\n- **BERT** (shares: `fine-tuning`, `pre-training`)\n\nLoRA reduces trainable parameters by ~10,000× versus full fine-tuning by injecting low-rank matrices.",
        "cypher": "MATCH (p:Paper {id:'2106.09685'})-[:RELATED_TO]-(n:Paper)\nMATCH (p)-[:COVERS]->(c:Concept)<-[:COVERS]-(n)\nRETURN n.title, collect(c.name) AS shared",
    },
]
