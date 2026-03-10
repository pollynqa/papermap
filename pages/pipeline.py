"""pages/pipeline.py — How PaperMap works, with code snippets"""
import streamlit as st
import streamlit.components.v1 as components


STEPS = [
    {
        "icon": "🔍", "label": "arXiv Search",
        "desc": "Query the arXiv API for papers matching your topic. Returns title, abstract, authors, year.",
        "color": "#3b82f6",
        "code": """\
import arxiv

client = arxiv.Client()
search = arxiv.Search(
    query="graph neural networks",
    max_results=15,
    sort_by=arxiv.SortCriterion.Relevance,
)
papers = list(client.results(search))
""",
    },
    {
        "icon": "🤖", "label": "Concept Extraction",
        "desc": "GPT-4o-mini reads each abstract and extracts 4–8 key ML concepts per paper.",
        "color": "#22d3a5",
        "code": """\
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY a JSON array of 4-8 concept strings."),
    ("human",  "Title: {title}\\nAbstract: {abstract}"),
])
chain    = prompt | llm
concepts = chain.invoke({"title": ..., "abstract": ...})
""",
    },
    {
        "icon": "🕸", "label": "Neo4j Graph",
        "desc": "Papers and concepts are written as nodes. Papers sharing concepts get RELATED_TO edges.",
        "color": "#f59e0b",
        "code": """\
from neo4j import GraphDatabase

driver = GraphDatabase.driver(uri, auth=(user, pw))

with driver.session() as s:
    # Paper node
    s.run("MERGE (p:Paper {id:$id}) SET p += $props", ...)

    # Concept + edge
    s.run(
        "MERGE (c:Concept {name: $concept}) "
        "MATCH (p:Paper {id: $pid}) "
        "MERGE (p)-[:COVERS]->(c)",
        ...
    )

    # Auto-connect papers sharing concepts
    s.run(
        "MATCH (p1:Paper)-[:COVERS]->(c)<-[:COVERS]-(p2:Paper) "
        "WHERE p1.id < p2.id "
        "MERGE (p1)-[r:RELATED_TO]->(p2) "
        "ON CREATE SET r.shared_concepts = 1 "
        "ON MATCH SET r.shared_concepts = r.shared_concepts + 1"
    )
""",
    },
    {
        "icon": "🔢", "label": "Vector Embeddings",
        "desc": "Abstracts are embedded with text-embedding-3-large and stored in Neo4j's vector index for semantic search.",
        "color": "#a78bfa",
        "code": """\
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Neo4jVector.from_existing_graph(
    embeddings,
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS,
    index_name="paper-abstracts",
    node_label="Paper",
    text_node_properties=["title", "abstract"],
    embedding_node_property="embedding",
)
""",
    },
    {
        "icon": "💬", "label": "GraphRAG Chat",
        "desc": "Questions are answered by combining vector similarity search with Cypher graph traversal, fed to GPT-4o.",
        "color": "#f87171",
        "code": """\
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(url=..., username=..., password=...)
chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    graph=graph,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

# Hybrid: vector context + graph traversal
vec_docs    = vector_store.similarity_search(question, k=5)
graph_result = chain.invoke({"query": question})
# → merge both contexts → LLM generates final answer
""",
    },
]


def render():
    st.markdown("""
<div style="margin-bottom:20px">
  <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;margin-bottom:4px">
    How PaperMap works
  </div>
  <div style="font-size:.8rem;color:#5a7090">
    A 5-step pipeline from arXiv search to interactive knowledge graph
  </div>
</div>
""", unsafe_allow_html=True)

    # Pipeline diagram
    _draw_pipeline()
    st.markdown("---")

    # Step detail cards
    for i, step in enumerate(STEPS):
        with st.expander(f"{step['icon']}  Step {i+1}: {step['label']}", expanded=(i == 0)):
            col_desc, col_code = st.columns([1, 2])
            with col_desc:
                st.markdown(
                    f"<p style='font-size:.85rem;line-height:1.7;color:#94a3b8'>{step['desc']}</p>",
                    unsafe_allow_html=True,
                )
            with col_code:
                st.code(step["code"], language="python")

    st.markdown("---")
    st.markdown("### Neo4j Schema")
    st.code("""\
// ── Nodes ────────────────────────────────────────────────────────
(:Paper   { id, title, abstract, year, venue, url, embedding })
(:Concept { name })
(:Author  { name })

// ── Relationships ─────────────────────────────────────────────────
(:Author)-[:AUTHORED]->    (:Paper)
(:Paper) -[:COVERS]->      (:Concept)   // paper covers a concept
(:Paper) -[:RELATED_TO {shared_concepts: INT}]-> (:Paper)
         // auto-created when two papers share ≥1 concept

// ── Vector index (for semantic search) ───────────────────────────
CREATE VECTOR INDEX `paper-abstracts`
FOR (p:Paper) ON (p.embedding)
OPTIONS { indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
}}
""", language="cypher")


def _draw_pipeline():
    """Render a simple horizontal SVG pipeline."""
    colors = [s["color"] for s in STEPS]
    icons  = [s["icon"]  for s in STEPS]
    labels = [s["label"] for s in STEPS]

    n   = len(STEPS)
    bw  = 110   # box width
    bh  = 72
    gap = 28
    pad = 20
    W   = pad * 2 + n * bw + (n - 1) * gap
    H   = bh + 40

    boxes   = ""
    arrows  = ""
    for i in range(n):
        x = pad + i * (bw + gap)
        c = colors[i]
        boxes += f"""
<rect x="{x}" y="20" width="{bw}" height="{bh}" rx="8"
      fill="{c}18" stroke="{c}" stroke-width="1.5"/>
<text x="{x + bw//2}" y="46" text-anchor="middle" font-size="18">{icons[i]}</text>
<text x="{x + bw//2}" y="63" text-anchor="middle"
      fill="{c}" font-size="10" font-family="DM Sans,sans-serif" font-weight="600">
  {labels[i]}
</text>
<text x="{x + bw//2}" y="76" text-anchor="middle"
      fill="#2a3d55" font-size="9" font-family="DM Mono,monospace">
  Step {i+1}
</text>"""
        if i < n - 1:
            ax = x + bw + 2
            ay = 20 + bh // 2
            arrows += f"""
<line x1="{ax}" y1="{ay}" x2="{ax + gap - 4}" y2="{ay}"
      stroke="{c}" stroke-width="1.5" marker-end="url(#a{i})"/>
<defs>
  <marker id="a{i}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
    <path d="M0,0 L0,6 L6,3 z" fill="{c}"/>
  </marker>
</defs>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{H}"
         viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">
  <rect width="{W}" height="{H}" fill="#080c14"/>
  {arrows}
  {boxes}
</svg>"""

    st.markdown(svg, unsafe_allow_html=True)
