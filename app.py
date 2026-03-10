"""
PaperMap — Discover connected research papers via arXiv + Neo4j + LangChain RAG
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PaperMap",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:        #080c14;
  --surface:   #0d1422;
  --surface2:  #121b2e;
  --border:    #1c2d47;
  --accent:    #3b82f6;
  --accent-lo: #3b82f620;
  --green:     #22d3a5;
  --green-lo:  #22d3a510;
  --amber:     #f59e0b;
  --amber-lo:  #f59e0b12;
  --red:       #f87171;
  --text:      #e2eaf6;
  --muted:     #5a7090;
  --font-head: 'DM Serif Display', Georgia, serif;
  --font-body: 'DM Sans', sans-serif;
  --font-mono: 'DM Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: var(--font-head) !important;
}

/* ── App viewport ── */
[data-testid="stAppViewContainer"] > .main { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 14px 18px !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  color: var(--accent) !important;
  font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .75rem !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface2) !important;
  color: var(--text) !important;
  transition: all .15s !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--accent-lo) !important;
}

/* Primary button override */
.primary-btn > button {
  background: var(--accent) !important;
  color: #fff !important;
  border-color: var(--accent) !important;
}
.primary-btn > button:hover { opacity: .85 !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-lo) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--font-body) !important;
  font-weight: 500 !important;
  font-size: .875rem !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 22px !important;
  transition: all .15s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] { padding-top: 4px !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Code ── */
code { font-family: var(--font-mono) !important; color: var(--green) !important; }
pre  { background: var(--surface2) !important; border: 1px solid var(--border) !important;
       border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Alerts ── */
.stAlert { border-radius: 8px !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:4px 0 16px">
  <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;letter-spacing:-.01em">
    🗺️ PaperMap
  </div>
  <div style="font-size:.75rem;color:#5a7090;margin-top:2px">
    Connected research, visualised
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🔌 Neo4j")
    st.text_input("URI",      value="bolt://localhost:7687", key="neo4j_uri")
    st.text_input("User",     value="neo4j",                key="neo4j_user")
    st.text_input("Password", type="password",              key="neo4j_pass")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test", use_container_width=True):
            st.success("✓ OK")
    with col2:
        if st.button("Reset DB", use_container_width=True):
            st.warning("Cleared")

    st.markdown("---")
    st.markdown("### 🤖 LLM / Embeddings")
    st.text_input("OpenAI API Key", type="password", key="openai_key",
                  placeholder="sk-...")
    st.selectbox("Chat model",
                 ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo",
                  "claude-3-5-sonnet-20241022"],
                 key="chat_model")
    st.selectbox("Embed model",
                 ["text-embedding-3-large", "text-embedding-3-small"],
                 key="embed_model")

    st.markdown("---")
    st.markdown("### ⚙️ Graph Settings")
    st.slider("Max papers per search", 5, 50, 15, key="max_papers")
    st.slider("Citation hops",         1,  4,  2, key="hops")
    st.slider("Min similarity score",  0.0, 1.0, 0.6, 0.05, key="min_sim")

    st.markdown("---")
    # Live graph stats
    st.markdown("### 📊 Graph Stats")
    papers = st.session_state.get("graph_paper_count", 0)
    edges  = st.session_state.get("graph_edge_count",  0)
    st.metric("Papers in graph", papers)
    st.metric("Connections",     edges)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 20px">
  <h1 style="font-family:'DM Serif Display',serif;font-size:2rem;
             font-weight:400;letter-spacing:-.02em;margin:0;line-height:1.1">
    Discover connected papers
    <span style="color:#3b82f6;font-style:italic"> on any topic</span>
  </h1>
  <p style="color:#5a7090;font-size:.875rem;margin-top:6px">
    Search arXiv → extract concepts & citations → build a live Neo4j knowledge graph → explore & chat
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pages.discover import render as render_discover
from pages.graph    import render as render_graph
from pages.chat     import render as render_chat
from pages.pipeline import render as render_pipeline

tab_discover, tab_graph, tab_chat, tab_pipeline = st.tabs([
    "🔍  Discover",
    "🕸  Graph Explorer",
    "💬  Chat",
    "⚙️  How It Works",
])

with tab_discover: render_discover()
with tab_graph:    render_graph()
with tab_chat:     render_chat()
with tab_pipeline: render_pipeline()
