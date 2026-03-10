"""
pages/discover.py
─────────────────
Search by topic keyword OR upload a PDF.
Fetches REAL papers from arXiv, extracts concepts, builds edges.
"""

import streamlit as st
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.engine import MOCK_PAPERS, MOCK_EDGES, MOCK_CONCEPTS


def render():
    col_search, col_pdf = st.columns([1, 1], gap="large")

    with col_search:
        st.markdown("""
<div style="margin-bottom:12px">
  <div style="font-size:1.05rem;font-weight:600;margin-bottom:4px">🔍 Search by topic</div>
  <div style="font-size:.8rem;color:#5a7090">Fetches real papers from arXiv and maps their connections</div>
</div>
""", unsafe_allow_html=True)

        topic = st.text_input("Topic", placeholder="e.g. graph neural networks, diffusion models, RAG",
                              label_visibility="collapsed", key="topic_input")

        col_n, col_go = st.columns([2, 1])
        with col_n:
            n_papers = st.slider("Papers to fetch", 5, 40, 15, key="n_papers_slider")
        with col_go:
            st.markdown("<div style='padding-top:28px'>", unsafe_allow_html=True)
            search_clicked = st.button("Search arXiv", use_container_width=True, key="search_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.75rem;color:#5a7090;margin:8px 0 4px'>Quick topics:</div>", unsafe_allow_html=True)
        quick_topics = ["transformers", "diffusion models", "graph neural networks", "RAG", "LoRA"]
        qt_cols = st.columns(len(quick_topics))
        for i, qt in enumerate(quick_topics):
            with qt_cols[i]:
                if st.button(qt, key=f"qt_{i}", use_container_width=True):
                    st.session_state._run_search = qt
                    st.rerun()

    with col_pdf:
        st.markdown("""
<div style="margin-bottom:12px">
  <div style="font-size:1.05rem;font-weight:600;margin-bottom:4px">📎 Upload a paper PDF</div>
  <div style="font-size:.8rem;color:#5a7090">Extract its concepts, then find related papers on arXiv</div>
</div>
""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop PDF here", type=["pdf"],
                                    label_visibility="collapsed", key="pdf_upload")
        if uploaded:
            st.markdown(f"""
<div style="background:#0d1422;border:1px solid #1c2d47;border-radius:8px;
            padding:12px 14px;font-size:.85rem;margin-bottom:10px">
  <div style="font-weight:600">{uploaded.name}</div>
  <div style="color:#5a7090">{uploaded.size // 1024} KB</div>
</div>""", unsafe_allow_html=True)

            if st.button("Parse and find related papers", use_container_width=True, key="pdf_btn"):
                from utils.engine import PDFParser
                with st.spinner("Parsing PDF..."):
                    try:
                        paper    = PDFParser().parse(uploaded.read())
                        keywords = " ".join(paper.get("concepts", [])[:4]) or paper["title"]
                        st.success(f"Found: {paper['title'][:60]}")
                        st.session_state._run_search = keywords
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not parse PDF: {e}")

    st.markdown("---")

    run_query = st.session_state.pop("_run_search", None)
    if search_clicked and topic:
        run_query = topic

    if run_query:
        _run_discovery(run_query, n_papers)
        return

    if st.session_state.get("graph_loaded"):
        _show_results_preview()
    else:
        _show_empty_state()


def _run_discovery(query: str, n_papers: int):
    from utils.engine import ArxivFetcher, ConceptExtractor

    progress_bar = st.progress(0)
    status = st.empty()

    def update(pct, msg):
        progress_bar.progress(min(float(pct), 1.0))
        status.markdown(
            f"<div style='font-size:.85rem;color:#5a7090;font-family:\"DM Mono\",monospace'>{msg}</div>",
            unsafe_allow_html=True)

    update(0.05, f"Searching arXiv for {query}...")
    try:
        papers = ArxivFetcher().search(query, max_results=n_papers, progress=update)
    except Exception as e:
        st.error(f"arXiv fetch failed: {e}")
        papers = MOCK_PAPERS

    if not papers:
        st.warning("No papers found. Try a different search term.")
        progress_bar.empty()
        status.empty()
        return

    update(0.55, f"Fetched {len(papers)} papers. Extracting concepts...")
    openai_key = st.session_state.get("openai_key", "")
    try:
        extractor = ConceptExtractor(openai_api_key=openai_key)
        papers = extractor.extract(papers, progress=update)
    except Exception:
        papers = ConceptExtractor(openai_api_key="").extract(papers)

    update(0.88, "Computing connections between papers...")
    edges    = _build_edges(papers)
    concepts = _count_concepts(papers)

    update(1.0, f"Done! {len(papers)} papers, {len(edges)} connections found")
    time.sleep(0.8)
    progress_bar.empty()
    status.empty()

    st.session_state.graph_loaded      = True
    st.session_state.last_query        = query
    st.session_state.result_papers     = papers
    st.session_state.result_edges      = edges
    st.session_state.result_concepts   = concepts
    st.session_state.graph_paper_count = len(papers)
    st.session_state.graph_edge_count  = len(edges)

    _show_results_preview()


def _build_edges(papers):
    edges = []
    for i, p1 in enumerate(papers):
        for j, p2 in enumerate(papers):
            if j <= i:
                continue
            c1     = set(c.lower() for c in p1.get("concepts", []))
            c2     = set(c.lower() for c in p2.get("concepts", []))
            shared = c1 & c2
            if shared:
                edges.append({"from": p1["id"], "to": p2["id"], "weight": len(shared)})
    return edges


def _count_concepts(papers):
    counts = {}
    for p in papers:
        for c in p.get("concepts", []):
            key = c.lower()
            counts[key] = counts.get(key, 0) + 1
    return [{"name": k, "paper_count": v}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])][:20]


def _show_results_preview():
    papers   = st.session_state.get("result_papers", [])
    edges    = st.session_state.get("result_edges", [])
    concepts = st.session_state.get("result_concepts", [])
    query    = st.session_state.get("last_query", "")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Papers found", len(papers))
    with c2: st.metric("Connections",  len(edges))
    with c3: st.metric("Concepts",     len(concepts))
    with c4: st.metric("Year range",
                       f"{min(p['year'] for p in papers)} - {max(p['year'] for p in papers)}"
                       if papers else "-")

    st.markdown(f"""
<div style="background:#0d1422;border:1px solid #1c2d47;border-radius:8px;
            padding:10px 16px;font-size:.8rem;color:#5a7090;margin:10px 0">
  Graph built for <strong style="color:#e2eaf6">{query}</strong>.
  Go to <strong style="color:#3b82f6">Graph Explorer</strong> to see the visual map.
</div>""", unsafe_allow_html=True)

    if concepts:
        pills = " ".join(
            f"<span style='display:inline-block;padding:3px 10px;border-radius:20px;"
            f"background:#22d3a510;color:#22d3a5;border:1px solid #22d3a530;"
            f"font-size:.75rem;margin:2px;font-family:\"DM Mono\",monospace'>"
            f"{c['name']} x{c['paper_count']}</span>"
            for c in concepts[:15]
        )
        st.markdown(f"<div style='margin-bottom:14px'>{pills}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:.9rem;font-weight:600;margin-bottom:10px'>Papers ({len(papers)})</div>",
                unsafe_allow_html=True)
    for paper in papers:
        _paper_card(paper, edges)


def _paper_card(paper: dict, all_edges: list):
    conn = sum(1 for e in all_edges if e["from"] == paper["id"] or e["to"] == paper["id"])
    pills = " ".join(
        f"<span style='padding:1px 7px;border-radius:12px;background:#3b82f620;"
        f"color:#3b82f6;border:1px solid #3b82f640;font-size:.7rem;"
        f"font-family:\"DM Mono\",monospace'>{c}</span>"
        for c in paper.get("concepts", [])[:5]
    )
    with st.expander(f"**{paper['title']}** - {paper['year']}", expanded=False):
        col_meta, col_abs = st.columns([1, 2])
        with col_meta:
            authors = paper.get("authors", [])
            st.markdown(f"""
<div style="font-size:.8rem;line-height:1.8;color:#5a7090">
  <div>👤 {', '.join(authors[:2])}{'...' if len(authors) > 2 else ''}</div>
  <div>📅 {paper['year']}</div>
  <div style="color:#22d3a5">🔗 {conn} connection{'s' if conn != 1 else ''}</div>
  <div style="margin-top:8px">{pills}</div>
</div>""", unsafe_allow_html=True)
        with col_abs:
            st.markdown(
                f"<div style='font-size:.82rem;line-height:1.65;color:#94a3b8'>"
                f"{paper.get('abstract','')[:400]}...</div>",
                unsafe_allow_html=True)
        if paper.get("url"):
            st.markdown(f"<a href='{paper['url']}' target='_blank' "
                        f"style='font-size:.75rem;color:#3b82f6'>View on arXiv</a>",
                        unsafe_allow_html=True)


def _show_empty_state():
    st.markdown("""
<div style="text-align:center;padding:60px 20px">
  <div style="font-size:3rem;margin-bottom:16px">🗺️</div>
  <div style="font-size:1.4rem;margin-bottom:8px">Start by searching a topic</div>
  <div style="color:#5a7090;font-size:.875rem;max-width:420px;margin:0 auto;line-height:1.7">
    Type a research topic above or upload a PDF. PaperMap fetches real papers from arXiv,
    extracts key concepts, and builds a visual map of how they connect.
  </div>
</div>""", unsafe_allow_html=True)
