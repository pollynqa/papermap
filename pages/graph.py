"""
pages/graph.py
──────────────
Interactive graph explorer using vis.js.
Nodes = papers, edges = shared concepts.
Click a node → see its neighbours + concepts.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.engine import MOCK_PAPERS, MOCK_EDGES


# ─────────────────────────────────────────────────────────────────────────────
def render():
    papers = st.session_state.get("result_papers", MOCK_PAPERS)
    edges  = st.session_state.get("result_edges",  MOCK_EDGES)

    if not papers:
        st.info("Search for a topic on the **Discover** tab first to populate the graph.")
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    col_filter, col_search, col_layout = st.columns([2, 2, 1])
    with col_filter:
        year_range = st.slider(
            "Year range",
            min_value=min(p["year"] for p in papers),
            max_value=max(p["year"] for p in papers),
            value=(min(p["year"] for p in papers), max(p["year"] for p in papers)),
            key="year_range",
        )
    with col_search:
        highlight = st.text_input(
            "🔍 Highlight paper",
            placeholder="Type part of a title…",
            key="graph_search",
            label_visibility="collapsed",
        )
    with col_layout:
        layout = st.selectbox("Layout", ["force", "hierarchical"], key="graph_layout",
                              label_visibility="collapsed")

    # Filter
    filtered_papers = [p for p in papers if year_range[0] <= p["year"] <= year_range[1]]
    filtered_ids    = {p["id"] for p in filtered_papers}
    filtered_edges  = [e for e in edges if e["from"] in filtered_ids and e["to"] in filtered_ids]

    # ── Graph HTML ────────────────────────────────────────────────────────────
    graph_html = _build_graph_html(filtered_papers, filtered_edges, highlight, layout)
    components.html(graph_html, height=560, scrolling=False)

    # ── Bottom panels ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_stats, col_top, col_legend = st.columns([1, 2, 1])

    with col_stats:
        st.markdown("**📊 Graph stats**")
        st.metric("Nodes (papers)", len(filtered_papers))
        st.metric("Edges (connections)", len(filtered_edges))
        avg_conn = len(filtered_edges) * 2 / len(filtered_papers) if filtered_papers else 0
        st.metric("Avg connections / paper", f"{avg_conn:.1f}")

    with col_top:
        st.markdown("**🔗 Most connected papers**")
        conn_count = {}
        for e in filtered_edges:
            conn_count[e["from"]] = conn_count.get(e["from"], 0) + 1
            conn_count[e["to"]]   = conn_count.get(e["to"],   0) + 1
        top_papers = sorted(conn_count.items(), key=lambda x: -x[1])[:5]
        paper_map  = {p["id"]: p for p in filtered_papers}
        for pid, cnt in top_papers:
            p = paper_map.get(pid)
            if p:
                st.markdown(
                    f"<div style='padding:6px 0;border-bottom:1px solid #1c2d47;"
                    f"font-size:.82rem'>"
                    f"<span style='color:#e2eaf6'>{p['title'][:55]}…</span> "
                    f"<span style='color:#3b82f6;font-family:\"DM Mono\",monospace;"
                    f"font-size:.75rem'>{cnt} links</span></div>",
                    unsafe_allow_html=True,
                )

    with col_legend:
        st.markdown("**Legend**")
        legend_items = [
            ("#3b82f6", "Recent paper"),
            ("#22d3a5", "Older paper"),
            ("#f59e0b", "Highlighted"),
        ]
        for color, label in legend_items:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;"
                f"font-size:.8rem;margin-bottom:6px'>"
                f"<div style='width:12px;height:12px;border-radius:50%;"
                f"background:{color}'></div>{label}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='font-size:.75rem;color:#5a7090;margin-top:8px'>"
            "Edge thickness = shared concepts</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
def _build_graph_html(papers, edges, highlight, layout):
    paper_map = {p["id"]: p for p in papers}
    max_year  = max((p["year"] for p in papers), default=2024)
    min_year  = min((p["year"] for p in papers), default=2017)

    def year_color(year):
        if max_year == min_year:
            return "#3b82f6"
        t = (year - min_year) / (max_year - min_year)
        # interpolate: old=#22d3a5 (teal), new=#3b82f6 (blue)
        r = int(0x22 + t * (0x3b - 0x22))
        g = int(0xd3 + t * (0x82 - 0xd3))
        b = int(0xa5 + t * (0xf6 - 0xa5))
        return f"#{r:02x}{g:02x}{b:02x}"

    vis_nodes = []
    for p in papers:
        is_hl   = highlight and highlight.lower() in p["title"].lower()
        color   = "#f59e0b" if is_hl else year_color(p["year"])
        label   = p["title"][:30] + ("…" if len(p["title"]) > 30 else "")
        conn    = sum(1 for e in edges if e["from"] == p["id"] or e["to"] == p["id"])
        size    = 16 + conn * 4

        vis_nodes.append({
            "id":    p["id"],
            "label": label,
            "title": f"{p['title']} ({p['year']})\n{len(p.get('concepts',[]))} concepts",
            "color": {
                "background": color + "33",
                "border":     color,
                "highlight":  {"background": color + "55", "border": "#fff"},
            },
            "font":        {"color": color, "size": 11, "face": "DM Sans, sans-serif"},
            "size":        size,
            "borderWidth": 2,
            "shape":       "dot",
        })

    vis_edges = []
    for e in edges:
        w = e.get("weight", 1)
        vis_edges.append({
            "from":  e["from"],
            "to":    e["to"],
            "width": 1 + w * 0.8,
            "color": {"color": "#1c2d47", "highlight": "#3b82f6", "opacity": 0.9},
            "smooth": {"type": "continuous"},
            "title": f"{w} shared concept{'s' if w > 1 else ''}",
        })

    physics = {
        "force": {
            "enabled": True,
            "barnesHut": {"gravitationalConstant": -5000, "springLength": 140, "springConstant": 0.04},
            "stabilization": {"iterations": 120},
        },
        "hierarchical": {
            "enabled": True,
            "hierarchical": {"direction": "UD", "sortMethod": "directed", "nodeSpacing": 120, "levelSeparation": 130},
        },
    }[layout]

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body  {{ background: #080c14; overflow: hidden; font-family: 'DM Sans', sans-serif; }}
  #net  {{ width: 100%; height: 560px; }}
  #tooltip {{
    position: absolute; pointer-events: none;
    background: #0d1422; border: 1px solid #1c2d47; border-radius: 8px;
    padding: 10px 14px; font-size: 12px; color: #e2eaf6;
    max-width: 240px; display: none; z-index: 100; line-height: 1.5;
  }}
  #tooltip .t-title {{ color: #3b82f6; font-weight: 600; margin-bottom: 4px; font-size: 12px; }}
  #tooltip .t-year  {{ color: #5a7090; font-size: 11px; }}
  #tooltip .t-conc  {{ margin-top: 6px; }}
  #tooltip .tag {{
    display: inline-block; padding: 1px 6px; border-radius: 10px;
    background: #22d3a510; color: #22d3a5; border: 1px solid #22d3a530;
    font-size: 10px; margin: 2px 1px; font-family: 'DM Mono', monospace;
  }}
  #hint {{ position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
           font-size: 11px; color: #2a3d55; pointer-events: none; }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet">
<link  href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600&family=DM+Mono&display=swap" rel="stylesheet">
</head>
<body>
<div id="net"></div>
<div id="tooltip"></div>
<div id="hint">Click a node to explore · scroll to zoom · drag to pan</div>

<script>
const paperData = {json.dumps({p["id"]: p for p in papers})};

const nodes = new vis.DataSet({json.dumps(vis_nodes)});
const edges = new vis.DataSet({json.dumps(vis_edges)});

const options = {{
  physics: {json.dumps(physics)},
  interaction: {{ hover: true, tooltipDelay: 0, zoomView: true, dragView: true }},
  nodes: {{
    shadow: {{ enabled: true, color: "#00000066", size: 10, x: 2, y: 2 }},
  }},
  edges: {{ shadow: false, selectionWidth: 2 }},
}};

const network = new vis.Network(
  document.getElementById("net"),
  {{ nodes, edges }},
  options
);

const tooltip = document.getElementById("tooltip");

network.on("hoverNode", function(params) {{
  const id   = params.node;
  const p    = paperData[id];
  if (!p) return;
  const tags = (p.concepts || []).slice(0,5).map(c =>
    `<span class="tag">${{c}}</span>`).join(" ");
  tooltip.innerHTML = `
    <div class="t-title">${{p.title}}</div>
    <div class="t-year">${{p.authors ? p.authors.slice(0,2).join(", ") : ""}} · ${{p.year}}</div>
    <div class="t-conc">${{tags}}</div>
  `;
  tooltip.style.display = "block";
  document.getElementById("net").style.cursor = "pointer";
}});

network.on("blurNode", function() {{
  tooltip.style.display = "none";
  document.getElementById("net").style.cursor = "default";
}});

network.on("mouseMoveTitle", function(params) {{
  tooltip.style.left = (params.event.clientX + 16) + "px";
  tooltip.style.top  = (params.event.clientY - 10) + "px";
}});

document.getElementById("net").addEventListener("mousemove", function(e) {{
  tooltip.style.left = (e.clientX + 16) + "px";
  tooltip.style.top  = (e.clientY - 10) + "px";
}});

network.on("click", function(params) {{
  if (params.nodes.length === 0) return;
  const id = params.nodes[0];
  const p  = paperData[id];
  if (p && p.url) {{
    // highlight neighbours
    const connectedEdges = network.getConnectedEdges(id);
    const connectedNodes = network.getConnectedNodes(id);
    network.selectNodes([id, ...connectedNodes]);
    network.selectEdges(connectedEdges);
  }}
}});

network.on("stabilizationIterationsDone", function() {{
  network.setOptions({{ physics: {{ enabled: false }} }});
}});
</script>
</body>
</html>"""
