"""pages/chat.py — Chat with your discovered paper graph"""
import streamlit as st
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.engine import MOCK_CHAT_RESPONSES

SUGGESTED = [
    "Which papers are most connected in this graph?",
    "What are the key concepts linking these papers?",
    "Summarise how this research area has evolved over time",
    "Which paper should I read first as a beginner?",
    "Find papers most similar to the Transformer paper",
    "What are the main research clusters here?",
]


def render():
    if not st.session_state.get("graph_loaded"):
        st.info("🔍 Search for a topic on the **Discover** tab first — then come back to chat about the graph!")
        return

    query = st.session_state.get("last_query", "AI/ML papers")

    st.markdown(f"""
<div style="background:#0d1422;border:1px solid #1c2d47;border-radius:8px;
            padding:10px 16px;font-size:.8rem;margin-bottom:16px">
  💬 Chatting about the graph for: <strong style="color:#3b82f6">{query}</strong>
  · {len(st.session_state.get('result_papers', []))} papers
  · {len(st.session_state.get('result_edges', []))} connections
</div>
""", unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role":    "assistant",
                "content": (
                    f"I've analysed the graph for **{query}**. "
                    f"I can see {len(st.session_state.get('result_papers',[]))} papers "
                    f"with {len(st.session_state.get('result_edges',[]))} concept-based connections.\n\n"
                    "Ask me anything — which papers are most central, what connects them, "
                    "where to start reading, or deeper questions about specific papers."
                ),
                "meta": None,
            }
        ]
    if "mock_idx" not in st.session_state:
        st.session_state.mock_idx = 0

    col_chat, col_side = st.columns([3, 1])

    with col_side:
        st.markdown("<div style='font-size:.8rem;color:#5a7090;margin-bottom:8px'>Suggested questions</div>", unsafe_allow_html=True)
        for q in SUGGESTED:
            if st.button(q, key=f"sug_{q[:20]}", use_container_width=True):
                st.session_state._inject_q = q

        st.markdown("---")
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("<div style='font-size:.8rem;color:#5a7090;margin-bottom:6px'>Last query</div>", unsafe_allow_html=True)
        last = st.session_state.chat_messages[-1] if st.session_state.chat_messages else None
        if last and last.get("meta"):
            m = last["meta"]
            st.markdown(
                f"<div style='font-family:\"DM Mono\",monospace;font-size:.72rem;"
                f"background:#0d1422;border:1px solid #1c2d47;border-radius:6px;"
                f"padding:8px;color:#5a7090'>{m.get('cypher','')}</div>",
                unsafe_allow_html=True,
            )

    with col_chat:
        # Messages
        chat_box = st.container(height=440)
        with chat_box:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"], avatar="🗺️" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

        # Input
        injected = st.session_state.pop("_inject_q", None)
        prompt   = st.chat_input("Ask about the paper graph…") or injected

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt, "meta": None})

            with st.spinner("Querying graph + generating answer…"):
                steps = [
                    "🔍 Generating Cypher query…",
                    "🕸 Traversing Neo4j graph…",
                    "📊 Running vector similarity search…",
                    "🤖 Composing answer…",
                ]
                stat = st.status("Running GraphRAG pipeline…", expanded=True)
                with stat:
                    for step in steps:
                        st.write(step)
                        time.sleep(0.3)
                stat.update(label="✅ Done", state="complete")

                # Mock (replace with: GraphRAGEngine().query(prompt))
                resp = MOCK_CHAT_RESPONSES[st.session_state.mock_idx % len(MOCK_CHAT_RESPONSES)]
                st.session_state.mock_idx += 1

            st.session_state.chat_messages.append({
                "role":    "assistant",
                "content": resp["answer"],
                "meta":    {"cypher": resp["cypher"]},
            })
            st.rerun()
