"""
ui/app.py — Streamlit UI.

Features:
  - Thread selector dropdown
  - Chat interface with streaming-style display
  - "Search outside thread" toggle
  - Agent debug panel:
      entity register state
      which agents fired + latency
      resolved query / step-back query
      retrieved chunk ids + scores
      grounding score
      inline citations
"""

import os
import time
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Email RAG",
    page_icon="✉️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────

if "session_id"       not in st.session_state: st.session_state.session_id       = None
if "thread_id"        not in st.session_state: st.session_state.thread_id        = None
if "messages"         not in st.session_state: st.session_state.messages         = []
if "last_debug"       not in st.session_state: st.session_state.last_debug       = None
if "entity_register"  not in st.session_state: st.session_state.entity_register  = {}


# ──────────────────────────────────────────────
# Sidebar — thread selector + controls
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("✉️ Email RAG")
    st.markdown("---")

    # Load thread list
    try:
        threads_resp = requests.get(f"{API_BASE}/threads", timeout=3)
        available_threads = threads_resp.json().get("threads", [])
    except Exception:
        available_threads = []

    if not available_threads:
        st.warning("No threads found. Run `python ingest.py` first.")
        thread_input = st.text_input("Thread ID (manual)", value="T-0001")
        available_threads = [thread_input]

    selected_thread = st.selectbox("Select thread", available_threads)

    search_outside = st.toggle("🔍 Search outside thread", value=False)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start / Switch", use_container_width=True):
            if st.session_state.session_id is None:
                # New session
                resp = requests.post(f"{API_BASE}/start_session",
                                     json={"thread_id": selected_thread})
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.thread_id  = selected_thread
                st.session_state.messages   = []
                st.session_state.entity_register = {}
                st.success(f"Session started: {selected_thread}")
            else:
                # Switch thread
                resp = requests.post(f"{API_BASE}/switch_thread",
                                     json={"session_id": st.session_state.session_id,
                                           "thread_id": selected_thread})
                st.session_state.thread_id = selected_thread
                st.info(f"Switched to {selected_thread}")

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            if st.session_state.session_id:
                requests.post(f"{API_BASE}/reset_session",
                              json={"session_id": st.session_state.session_id})
                st.session_state.messages = []
                st.session_state.entity_register = {}
                st.session_state.last_debug = None
                st.success("Session reset")

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session_id or 'none'}`")
    st.caption(f"Thread: `{st.session_state.thread_id or 'none'}`")
    st.caption(f"API: `{API_BASE}`")


# ──────────────────────────────────────────────
# Main layout: chat | debug panel
# ──────────────────────────────────────────────

chat_col, debug_col = st.columns([3, 2])

with chat_col:
    st.subheader("Chat")

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input(
        "Ask about this thread…",
        disabled=(st.session_state.session_id is None),
    ):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                t0 = time.time()
                try:
                    resp = requests.post(
                        f"{API_BASE}/ask",
                        json={
                            "session_id": st.session_state.session_id,
                            "text": prompt,
                            "search_outside_thread": search_outside,
                        },
                        timeout=120,
                    )
                    data = resp.json()
                    elapsed = round((time.time() - t0) * 1000)

                    answer = data.get("answer", "No answer returned.")
                    st.markdown(answer)
                    st.caption(f"⏱ {elapsed}ms · grounding: {data.get('grounding_score', 0):.0%}")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_debug = data

                    # Update entity register from API response
                    if data.get("entity_register"):
                        st.session_state.entity_register = data["entity_register"]

                except Exception as e:
                    st.error(f"API error: {e}")


with debug_col:
    st.subheader("🔍 Agent debug panel")

    dbg = st.session_state.last_debug

    if dbg is None:
        st.info("Ask a question to see agent internals here.")
    else:
        # ── Query rewrites ──
        with st.expander("Query rewrites", expanded=True):
            st.markdown(f"**Resolved query:** {dbg.get('rewrite', '—')}")
            if dbg.get("stepback"):
                st.markdown(f"**Step-back:** {dbg['stepback']}")

        # ── Agent trace ──
        with st.expander("Agent trace", expanded=True):
            for entry in dbg.get("agent_trace", []):
                agent = entry.get("agent", "")
                latency = entry.get("latency_ms", 0)
                routing = entry.get("routing", "→")

                # Colour-code by agent type
                if "entity" in agent:
                    icon = "🧠"
                elif "prefilter" in agent:
                    icon = "🔎"
                elif "expansion" in agent:
                    icon = "🔭"
                elif "breakdown" in agent:
                    icon = "✂️"
                elif "retrieval" in agent:
                    icon = "📦"
                elif "citation" in agent:
                    icon = "✅"
                elif "timeline" in agent:
                    icon = "📅"
                else:
                    icon = "⚙️"

                st.markdown(
                    f"{icon} **{agent}** &nbsp;`{latency:.0f}ms`&nbsp; → `{routing}`"
                )

                # Extra fields per agent
                if "grounding_score" in entry:
                    score = entry["grounding_score"]
                    colour = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
                    st.caption(f"  {colour} grounding: {score:.0%}")
                if "sub_queries" in entry and entry["sub_queries"]:
                    st.caption(f"  sub-queries: {entry['sub_queries']}")
                if entry.get("retrieval_insufficient"):
                    st.caption("  ⚠️ triggered retry")

        # ── Retrieved chunks ──
        with st.expander(f"Retrieved chunks ({len(dbg.get('retrieved', []))})", expanded=False):
            for i, chunk in enumerate(dbg.get("retrieved", []), 1):
                score = chunk.get("rerank_score", 0)
                bar_len = int(score * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                page_info = f", p.{chunk['page_no']}" if chunk.get("page_no") else ""
                st.markdown(
                    f"**{i}.** `{chunk['message_id']}{page_info}` &nbsp; "
                    f"`{bar}` {score:.3f}"
                )
                st.caption(chunk.get("preview", ""))

        # ── Citations ──
        with st.expander(f"Validated citations ({len(dbg.get('citations', []))})", expanded=False):
            for c in dbg.get("citations", []):
                page_str = f", page {c['page_no']}" if c.get("page_no") else ""
                st.markdown(
                    f"- **[msg: `{c['message_id']}`{page_str}]** "
                    f"confidence: {c['confidence']:.0%}"
                )
                st.caption(f"Claim: {c['claim_text'][:100]}…")

        # ── Entity register ──
        with st.expander("Entity register", expanded=False):
            register = st.session_state.entity_register
            if not register:
                st.caption("No entities resolved yet in this session.")
            else:
                for key, val in list(register.items())[:15]:
                    st.markdown(
                        f"- **{val.get('text', key)}** ({val.get('type', '?')}) "
                        f"— turn {val.get('turn', 0)}"
                    )

        # ── Grounding score gauge ──
        score = dbg.get("grounding_score", 0.0)
        colour = "normal" if score >= 0.8 else "off"
        st.metric(
            label="Grounding score",
            value=f"{score:.0%}",
            delta="✓ above threshold" if score >= 0.8 else "⚠ retry triggered",
            delta_color=colour,
        )

        st.caption(f"trace_id: `{dbg.get('trace_id', '—')}`")
