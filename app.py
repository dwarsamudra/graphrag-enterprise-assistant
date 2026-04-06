import os
from datetime import datetime

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="GraphRAG Enterprise Assistant",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
POLICY_FILE_PATH = "data/company_policy.txt"

# =========================================================
# VALIDATE ENV
# =========================================================

missing_env = []
if not NEO4J_URI:
    missing_env.append("NEO4J_URI")
if not NEO4J_USERNAME:
    missing_env.append("NEO4J_USERNAME")
if not NEO4J_PASSWORD:
    missing_env.append("NEO4J_PASSWORD")

if missing_env:
    st.error(f"Missing environment variables: {', '.join(missing_env)}")
    st.stop()

# =========================================================
# CONNECT TO NEO4J
# =========================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("⚙️ Settings")

    top_k = st.slider("Top K Vector Chunks", min_value=1, max_value=5, value=3)
    graph_limit = st.slider("Related Graph Policies", min_value=1, max_value=5, value=2)

    show_all_chunks = st.checkbox("Show document chunks", value=False)
    show_neo4j_data = st.checkbox("Show Neo4j policies", value=False)
    show_graph_context = st.checkbox("Show graph retrieval", value=True)
    show_final_context = st.checkbox("Show final LLM context", value=False)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

# =========================================================
# NEO4J FUNCTIONS
# =========================================================

def clear_graph(tx):
    tx.run("MATCH (n:Policy) DETACH DELETE n")


def create_policy_node(tx, name, content, index):
    tx.run(
        """
        MERGE (p:Policy {name: $name})
        SET p.content = $content,
            p.index = $index
        """,
        name=name,
        content=content,
        index=index
    )


def create_next_relationship(tx, current_name, next_name):
    tx.run(
        """
        MATCH (a:Policy {name: $current_name})
        MATCH (b:Policy {name: $next_name})
        MERGE (a)-[:NEXT]->(b)
        """,
        current_name=current_name,
        next_name=next_name
    )


def create_related_relationship(tx, name_a, name_b):
    tx.run(
        """
        MATCH (a:Policy {name: $name_a})
        MATCH (b:Policy {name: $name_b})
        WHERE a.name <> b.name
        MERGE (a)-[:RELATED_TO]-(b)
        """,
        name_a=name_a,
        name_b=name_b
    )


def get_policies(tx):
    result = tx.run(
        """
        MATCH (p:Policy)
        RETURN p.name AS name, p.content AS content, p.index AS index
        ORDER BY p.index
        """
    )
    return [record for record in result]


def get_related_policies(tx, policy_names, limit_count):
    result = tx.run(
        """
        MATCH (p:Policy)-[:RELATED_TO]-(related:Policy)
        WHERE p.name IN $policy_names
        RETURN DISTINCT related.name AS name, related.content AS content
        LIMIT $limit_count
        """,
        policy_names=policy_names,
        limit_count=limit_count
    )
    return [record for record in result]

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def load_policy_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_into_chunks(text):
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def extract_policy_title(chunk, index):
    if ":" in chunk:
        return chunk.split(":", 1)[0].strip()
    return f"Policy {index + 1}"


def keyword_set(text):
    words = (
        text.lower()
        .replace(":", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .split()
    )

    stopwords = {
        "the", "is", "are", "a", "an", "of", "to", "and", "in", "for",
        "with", "on", "by", "be", "this", "that", "from", "as", "at",
        "or", "it", "employees", "employee", "policy", "must", "may",
        "can", "should", "into", "their", "them", "been", "have",
        "will", "your", "only", "used"
    }

    return {w for w in words if len(w) > 3 and w not in stopwords}


def build_related_edges(session, chunks):
    titles = [extract_policy_title(chunk, i) for i, chunk in enumerate(chunks)]
    keyword_maps = [keyword_set(chunk) for chunk in chunks]

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            overlap = keyword_maps[i].intersection(keyword_maps[j])
            if len(overlap) >= 2:
                session.execute_write(create_related_relationship, titles[i], titles[j])


def build_graph(chunks):
    with driver.session() as session:
        session.execute_write(clear_graph)

        titles = []
        for i, chunk in enumerate(chunks):
            title = extract_policy_title(chunk, i)
            titles.append(title)
            session.execute_write(create_policy_node, title, chunk, i)

        for i in range(len(titles) - 1):
            session.execute_write(create_next_relationship, titles[i], titles[i + 1])

        build_related_edges(session, chunks)


def generate_embedding(text):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()

    if "embedding" not in data:
        raise ValueError("Embedding API response missing 'embedding' key.")

    return data["embedding"]


def generate_answer(context, question, chat_history):
    history_text = "\n".join(
        [
            f"User: {item['question']}\nAssistant: {item['answer']}"
            for item in chat_history[-3:]
        ]
    )

    prompt = f"""
You are an enterprise policy assistant.

Rules:
- Answer only from the provided context.
- If the answer is not found, say: "I could not find that in the provided policy context."
- Be concise, clear, and professional.
- If possible, mention which policy or policies support the answer.

Previous chat history:
{history_text}

Context:
{context}

Question:
{question}
"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise ValueError("Generation API response missing 'response' key.")

    return data["response"]


def compute_confidence(similarities, top_indices):
    if len(top_indices) == 0:
        return "Low"

    top_scores = [similarities[i] for i in top_indices]
    avg_score = float(np.mean(top_scores))

    if avg_score >= 0.80:
        return "High"
    if avg_score >= 0.60:
        return "Medium"
    return "Low"


def build_report(question, answer, confidence, vector_titles, related_titles):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    vector_section = "\n".join([f"- {title}" for title in vector_titles]) if vector_titles else "- None"
    related_section = "\n".join([f"- {title}" for title in related_titles]) if related_titles else "- None"

    report = f"""GraphRAG Enterprise Assistant Report
Generated at: {timestamp}

Question:
{question}

Answer:
{answer}

Confidence:
{confidence}

Top Vector Policies:
{vector_section}

Related Graph Policies:
{related_section}
"""
    return report.strip()


@st.cache_data(show_spinner=False)
def get_embeddings_cached(chunks):
    return [generate_embedding(chunk) for chunk in chunks]

# =========================================================
# MAIN UI
# =========================================================

st.title("GraphRAG Enterprise Assistant 🚀")
st.write("Hybrid retrieval using Neo4j graph + vector similarity + local LLM")

question = st.text_input("Ask a question about company policy:")

# =========================================================
# LOAD DOCUMENT
# =========================================================

try:
    data = load_policy_file(POLICY_FILE_PATH)
    st.success("Policy file loaded successfully ✅")
except Exception as e:
    st.error(f"File Loading Error: {e}")
    st.stop()

chunks = split_into_chunks(data)

if not chunks:
    st.error("No chunks found in company_policy.txt")
    st.stop()

if show_all_chunks:
    st.subheader("📄 Document Chunks")
    for i, chunk in enumerate(chunks, start=1):
        st.write(f"**Chunk {i}:**")
        st.write(chunk)
        st.write("---")

# =========================================================
# BUILD GRAPH
# =========================================================

try:
    build_graph(chunks)
    st.success("Neo4j graph built successfully ✅")
except Exception as e:
    st.error(f"Neo4j Build Error: {e}")
    st.stop()

# =========================================================
# READ GRAPH DATA
# =========================================================

try:
    with driver.session() as session:
        records = session.execute_read(get_policies)

    if show_neo4j_data:
        st.subheader("🧠 Policies from Neo4j")
        for record in records:
            st.write(f"**Policy:** {record['name']}")
            st.write(record["content"])
            st.write("---")
except Exception as e:
    st.error(f"Neo4j Read Error: {e}")
    st.stop()

# =========================================================
# EMBEDDINGS
# =========================================================

try:
    with st.spinner("Generating embeddings..."):
        embeddings = get_embeddings_cached(chunks)
    st.success(f"Embeddings created: {len(embeddings)} ✅")
except Exception as e:
    st.error(f"Embedding Error: {e}")
    st.stop()

# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:
    try:
        with st.spinner("Retrieving vector and graph context..."):
            question_embedding = generate_embedding(question)

            similarities = [
                cosine_similarity(question_embedding, emb)
                for emb in embeddings
            ]

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_chunks = [chunks[i] for i in top_indices]
            top_titles = [extract_policy_title(chunks[i], i) for i in top_indices]

            with driver.session() as session:
                related_records = session.execute_read(
                    get_related_policies,
                    top_titles,
                    graph_limit
                )

            related_chunks = [record["content"] for record in related_records]
            related_titles = [record["name"] for record in related_records]

            vector_context = "\n\n".join(top_chunks)
            graph_context = "\n\n".join(related_chunks)

            final_context_parts = []

            if vector_context.strip():
                final_context_parts.append("VECTOR CONTEXT:\n" + vector_context)

            if graph_context.strip():
                final_context_parts.append("GRAPH CONTEXT:\n" + graph_context)

            final_context = "\n\n".join(final_context_parts)
            confidence = compute_confidence(similarities, top_indices)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Top Vector Policies")
            for rank, idx in enumerate(top_indices, start=1):
                title = extract_policy_title(chunks[idx], idx)
                st.write(f"**Top {rank}: {title}**")
                st.caption(f"Similarity Score: {similarities[idx]:.4f}")
                st.write(chunks[idx])
                st.write("---")

        with col2:
            if show_graph_context:
                st.subheader("🕸️ Related Graph Policies")
                if related_records:
                    for i, record in enumerate(related_records, start=1):
                        st.write(f"**Related {i}: {record['name']}**")
                        st.write(record["content"])
                        st.write("---")
                else:
                    st.info("No RELATED_TO graph matches found.")

        if show_final_context:
            st.subheader("🎯 Final Context Sent to LLM")
            st.write(final_context)

        with st.spinner("Thinking... 🤖"):
            answer = generate_answer(
                final_context,
                question,
                st.session_state.chat_history
            )

        st.subheader("🤖 AI Answer")
        st.write(answer)

        st.subheader("📌 Answer Metadata")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Source Policies:** {', '.join(top_titles) if top_titles else 'None'}")

        report_text = build_report(
            question=question,
            answer=answer,
            confidence=confidence,
            vector_titles=top_titles,
            related_titles=related_titles
        )

        st.download_button(
            label="📥 Download Answer Report",
            data=report_text,
            file_name="graphrag_answer_report.txt",
            mime="text/plain"
        )

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": top_titles
            }
        )

    except Exception as e:
        st.error(f"Question Processing Error: {e}")

# =========================================================
# CHAT HISTORY
# =========================================================

if st.session_state.chat_history:
    st.subheader("🗂️ Chat History")

    for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"Q{i}: {item['question']}"):
            st.write(f"**Answer:** {item['answer']}")
            st.write(f"**Confidence:** {item['confidence']}")
            st.write(
                f"**Sources:** {', '.join(item['sources']) if item['sources'] else 'None'}"
            )