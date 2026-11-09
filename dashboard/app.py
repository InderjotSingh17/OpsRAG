
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.rag import create_vector_store, ask_question
from backend.observability import get_observability_summary
from backend.cost_tracker import get_cost_summary



st.set_page_config(
    page_title="OpsRAG | LLMOps Platform",
    page_icon="🤖",
    layout="wide"
)

COST_LOG = BASE_DIR / "experiments" / "cost_logs.jsonl"
OBSERVABILITY_LOG = BASE_DIR / "experiments" / "observability.jsonl"
COMPARISON_FILE = BASE_DIR / "experiments" / "controlled_comparison.json"

LANGSMITH_URL = (
    "https://smith.langchain.com/o/"
    "1cf94c8b-2528-440e-8f60-c415be63f4c9/"
    "datasets/14001c8f-9182-48ae-880b-813b3a9c9841/"
    "compare?selectedSessions=b46e8af4-9701-495c-9469-9bdd363e6fed"
)



def read_jsonl(path):
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                if line.strip():
                    records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records


def read_comparison():
    if not COMPARISON_FILE.exists():
        return []

    try:
        with COMPARISON_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def get_event_type(record):
    return record.get("event_type", record.get("event"))


@st.cache_resource(show_spinner="Loading documents and embeddings...")
def load_vector_store():
    return create_vector_store()



st.title("🤖 OpsRAG")
st.caption("Production RAG & LLMOps Monitoring Platform")

st.markdown(
    "Monitor your RAG application, evaluate experiments, "
    "track model usage and interact with your documents."
)

with st.sidebar:
    st.header("⚙️ Configuration")

    retrieval_k = st.slider(
        "Retrieved document chunks (K)",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.caption("Model: Gemini 2.5 Flash")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector DB: ChromaDB")

    if st.button("Refresh dashboard"):
        st.cache_data.clear()
        st.rerun()


(
    assistant_tab,
    overview_tab,
    cost_tab,
    evaluation_tab,
    experiments_tab,
    observability_tab
) = st.tabs([
    "💬 RAG Assistant",
    "📊 Overview",
    "💰 Cost Analytics",
    "🧪 Evaluation",
    "⚗️ Experiments",
    "🔍 Observability"
])


with assistant_tab:

    st.subheader("Ask your documents")

    st.write(
        "Ask questions about the indexed company policies "
        "and other documents."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input(
        "Your question",
        placeholder="How many paid leaves are employees entitled to?"
    )

    if st.button("Ask OpsRAG", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                vector_store = load_vector_store()

                with st.spinner("Retrieving documents and generating answer..."):
                    result = ask_question(
                        vector_store,
                        question,
                        retrieval_k=retrieval_k
                    )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "context": result["retrieved_context"]
                })

            except Exception as error:
                st.error(f"Request failed: {error}")

    for item in reversed(st.session_state.chat_history):

        with st.container(border=True):

            st.markdown("**You**")
            st.write(item["question"])

            st.markdown("**OpsRAG**")
            st.write(item["answer"])

            with st.expander("Retrieved sources and context"):

                if item["sources"]:
                    st.json(item["sources"])
                else:
                    st.info("No source documents returned.")

                st.text(item["context"])



with overview_tab:

    st.subheader("Request Analytics")

    summary = get_observability_summary()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Requests", summary["total_requests"])
    col2.metric("Successful", summary["successful_requests"])
    col3.metric("Rejected", summary["rejected_requests"])
    col4.metric("Failed", summary["failed_requests"])

    st.divider()

    st.subheader("Performance Monitoring")

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Request Latency",
        f"{summary['average_latency_seconds']:.2f}s"
    )

    cost_summary = get_cost_summary()

    col2.metric(
        "Total Tokens Used",
        f"{cost_summary.get('total_tokens', 0):,}"
    )

    events = read_jsonl(OBSERVABILITY_LOG)

    completed_events = [
        event for event in events
        if get_event_type(event) == "request_completed"
    ]

    if completed_events:

        performance_data = pd.DataFrame(completed_events)

        latency_columns = [
            column for column in [
                "retrieval_latency_seconds",
                "generation_latency_seconds",
                "total_latency_seconds"
            ]
            if column in performance_data.columns
        ]

        if latency_columns:
            st.markdown("**Request latency breakdown**")
            st.bar_chart(
                performance_data[latency_columns].reset_index(drop=True)
            )

    else:
        st.info("No completed request latency records found yet.")



with cost_tab:

    st.subheader("LLM Cost and Token Analytics")

    cost_summary = get_cost_summary()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model Requests",
        cost_summary.get("total_requests", 0)
    )

    col2.metric(
        "Input Tokens",
        f"{cost_summary.get('total_input_tokens', 0):,}"
    )

    col3.metric(
        "Output Tokens",
        f"{cost_summary.get('total_output_tokens', 0):,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Total Tokens",
        f"{cost_summary.get('total_tokens', 0):,}"
    )

    col5.metric(
        "Estimated Total Cost",
        f"${cost_summary.get('total_cost', 0):.6f}"
    )

    col6.metric(
        "Average Cost / Request",
        f"${cost_summary.get('average_cost_per_request', 0):.6f}"
    )

    st.divider()

    st.caption(
        "Cost estimates depend on the pricing configured in your .env file. "
        "A zero value means pricing is not configured; it does not mean "
        "the model API is necessarily free."
    )

    cost_records = read_jsonl(COST_LOG)

    if cost_records:
        cost_df = pd.DataFrame(cost_records)

        token_columns = [
            column for column in [
                "input_tokens",
                "output_tokens",
                "total_tokens"
            ]
            if column in cost_df.columns
        ]

        if token_columns:
            st.markdown("**Token usage records**")
            st.line_chart(cost_df[token_columns].reset_index(drop=True))

    else:
        st.info("No cost records found.")



with evaluation_tab:

    st.subheader("RAG Evaluation Results")

    comparison = read_comparison()

    if comparison:

        st.info(
            "These metrics are from your controlled experiment runs. "
            "The faithfulness and retrieval values are similarity proxies, "
            "not claim-level verification."
        )

        evaluation_rows = []

        for run in comparison:
            metrics = run.get("metrics", {})

            evaluation_rows.append({
                "Configuration": run.get("configuration"),
                "Correctness": metrics.get("avg_correctness"),
                "Answer Relevance": metrics.get("avg_answer_relevance"),
                "Faithfulness Proxy": metrics.get(
                    "avg_faithfulness_similarity_proxy"
                ),
                "Retrieval Proxy": metrics.get(
                    "avg_retrieval_similarity_proxy"
                ),
                "Test Cases": metrics.get("completed_test_cases")
            })

        evaluation_df = pd.DataFrame(evaluation_rows)

        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**Evaluation metric comparison**")

        chart_df = evaluation_df.set_index("Configuration")[
            [
                "Correctness",
                "Answer Relevance",
                "Faithfulness Proxy",
                "Retrieval Proxy"
            ]
        ]

        st.bar_chart(chart_df)

    else:
        st.warning(
            "Controlled comparison results not found. "
            "Run your experiment script first."
        )

    st.divider()

    st.markdown("**LangSmith evaluation workspace**")

    st.link_button(
        "Open LangSmith Evaluation",
        LANGSMITH_URL
    )

with experiments_tab:

    st.subheader("Controlled Experiment Comparison")

    comparison = read_comparison()

    if len(comparison) >= 2:

        experiment_rows = []

        for run in comparison:

            metrics = run.get("metrics", {})

            experiment_rows.append({
                "Configuration": run.get("configuration"),
                "Run ID": run.get("run_id"),
                "Chunks": run.get("chunk_count"),
                "Average Latency (s)": metrics.get(
                    "avg_latency_seconds"
                ),
                "Input Tokens": metrics.get("total_input_tokens"),
                "Output Tokens": metrics.get("total_output_tokens"),
                "Total Tokens": metrics.get("total_tokens"),
                "Estimated Cost ($)": metrics.get("estimated_cost_usd")
            })

        experiment_df = pd.DataFrame(experiment_rows)

        st.dataframe(
            experiment_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.markdown("**Average latency comparison**")

        latency_df = experiment_df.set_index("Configuration")[
            ["Average Latency (s)"]
        ]

        st.bar_chart(latency_df)

        st.markdown("**Token usage comparison**")

        token_df = experiment_df.set_index("Configuration")[
            ["Input Tokens", "Output Tokens"]
        ]

        st.bar_chart(token_df)

        st.markdown("**Experiment observations**")

        baseline = comparison[0]
        experiment = comparison[1]

        baseline_metrics = baseline["metrics"]
        experiment_metrics = experiment["metrics"]

        st.write(
            f"- Baseline indexed {baseline['chunk_count']} chunks; "
            f"Experiment B indexed {experiment['chunk_count']} chunks."
        )

        st.write(
            f"- Baseline average latency: "
            f"{baseline_metrics['avg_latency_seconds']:.2f}s."
        )

        st.write(
            f"- Experiment B average latency: "
            f"{experiment_metrics['avg_latency_seconds']:.2f}s."
        )

        st.write(
            f"- Baseline input tokens: "
            f"{baseline_metrics['total_input_tokens']:,}."
        )

        st.write(
            f"- Experiment B input tokens: "
            f"{experiment_metrics['total_input_tokens']:,}."
        )

        st.caption(
            "These are results from one recorded experiment run per "
            "configuration. Latency differences should not be treated "
            "as statistically established."
        )

    else:
        st.warning("Two experiment configurations are required.")



with observability_tab:

    st.subheader("Application Observability")

    events = read_jsonl(OBSERVABILITY_LOG)

    if events:

        events_df = pd.DataFrame(events)

        st.markdown("**Recent application events**")

        if "event_type" not in events_df.columns and "event" in events_df.columns:
            events_df["event_type"] = events_df["event"]

        display_columns = [
            column for column in [
                "timestamp",
                "event_type",
                "request_id",
                "stage",
                "status",
                "total_latency_seconds",
                "error_type"
            ]
            if column in events_df.columns
        ]

        if display_columns:
            st.dataframe(
                events_df[display_columns].tail(100),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(events_df.tail(100), use_container_width=True)

        event_counts = (
            events_df["event_type"].value_counts()
            if "event_type" in events_df.columns
            else None
        )

        if event_counts is not None:
            st.markdown("**Event distribution**")
            st.bar_chart(event_counts)

    else:
        st.info("No observability events recorded yet.")



st.divider()

st.caption(
    "OpsRAG | Python • LangChain • Gemini • ChromaDB • "
    "LangSmith • MLflow • Streamlit"
)
