import math
import re
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from backend.rag import ask_question, create_vector_store


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

DATASET_NAME = "OpsRAG Evaluation"

FALLBACK_ANSWER = (
    "I could not find this information in the provided documents."
)

client = Client()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def opsrag_target(inputs: dict) -> dict:

    question = inputs["question"]

    result = ask_question(vector_store, question)

    return {
        "answer": result["answer"],
        "retrieved_context": result["retrieved_context"],
        "sources": result["sources"]
    }


def normalize_text(text: str) -> str:

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text


def semantic_similarity(text_a: str, text_b: str) -> float:

    if not text_a.strip() or not text_b.strip():
        return 0.0

    vector_a = embeddings.embed_query(text_a)
    vector_b = embeddings.embed_query(text_b)

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    return max(0.0, min(1.0, similarity))


def answer_correctness(
    outputs: dict | None,
    reference_outputs: dict
) -> dict:

    outputs = outputs or {}
    reference_outputs = reference_outputs or {}

    answer = normalize_text(
        str(outputs.get("answer") or "")
    )

    expected = normalize_text(
        str(reference_outputs.get("expected_answer") or "")
    )

    score = bool(
        answer
        and expected
        and (
            answer == expected
            or expected in answer
        )
    )

    return {
        "key": "correctness",
        "score": float(score)
    }


def answer_relevance(
    outputs: dict | None,
    inputs: dict,
    reference_outputs: dict
) -> dict:

    outputs = outputs or {}
    inputs = inputs or {}
    reference_outputs = reference_outputs or {}

    answer = str(outputs.get("answer") or "")
    question = str(inputs.get("question") or "")
    expected = str(
        reference_outputs.get("expected_answer") or ""
    )

    if not answer.strip():
        score = 0.0

    elif normalize_text(expected) == normalize_text(FALLBACK_ANSWER):

        score = float(
            normalize_text(answer) == normalize_text(FALLBACK_ANSWER)
        )

    else:

        score = semantic_similarity(question, answer)

    return {
        "key": "answer_relevance",
        "score": score
    }



def answer_faithfulness(
    outputs: dict | None
) -> dict:

    outputs = outputs or {}

    answer = str(outputs.get("answer") or "")
    context = str(outputs.get("retrieved_context") or "")

    score = semantic_similarity(answer, context)

    return {
        "key": "faithfulness_similarity_proxy",
        "score": score
    }



def retrieval_quality(
    outputs: dict | None,
    reference_outputs: dict
) -> dict:

    outputs = outputs or {}
    reference_outputs = reference_outputs or {}

    expected = str(
        reference_outputs.get("expected_answer") or ""
    )

    context = str(
        outputs.get("retrieved_context") or ""
    )

    if normalize_text(expected) == normalize_text(FALLBACK_ANSWER):
        score = 0.0
    else:
        score = semantic_similarity(expected, context)

    return {
        "key": "retrieval_similarity_proxy",
        "score": score
    }


if __name__ == "__main__":

    vector_store = create_vector_store()

    print("\nStarting OpsRAG evaluation...\n")

    experiment_results = evaluate(
        opsrag_target,
        data=DATASET_NAME,
        evaluators=[
            answer_correctness,
            answer_relevance,
            answer_faithfulness,
            retrieval_quality
        ],
        experiment_prefix="OpsRAG-Improved-Evaluation",
        client=client
    )

    print("\nEvaluation completed!")
    print("Four evaluation metrics have been executed.")
    print("Inspect the experiment in LangSmith.")