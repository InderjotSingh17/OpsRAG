
import json
import time
import uuid
from pathlib import Path
from backend.cost_tracker import get_cost_summary
import mlflow

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from backend.rag import load_documents, ask_question

from backend.evaluate import (
    answer_correctness,
    answer_relevance,
    answer_faithfulness,
    retrieval_quality,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "evaluation" / "test_cases.json"
TRACKING_DB = BASE_DIR / "mlflow.db"

EXPERIMENT_NAME = "OpsRAG-Experiments"

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CONFIGURATIONS = [
    {
        "name": "Baseline-RAG-v2",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "retrieval_k": 3,
    },
    {
        "name": "Experiment-B-v2",
        "chunk_size": 800,
        "chunk_overlap": 100,
        "retrieval_k": 5,
    },
]


with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

if not test_cases:
    raise ValueError("Evaluation dataset is empty.")



documents = load_documents()

if not documents:
    raise ValueError("No readable documents found.")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

def create_experiment_vector_store(config):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No document chunks were generated.")

    collection_name = (
        f"opsrag_{uuid.uuid4().hex[:12]}"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )

    print(f"\nConfiguration: {config['name']}")
    print(f"Documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Retrieval K: {config['retrieval_k']}")

    return vector_store, len(chunks)


def run_evaluation(vector_store, config):

    results = []

    metric_values = {
        "correctness": [],
        "answer_relevance": [],
        "faithfulness_similarity_proxy": [],
        "retrieval_similarity_proxy": [],
    }

    total_latency = 0.0

    print(f"\nStarting {config['name']} evaluation...\n")

    for index, case in enumerate(test_cases, start=1):

        question = case["question"]

        reference_outputs = {
            "expected_answer": case["expected_answer"]
        }

        print(f"Test {index}/{len(test_cases)}")
        print(f"Question: {question}")

        start_time = time.perf_counter()

        outputs = ask_question(
            vector_store,
            question,
            retrieval_k=config["retrieval_k"]
        )

        latency = time.perf_counter() - start_time
        total_latency += latency

        inputs = {
            "question": question
        }

        scores = [
            answer_correctness(
                outputs,
                reference_outputs
            ),
            answer_relevance(
                outputs,
                inputs,
                reference_outputs
            ),
            answer_faithfulness(outputs),
            retrieval_quality(
                outputs,
                reference_outputs
            ),
        ]

        for score in scores:
            metric_values[score["key"]].append(
                score["score"]
            )

        result = {
            "question": question,
            "answer": outputs["answer"],
            "expected_answer": case["expected_answer"],
            "retrieved_context": outputs["retrieved_context"],
            "sources": outputs["sources"],
            "latency_seconds": round(latency, 4),
            "scores": {
                score["key"]: score["score"]
                for score in scores
            }
        }

        results.append(result)

        print(f"Latency: {latency:.2f}s")
        print(f"Correctness: {scores[0]['score']:.2f}\n")

    average_metrics = {
        f"avg_{name}": sum(values) / len(values)
        for name, values in metric_values.items()
        if values
    }

    average_metrics["avg_latency_seconds"] = (
        total_latency / len(results)
    )

    average_metrics["total_latency_seconds"] = total_latency
    average_metrics["completed_test_cases"] = len(results)

    return results, average_metrics


def calculate_run_cost(before, after):

    total_requests = (
        after["total_requests"] - before["total_requests"]
    )

    input_tokens = (
        after["total_input_tokens"] - before["total_input_tokens"]
    )

    output_tokens = (
        after["total_output_tokens"] - before["total_output_tokens"]
    )

    total_tokens = input_tokens + output_tokens

    estimated_cost = (
        after["total_cost"] - before["total_cost"]
    )

    cost_per_request = (
        estimated_cost / total_requests
        if total_requests > 0 else 0.0
    )

    return {
        "total_requests": total_requests,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost,
        "estimated_cost_per_request_usd": cost_per_request,
    }


def log_experiment(config, results, metrics, chunk_count):

    with mlflow.start_run(
        run_name=config["name"]
    ) as run:

        mlflow.log_params({
            "experiment_type": "controlled_rag_comparison",
            "model": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "corpus_documents": len(documents),
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "retrieval_k": config["retrieval_k"],
            "dataset_size": len(test_cases),
            "indexed_chunks": chunk_count,
        })

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            results,
            "evaluation_results.json"
        )

        mlflow.log_dict(
            metrics,
            "aggregate_metrics.json"
        )

        print("\nMLflow run recorded.")
        print(f"Run ID: {run.info.run_id}")

        return run.info.run_id


def main():

    mlflow.set_tracking_uri(
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = []

    for config in CONFIGURATIONS:

        vector_store, chunk_count = (
            create_experiment_vector_store(config)
        )

        cost_before = get_cost_summary()

        results, metrics = run_evaluation(
            vector_store,
            config
        )

        cost_after = get_cost_summary()

        cost_metrics = calculate_run_cost(
            cost_before,
            cost_after
        )

        metrics.update(cost_metrics)

        print("\nExperiment Cost Summary:")
        for name, value in cost_metrics.items():
            print(f"{name}: {value}")

        run_id = log_experiment(
            config,
            results,
            metrics,
            chunk_count
        )

        all_results.append({
            "configuration": config["name"],
            "run_id": run_id,
            "chunk_count": chunk_count,
            "metrics": metrics,
        })

        del vector_store


    comparison_path = (
        BASE_DIR / "experiments" / "controlled_comparison.json"
    )

    comparison_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_results,
            file,
            indent=4
        )

    print("\n========================================")
    print("CONTROLLED COMPARISON COMPLETED")
    print("========================================")

    for result in all_results:

        print(f"\n{result['configuration']}")
        print(f"Chunks: {result['chunk_count']}")

        for name, value in result["metrics"].items():
            print(f"{name}: {value:.4f}")

        print(f"Run ID: {result['run_id']}")

    print("\nComparison saved to:")
    print(comparison_path)


if __name__ == "__main__":
    main()
