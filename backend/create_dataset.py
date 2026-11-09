import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

client = Client()

dataset_path = ROOT_DIR / "evaluation" / "test_cases.json"

with open(dataset_path, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

dataset = client.create_dataset(
    dataset_name="OpsRAG Evaluation",
    description="Evaluation dataset for the OpsRAG RAG pipeline."
)

inputs = [
    {"question": case["question"]}
    for case in test_cases
]

outputs = [
    {"expected_answer": case["expected_answer"]}
    for case in test_cases
]

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset.id
)

print("Evaluation dataset created successfully!")
print(f"Dataset: {dataset.name}")
print(f"Total test cases uploaded: {len(test_cases)}")