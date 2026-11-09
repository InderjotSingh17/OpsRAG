from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

client = Client()

try:
    prompt = client.pull_prompt("opsrag-system")
    print("Prompt retrieved successfully!")
    print(prompt)

except Exception as e:
    print("Prompt retrieval failed:", e)