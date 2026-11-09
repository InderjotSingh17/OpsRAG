from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

client = Client()

prompts = list(client.list_prompts(limit=5))

print("Prompt access: SUCCESS")
print("Number of results:", len(prompts))

for prompt in prompts:
    print("Prompt result:", repr(prompt))