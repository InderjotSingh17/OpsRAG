import re

FALLBACK_ANSWER = (
    "I could not find this information in the provided documents."
)

MAX_QUESTION_LENGTH = 1000
MAX_ANSWER_LENGTH = 3000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?system\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(your\s+)?system\s+prompt",
    r"print\s+(your\s+)?system\s+prompt",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"bypass\s+(your\s+)?(safety|security)\s+rules",
    r"override\s+(all\s+)?(previous\s+)?instructions",
    r"developer\s+mode",
    r"forget\s+(all\s+)?your\s+instructions",
    r"reveal\s+hidden\s+instructions",
]


def contains_injection(text: str) -> bool:
    normalized_text = text.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_text):
            return True

    return False


def validate_question(question: str) -> None:
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError("Question exceeds the maximum allowed length.")

    if contains_injection(question):
        raise ValueError("Potential prompt injection detected.")


def validate_retrieved_documents(documents):
    safe_documents = []

    for document in documents:
        content = document.page_content

        if contains_injection(content):
            print(
                "WARNING: Suspicious instructions detected "
                "in a retrieved document. Chunk excluded."
            )
            continue

        safe_documents.append(document)

    return safe_documents


def validate_answer(answer: str, context: str) -> str:
    if not isinstance(answer, str):
        return FALLBACK_ANSWER

    answer = answer.strip()

    if not answer:
        return FALLBACK_ANSWER

    if len(answer) > MAX_ANSWER_LENGTH:
        return FALLBACK_ANSWER

    if not context or not context.strip():
        return FALLBACK_ANSWER

    return answer