from backend.rag import create_vector_store, ask_question

vector_store = create_vector_store()

question = "When must employees notify their manager about planned leave?"

try:
    result = ask_question(vector_store, question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    print(result["sources"])

except Exception as error:
    print("\nGENERATION FAILED")
    print(type(error).__name__)
    print(error)