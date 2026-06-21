from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

dataset = client.create_dataset("chat bot evaluation v66")

client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {"question": "Hi"},
            "outputs": {"answer": "Hello! How can I help you?"}
        },
        {
            "inputs": {"question": "What is 2+2?"},
            "outputs": {"answer": "4"}
        },
        {
            "inputs": {"question": "Who created Python?"},
            "outputs": {"answer": "Guido van Rossum"}
        }
    ]
)

print("Dataset and examples created successfully!")