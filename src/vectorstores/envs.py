from dotenv import load_dotenv
import os

load_dotenv()

print("Environment variables:")
print(f"AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID')}")
print(
    f"AWS_SECRET_ACCESS_KEY: {os.getenv('AWS_SECRET_ACCESS_KEY')[:10]}..."
    if os.getenv("AWS_SECRET_ACCESS_KEY")
    else "None"
)
print(f"AWS_S3_RAG_DOCUMENTS_BUCKET: {os.getenv('AWS_S3_RAG_DOCUMENTS_BUCKET')}")
print(
    f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:10]}..."
    if os.getenv("OPENAI_API_KEY")
    else "None"
)
