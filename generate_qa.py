import os
import json
from openai import AzureOpenAI

# Load slm_qa/.env into the environment if that file exists (python-dotenv).
# Optional: if the package is missing, OS environment variables still work.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration. The API key is read from the AZURE_OPENAI_KEY environment
# variable - set it as a system variable OR put it in a gitignored .env file.
# Never hardcode secrets in source files.
endpoint = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    "https://muham-mp9jnggo-eastus2.cognitiveservices.azure.com/",
)
deployment = "gpt-5-mini"
api_version = "2024-12-01-preview"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
if not subscription_key:
    raise SystemExit(
        "AZURE_OPENAI_KEY environment variable is not set.\n"
        "Set it first, e.g. (PowerShell):  $env:AZURE_OPENAI_KEY = \"<your-key>\"\n"
        "or permanently via System Environment Variables."
    )

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

print("Generating synthetic Pakistani teaching dataset using Azure OpenAI (gpt-5-mini)...")

prompt = """Generate 50 highly realistic, high-quality Q&A pairs for an AI Chatbot that is designed to help teachers in Pakistan. 
The topics should cover:
- Classroom management for large class sizes (50+ students)
- Dealing with the Single National Curriculum (SNC)
- FBISE, BISE, matric, and intermediate exam patterns
- Translanguaging (helping students transition from Urdu/Punjabi to English medium)
- Interacting with parents during PTMs in Pakistan
- Corporal punishment alternatives
- Load shedding and lack of resources in public schools

Output EXACTLY in JSONL format. Every single line must be a completely valid JSON object.
Format example:
{"text": "User: How do I handle a student who talks too much in class?\nBot: You should try to give them leadership roles or seat them near the front. [SEP]"}

Do not include markdown blocks like ```jsonl, just raw JSON lines text.
"""

try:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a strict data generation assistant. Output only raw JSONL format."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=16384,
        model=deployment
    )

    content = response.choices[0].message.content
    content = content.replace('```jsonl\n', '').replace('```json\n', '').replace('\n```', '')
    
    count = 0
    with open('data/qa/pk_teaching_qa.jsonl', 'a', encoding='utf-8') as f:
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                f.write(line + '\n')
                count += 1

    print(f"\nSuccessfully appended {count} new high-quality QA pairs to data/qa/pk_teaching_qa.jsonl!")
except Exception as e:
    print(f"Error: {e}")
