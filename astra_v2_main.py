from textwrap import dedent

astra_phase_4_code = dedent("""
# astra_v2_main.py (Phase 4.0 – Seamless Conversational Voice AI with Streaming, Fallback, and Playbook Loading)

from flask import Flask, request, jsonify
import openai, os, json, redis, time
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX"))
namespace = os.environ.get("PINECONE_NAMESPACE")

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    password=os.environ.get("REDIS_PASSWORD", None),
    db=0,
    decode_responses=True,
    ssl=True
)

app = Flask(__name__)
HISTORY_KEY_PREFIX = "astra-session:"

# On startup: upload batch2.json to Pinecone
try:
    with open("batch2.json", "r") as f:
        qa_data = json.load(f)
    for i, item in enumerate(qa_data):
        question, answer = item["question"], item["answer"]
        embed = openai.embeddings.create(input=[question], model="text-embedding-3-large")
        vector = embed.data[0].embedding
        index.upsert(
            vectors=[{
                "id": f"astra-batch2-{i}",
                "values": vector,
                "metadata": {
                    "text": answer,
                    "source": "brain",
                    "topic": item.get("category", "general")
                }
            }],
            namespace=namespace
        )
    print("✅ batch2.json uploaded to Pinecone")
except Exception as e:
    print(f"⚠️ Pinecone preload failed: {e}")
""")

astra_phase_4_code[:1000]


