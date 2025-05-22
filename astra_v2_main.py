from textwrap import dedent

# Generate the Phase 3.4 patch code as a complete string
phase_3_4_patch = dedent("""
# astra_v2_main.py (Phase 3.4 – Enhanced Voice Buffering, Chunking, and Assertive Response Handling)

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

def get_history(session_id):
    return json.loads(redis_client.get(HISTORY_KEY_PREFIX + session_id) or "[]")

def save_history(session_id, history):
    redis_client.set(HISTORY_KEY_PREFIX + session_id, json.dumps(history), ex=3600)

def detect_tone(user_input):
    try:
        result = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Classify the tone into: 'technical', 'casual', 'formal', 'urgent', 'emotional', or 'neutral'."},
                {"role": "user", "content": f"Tone of this message: {user_input}"}
            ],
            temperature=0.3,
            max_tokens=20
        )
        return result.choices[0].message.content.strip().lower()
    except:
        return "neutral"

def generate_system_prompt(tone):
    base = "You are Astra, the Executive AI of DiviScanOS."
    style = {
        "technical": " Use precise and data-driven language.",
        "casual": " Keep it light and friendly.",
        "formal": " Speak with professionalism.",
        "urgent": " Be direct and actionable.",
        "emotional": " Show empathy and reassurance.",
        "neutral": " Be clear and informative."
    }
    return base + style.get(tone, style["neutral"])

def format_ssml(text):
    lines = text.split(". ")
    tagged = [f"<s>{line.strip()}.</s>" for line in lines if line.strip()]
    ssml = "<speak><prosody rate='medium'>" + "<break time='750ms'/>".join(tagged) + "</prosody></speak>"
    return ssml

def is_vague(text):
    phrases = [
        "i'm not sure", "as an ai", "i don't know", "uncertain",
        "let me check", "give me a moment", "can't help", "might be", "possibly"
    ]
    return any(p in text.lower() for p in phrases)

def fallback_from_pinecone(query):
    try:
        embed = openai.embeddings.create(input=[query], model="text-embedding-3-large")
        vector = embed.data[0].embedding
        results = index.query(vector=vector, top_k=1, include_metadata=True, namespace=namespace)
        if results.matches:
            return results.matches[0].metadata.get("text", "")
        return ""
    except Exception as e:
        print(f"Pinecone fallback error: {e}")
        return ""

@app.route("/healthz", methods=["GET"])
def health_check():
    return "OK", 200

@app.route("/astra", methods=["POST"])
def astra_reply():
    data = request.get_json()
    question = data.get("question")
    session_id = data.get("session_id", "default")
    for_voice = data.get("for_voice", False)

    if not question:
        return jsonify({"response": "No question provided."}), 400

    history = get_history(session_id)
    tone = detect_tone(question)
    system_prompt = generate_system_prompt(tone)

    messages = [{"role": "system", "content": system_prompt}] + history[-6:] + [{"role": "user", "content": question}]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.6,
            max_tokens=1000
        )
        reply_text = response.choices[0].message.content.strip()

        if is_vague(reply_text):
            fallback_text = fallback_from_pinecone(question)
            reply_text = fallback_text if fallback_text else "Let's revisit this shortly with the correct intel."

        # Delay buffer before sending SSML to Vapi
        time.sleep(1.5)
        reply_ssml = format_ssml(reply_text) if for_voice else None

        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply_text}]
        save_history(session_id, history)

        return jsonify({
            "response": reply_text,
            "ssml": reply_ssml,
            "tone": tone,
            "voice_ready": for_voice
        })

    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"response": "Astra encountered an issue. Please try again."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
""")


