# astra_v2_main.py
# Phase 1–5: Memory + Tone + SSML + Self-Correction + Vapi Phone Integration

from flask import Flask, request, jsonify
import openai
import os
import json
import redis
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
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
    key = HISTORY_KEY_PREFIX + session_id
    return json.loads(redis_client.get(key) or "[]")


def save_history(session_id, history):
    key = HISTORY_KEY_PREFIX + session_id
    redis_client.set(key, json.dumps(history), ex=3600)


def detect_tone(user_input):
    prompt = f"Classify the tone of this user message into one of the following: 'technical', 'casual', 'formal', 'urgent', 'emotional', or 'neutral'.\\nMessage: {user_input}"
    try:
        result = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an NLP assistant for tone detection."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )
        return result.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Tone detection error: {e}")
        return "neutral"


def generate_system_prompt(tone):
    base = "You are Astra, the Executive AI of DiviScanOS."
    style = {
        "technical": " Use concise, high-precision language. Be data-driven and explicit.",
        "casual": " Keep it light and friendly. Use informal language where appropriate.",
        "formal": " Speak professionally and respectfully. Use polished and articulate language.",
        "urgent": " Prioritize clarity and actionability. Respond directly and assertively.",
        "emotional": " Be empathetic and reassuring. Use supportive language.",
        "neutral": " Provide clear and informative responses."
    }
    return base + style.get(tone, style["neutral"])


def format_ssml(text):
    lines = text.split(". ")
    tagged = [f"<s>{line.strip()}.</s>" for line in lines if line.strip()]
    ssml = "<speak>\\n<prosody rate='medium'>\\n" + "\\n<break time='500ms'/>\\n".join(tagged) + "\\n</prosody>\\n</speak>"
    return ssml


def is_vague(text):
    text = text.lower()
    vague_phrases = [
        "i'm not sure", "as an ai", "i don't know", "can't help with that", "uncertain", "unclear",
        "my training data", "no definitive answer", "beyond my knowledge"
    ]
    return any(phrase in text for phrase in vague_phrases)


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

    messages = [{"role": "system", "content": system_prompt}]
    messages += history[-6:]
    messages.append({"role": "user", "content": question})

    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.6,
            max_tokens=1000
        )
        reply_text = completion.choices[0].message.content.strip()

        if is_vague(reply_text):
            clarification_prompt = f"The previous answer was too vague. Please restate with more precision, examples, or technical clarity.\\nUser question: {question}"
            messages.append({"role": "assistant", "content": reply_text})
            messages.append({"role": "user", "content": clarification_prompt})
            completion_retry = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.6,
                max_tokens=1000
            )
            reply_text = completion_retry.choices[0].message.content.strip()

        reply_ssml = format_ssml(reply_text) if for_voice else None

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply_text})
        save_history(session_id, history)

        return jsonify({
            "response": reply_text,
            "ssml": reply_ssml,
            "tone": tone,
            "voice_ready": for_voice
        })

    except Exception as e:
        print(f"Error in GPT: {e}")
        return jsonify({"response": "Astra encountered an issue. Please try again."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



