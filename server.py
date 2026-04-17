import os
import json
import anthropic
import stripe
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from models import (init_db, create_user, get_user, get_user_by_email,
                    get_user_by_username, username_exists, get_db,
                    increment_sorts, activate_subscription,
                    deactivate_subscription, set_stripe_customer, hash_password)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Cratify API"})


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    username = data.get("username")

    if not email:
        return jsonify({"error": "email required"}), 400

    existing = get_user_by_email(email)
    if existing:
        return jsonify({"error": "email already registered"}), 409

    if username and username_exists(username):
        return jsonify({"error": "username already taken"}), 409

    user_id = create_user(email=email, username=username, password=password)
    return jsonify({
        "user_id": user_id,
        "username": username,
        "sorts_remaining": 25,
        "subscription_active": False
    })


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    identifier = data.get("email") or data.get("identifier")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"error": "email/username and password required"}), 400

    user = get_user_by_email(identifier)
    if not user:
        user = get_user_by_username(identifier)

    if not user:
        return jsonify({"error": "invalid credentials"}), 401

    if user.get("password_hash") != hash_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])
    return jsonify({
        "user_id": user["id"],
        "username": user.get("username"),
        "email": user["email"],
        "sorts_remaining": sorts_remaining if not user["subscription_active"] else None,
        "subscription_active": bool(user["subscription_active"])
    })


@app.route("/auth/check-username", methods=["GET"])
def check_username():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "username required"}), 400
    taken = username_exists(username)
    return jsonify({"available": not taken, "username": username})


@app.route("/subscription/status", methods=["GET"])
def subscription_status():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    sorts_remaining = None
    if not user["subscription_active"]:
        sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])

    return jsonify({
        "subscription_active": bool(user["subscription_active"]),
        "sorts_used": user["sorts_used"],
        "sorts_remaining": sorts_remaining,
        "trial_limit": user["trial_limit"],
        "username": user.get("username"),
        "email": user.get("email")
    })


@app.route("/classify", methods=["POST"])
def classify():
    data = request.json or {}
    filename = data.get("filename")
    user_id = data.get("user_id")

    if not filename or not user_id:
        return jsonify({"error": "filename and user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    if not user["subscription_active"]:
        if user["sorts_used"] >= user["trial_limit"]:
            return jsonify({"error": "trial_exhausted"}), 402

    prompt = f"""You are a music file classifier for a producer tool called Cratify.

Analyze this filename and return a JSON object with these fields:
- category: Bass, Lead, Pad, Pluck, FX, Drum, Vocal, Chord, Arp, Guitar, Piano, Strings, Brass, Synth, Texture, Ambient, or Other.
- drum_type: ONLY for Drum: Kick, Snare, Hi-Hat, Clap, Perc, Cymbal, Tom, Full Loop. Null for non-drums.
- subcategory: more specific description
- key: musical key if detectable (e.g. "Am", "C#") or null. Always null for drums.
- bpm: BPM if detectable as number or null
- file_type: "stem", "preset", "midi", "sample", or "loop"
- confidence: 0 to 1

CRITICAL CATEGORY RULES:
Never use Loop as a standalone category. Instead classify by the instrument type:
- Drum Loop, Beat, Break → category: Drum (set drum_type: "Full Loop")
- Bass Loop → category: Bass
- Synth Loop, Synth Riff → category: Synth
- Piano Loop → category: Piano
- Guitar Loop → category: Guitar
- Chord Loop, Chord Stab → category: Chord
- Melody Loop, Lead Loop → category: Lead
- Vocal Loop, Vox Loop → category: Vocal
- Arp Loop → category: Arp
- Pad Loop, Atmosphere Loop → category: Pad
- FX Loop, Riser, Sweep → category: FX
- Texture Loop, Ambient Loop → category: Ambient
- If a loop's instrument is truly unknown: category: Other

Set file_type to "loop" whenever the filename contains "loop", "lp", "riff", "break", or "beat".

Filename: {filename}

Return ONLY valid JSON. No markdown, no explanation."""

    try:
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as api_err:
        print(f"[classify] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": str(api_err)}), 500

    try:
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception:
        result = {
            "category": "Other", "drum_type": None, "subcategory": "Unknown",
            "key": None, "bpm": None, "file_type": "stem", "confidence": 0.5
        }

    # Post-process: drums should never have keys
    if result.get("category") == "Drum":
        result["key"] = None
    # Force preset category for preset file extensions
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if ext in ('fxp', 'vital', 'nmsv', 'xpf', 'aupreset', 'patch'):
        result['category'] = 'Preset'
        result['key'] = None
        result['bpm'] = None
        result['file_type'] = 'preset'
    increment_sorts(user_id)
    return jsonify(result)


@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["client_reference_id"]
        customer_id = session["customer"]
        subscription_id = session["subscription"]
        if user_id:
            set_stripe_customer(user_id, customer_id)
            activate_subscription(customer_id, subscription_id)
    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        deactivate_subscription(sub["customer"])

    return jsonify({"status": "ok"})


@app.route("/stripe/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.json or {}
    user_id = data.get("user_id")
    try:
        session = stripe.checkout.Session.create(
            ui_mode="embedded_page",
            line_items=[{"price": os.getenv("STRIPE_PRICE_ID"), "quantity": 1}],
            mode="subscription",
            return_url=f"https://www.cratify.app/dashboard?session_id={{CHECKOUT_SESSION_ID}}",
            client_reference_id=user_id,
        )
        return jsonify({"clientSecret": session.client_secret})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_INTENT_SYSTEM = """You parse music producer chat messages into structured bulk file actions.

Return ONLY a valid JSON object — no explanation, no markdown fences.

Supported actions: "export" (copy files to a destination folder) or "move" (move files).
Return action: null if the message is just a search or question, not a bulk operation.

Output schema:
{
  "action": "export" | "move" | null,
  "filter": {
    "key":      string | null,   // musical key, e.g. "C# minor", "Am", "G major"
    "category": string | null,   // e.g. "loop", "bass", "drum", "vocal", "pad"
    "bpm_min":  number | null,
    "bpm_max":  number | null,
    "file_type": string | null   // extension without dot: "wav", "mp3", "midi"
  },
  "destination": string | null   // absolute path the user mentioned, or null
}

Decision rules:
- "export / send / copy … to …"  → action: "export"
- "move … to …"                  → action: "move"
- "show / find / search / what"  → action: null
- Vague questions with no clear destination → action: null
- If destination folder is not explicitly stated → destination: null

Examples:
  "export all my C# minor loops to /Users/zee/Desktop"
  → {"action":"export","filter":{"key":"C# minor","category":"loop","bpm_min":null,"bpm_max":null,"file_type":null},"destination":"/Users/zee/Desktop"}

  "move all bass wav files to /Users/zee/Music/Project"
  → {"action":"move","filter":{"key":null,"category":"bass","bpm_min":null,"bpm_max":null,"file_type":"wav"},"destination":"/Users/zee/Music/Project"}

  "send everything between 120 and 130 bpm to my desktop"
  → {"action":"export","filter":{"key":null,"category":null,"bpm_min":120,"bpm_max":130,"file_type":null},"destination":"/Users/zee/Desktop"}

  "show me all loops in Am"
  → {"action":null,"filter":{},"destination":null}

  "find dark pads in C minor"
  → {"action":null,"filter":{},"destination":null}
"""


@app.route("/intent", methods=["POST"])
def intent():
    """Parse a chat message for bulk file action intent using Claude Haiku."""
    data = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"action": None})
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": message}]
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if model wraps anyway
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rsplit("```", 1)[0]
        result = json.loads(raw.strip())
        # Normalise: always return the top-level action key
        if "action" not in result:
            result["action"] = None
        return jsonify(result)
    except Exception as e:
        print(f"[/intent] error: {e}", flush=True)
        return jsonify({"action": None})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    system = data.get("system", "")
    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system=system,
            messages=messages
        )
        reply = message.content[0].text.strip()

        # Extract filenames from reply - multiple patterns to catch all formats
        import re
        _EXT = r'(?:wav|mp3|aiff|aif|flac|mid|midi|fxp|vital|serum)'

        # Pattern 1: bare filenames anywhere in text (word chars + common punctuation + extension)
        bare = re.findall(
            rf'[\w&#+\-\.\(\)\[\]\s]+\.{_EXT}',
            reply, re.IGNORECASE)

        # Pattern 2: numbered list lines  e.g. "1. filename.wav" or "1) filename.wav"
        numbered = re.findall(
            rf'^\s*\d+[.\)]\s+([\w&#+\-\.\(\)\[\] ]+\.{_EXT})',
            reply, re.IGNORECASE | re.MULTILINE)

        # Pattern 3: dashed list lines  e.g. "- filename.wav" or "• filename.wav"
        dashed = re.findall(
            rf'^\s*[-•]\s+([\w&#+\-\.\(\)\[\] ]+\.{_EXT})',
            reply, re.IGNORECASE | re.MULTILINE)

        # Pattern 4: quoted filenames  e.g. "filename.wav" or 'filename.wav'
        quoted = re.findall(
            rf'["\']+([\w&#+\-\.\(\)\[\] ]+\.{_EXT})["\']',
            reply, re.IGNORECASE)

        # Pattern 5: FILE: lines (may have markdown asterisks around FILE)
        file_lines = re.findall(r'\*{0,2}FILE:\*{0,2}\s*([^\n|]+)', reply, re.IGNORECASE)

        # Merge all hits, deduplicate preserving order
        seen = set()
        found_files = []
        for f in bare + numbered + dashed + quoted + file_lines:
            f = f.strip()
            if f and f not in seen:
                seen.add(f)
                found_files.append(f)

        return jsonify({"reply": reply, "found_files": found_files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_PAIR_MAP = {
    "kick":      ["snare", "clap", "hi-hat", "hihat"],
    "bass":      ["pad", "lead", "pluck"],
    "lead":      ["pad", "arp"],
    "loop":      ["drum loop", "bass"],
}

@app.route("/pair", methods=["POST"])
def pair():
    """Given a filepath + category, return top 5 complementary files from target categories."""
    data = request.json or {}
    filepath = data.get("filepath", "").strip()
    category = (data.get("category") or "").strip().lower()

    if not filepath or not category:
        return jsonify({"error": "filepath and category required"}), 400

    # Determine target categories from pairing map
    target_cats = None
    for key, targets in _PAIR_MAP.items():
        if key in category:
            target_cats = targets
            break
    if not target_cats:
        return jsonify({"error": f"no pairing defined for category '{category}'"}), 400

    try:
        import sqlite3 as _sqlite3
        import numpy as _np
        import base64 as _b64
        from pathlib import Path as _Path
        from scipy.spatial.distance import cosine as _cosine

        db_path = str(_Path.home() / ".cratify" / "index.db")

        # Load embedding for the query file
        conn = _sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT embedding FROM files WHERE filepath = ?", (filepath,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0]:
            return jsonify({"error": "no embedding found for this file — run indexer first"}), 404

        query_emb = _np.frombuffer(row[0], dtype=_np.float32)

        # Build SQL LIKE clause for target categories (case-insensitive)
        placeholders = " OR ".join(["LOWER(category) LIKE ?" for _ in target_cats])

        conn2 = _sqlite3.connect(db_path)
        c2 = conn2.cursor()
        c2.execute(
            f"SELECT filepath, filename, category, key, bpm, embedding FROM files "
            f"WHERE embedding IS NOT NULL AND filepath != ? AND ({placeholders})",
            [filepath] + [f"%{t}%" for t in target_cats],
        )
        candidates = c2.fetchall()
        conn2.close()

        results = []
        for fp, fn, cat, key, bpm, emb_blob in candidates:
            try:
                emb = _np.frombuffer(emb_blob, dtype=_np.float32)
                sim = float(1.0 - _cosine(query_emb, emb))
                results.append({
                    "filepath": fp,
                    "filename": fn,
                    "category": cat or "",
                    "key": key or "",
                    "bpm": bpm,
                    "similarity": round(sim, 4),
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return jsonify({"pairs": results[:5], "target_categories": target_cats})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/embed", methods=["POST"])
def embed():
    """Generate a 52-dim audio embedding (40 MFCC + 12 chroma) for a local file path."""
    data = request.json or {}
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "filepath required"}), 400

    import os as _os
    if not _os.path.isfile(filepath):
        return jsonify({"error": "file not found"}), 404

    try:
        import librosa
        import numpy as np
        import base64

        y, sr = librosa.load(filepath, sr=22050, mono=True, duration=30)
        if len(y) < 1024:
            return jsonify({"error": "audio too short"}), 400

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)  # shape (40,)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # shape (12,)

        embedding = np.concatenate([mfcc_mean, chroma_mean]).astype(np.float32)  # shape (52,)
        encoded = base64.b64encode(embedding.tobytes()).decode("ascii")
        return jsonify({"embedding": encoded, "dims": len(embedding)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import sys
    print("Starting Cratify API...", flush=True)
    try:
        init_db()
        print("Database initialized", flush=True)
    except Exception as e:
        print(f"Database error: {e}", flush=True)
        sys.exit(1)
    port = int(os.getenv("PORT", 5000))
    print(f"Running on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
