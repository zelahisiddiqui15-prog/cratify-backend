import os
import anthropic
import stripe
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from models import init_db, create_user, get_user, get_db, increment_sorts, activate_subscription, deactivate_subscription, set_stripe_customer

load_dotenv()

app = Flask(__name__)
CORS(app)

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "SortDrop API"})


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")

    if email:
        conn = get_db()
        try:
            existing = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
            conn.close()
            if existing:
                user = dict(existing)
                sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])
                return jsonify({
                    "user_id": user["id"],
                    "sorts_remaining": sorts_remaining,
                    "subscription_active": bool(user["subscription_active"])
                })
        except Exception:
            conn.close()

    user_id = create_user(email=email)
    return jsonify({
        "user_id": user_id,
        "sorts_remaining": 25,
        "subscription_active": False
    })


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
        "trial_limit": user["trial_limit"]
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

    prompt = f"""You are a music file classifier for a producer tool called SortDrop.

Analyze this filename and return a JSON object with these fields:
- category: the main sound category (Bass, Lead, Pad, Pluck, FX, Drum, Vocal, Chord, Arp, Texture, Ambient, or Other)
- subcategory: a more specific description (e.g. "Midrange Bass", "Supersaw Lead")
- key: musical key if detectable from filename (e.g. "Am", "C#", "Fm") or null
- bpm: BPM if detectable from filename as a number or null
- file_type: one of "stem", "preset", "midi", "sample", "loop"
- confidence: your confidence score from 0 to 1

Filename: {filename}

Return ONLY valid JSON, no explanation."""

    message = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    try:
        result = json.loads(message.content[0].text)
    except Exception:
        result = {
            "category": "Other",
            "subcategory": "Unknown",
            "key": None,
            "bpm": None,
            "file_type": "stem",
            "confidence": 0.5
        }

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

    if event["type"] == "customer.subscription.created":
        sub = event["data"]["object"]
        activate_subscription(sub["customer"], sub["id"])

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        deactivate_subscription(sub["customer"])

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)