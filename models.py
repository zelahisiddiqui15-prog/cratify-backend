import os
import uuid
import hashlib
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

TRIAL_LIMIT = 25

def get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode="prefer")
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT,
            sorts_used INTEGER DEFAULT 0,
            trial_limit INTEGER DEFAULT 25,
            subscription_active INTEGER DEFAULT 0,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT
        )
    """)
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT UNIQUE")
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT")
    # METER1 — per-user, per-month usage: the instrumentation a pricing
    # decision actually needs, and the ledger the ceilings check against.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usage_monthly (
            user_id TEXT NOT NULL,
            month TEXT NOT NULL,
            embed_count INTEGER DEFAULT 0,
            classify_bg_count INTEGER DEFAULT 0,
            classify_int_count INTEGER DEFAULT 0,
            library_size INTEGER,
            search_count INTEGER DEFAULT 0,
            updated_at TEXT,
            PRIMARY KEY (user_id, month)
        )
    """)
    cur.execute("ALTER TABLE usage_monthly ADD COLUMN IF NOT EXISTS search_count INTEGER DEFAULT 0")
    # METER2 — one counter per spending endpoint. No endpoint spends
    # without knowing who asked, and every ask is counted.
    cur.execute("ALTER TABLE usage_monthly ADD COLUMN IF NOT EXISTS describe_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE usage_monthly ADD COLUMN IF NOT EXISTS suggest_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE usage_monthly ADD COLUMN IF NOT EXISTS intent_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE usage_monthly ADD COLUMN IF NOT EXISTS summarize_count INTEGER DEFAULT 0")
    # METER1b -- first-index library size, set once, on users: the growth
    # baseline. usage_monthly.library_size is the per-month time series.
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS first_library_size INTEGER")
    conn.commit()
    cur.close()
    conn.close()

def create_user(email=None, username=None, password=None):
    conn = get_db()
    cur = conn.cursor()
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    password_hash = hash_password(password) if password else None
    cur.execute(
        "INSERT INTO users (id, email, username, password_hash, created_at) VALUES (%s, %s, %s, %s, %s)",
        (user_id, email, username, password_hash, now)
    )
    conn.commit()
    cur.close()
    conn.close()
    return user_id

def get_user(user_id):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return dict(user) if user else None

def get_user_by_email(email):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return dict(user) if user else None

def get_user_by_username(username):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return dict(user) if user else None

def username_exists(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists

def increment_sorts(user_id, count=1):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET sorts_used = sorts_used + %s WHERE id = %s", (count, user_id))
    conn.commit()
    cur.close()
    conn.close()

def month_key():
    return datetime.utcnow().strftime("%Y-%m")


def get_usage(user_id, month=None):
    """Current-month usage row, zeros if none yet."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT * FROM usage_monthly WHERE user_id = %s AND month = %s",
        (user_id, month or month_key()),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else {
        "user_id": user_id, "month": month or month_key(),
        "embed_count": 0, "classify_bg_count": 0,
        "classify_int_count": 0, "library_size": None,
        "search_count": 0, "describe_count": 0, "suggest_count": 0,
        "intent_count": 0, "summarize_count": 0,
    }


def add_usage(user_id, embed=0, classify_bg=0, classify_int=0, library_size=None, search=0,
              describe=0, suggest=0, intent=0, summarize=0):
    """Upsert-increment the month's counters. Instrumentation counts
    EVERYTHING — capped or not, background or interactive — because the
    point of deferring pricing is producing this data."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO usage_monthly (user_id, month, embed_count, classify_bg_count,
                                   classify_int_count, library_size, search_count,
                                   describe_count, suggest_count, intent_count,
                                   summarize_count, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id, month) DO UPDATE SET
            embed_count = usage_monthly.embed_count + EXCLUDED.embed_count,
            classify_bg_count = usage_monthly.classify_bg_count + EXCLUDED.classify_bg_count,
            classify_int_count = usage_monthly.classify_int_count + EXCLUDED.classify_int_count,
            library_size = COALESCE(EXCLUDED.library_size, usage_monthly.library_size),
            search_count = usage_monthly.search_count + EXCLUDED.search_count,
            describe_count = usage_monthly.describe_count + EXCLUDED.describe_count,
            suggest_count = usage_monthly.suggest_count + EXCLUDED.suggest_count,
            intent_count = usage_monthly.intent_count + EXCLUDED.intent_count,
            summarize_count = usage_monthly.summarize_count + EXCLUDED.summarize_count,
            updated_at = EXCLUDED.updated_at
        """,
        (user_id, month_key(), embed, classify_bg, classify_int,
         library_size, search, describe, suggest, intent, summarize,
         datetime.utcnow().isoformat()),
    )
    if library_size is not None:
        cur.execute(
            "UPDATE users SET first_library_size = COALESCE(first_library_size, %s) WHERE id = %s",
            (library_size, user_id),
        )
    conn.commit()
    cur.close()
    conn.close()


def activate_subscription(stripe_customer_id, stripe_subscription_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """UPDATE users SET subscription_active = 1,
           stripe_customer_id = %s, stripe_subscription_id = %s
           WHERE stripe_customer_id = %s""",
        (stripe_customer_id, stripe_subscription_id, stripe_customer_id)
    )
    conn.commit()
    cur.close()
    conn.close()

def deactivate_subscription(stripe_customer_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET subscription_active = 0 WHERE stripe_customer_id = %s", (stripe_customer_id,))
    conn.commit()
    cur.close()
    conn.close()

def set_stripe_customer(user_id, stripe_customer_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET stripe_customer_id = %s WHERE id = %s", (stripe_customer_id, user_id))
    conn.commit()
    cur.close()
    conn.close()