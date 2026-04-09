import os
import uuid
import hashlib
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

TRIAL_LIMIT = 25

def get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode="require")
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

def increment_sorts(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET sorts_used = sorts_used + 1 WHERE id = %s", (user_id,))
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