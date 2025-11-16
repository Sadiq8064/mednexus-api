"""
Developer API Gateway - MedNexus
All endpoints are GET only.
Normal user system removed.
Developer features:
- Register / Login (GET)
- Create / Revoke / Regenerate API key (GET)
- /dev/ask returns:
    {
      "answer": Gemini answer,
      "context": RAG answer,
      "source": RAG source,
      "cache_hit": bool,
      "response_time_ms": int,
      "calls_today": int,
      "remaining_today": int,
      "reset_at_ist": "<ISO>"
    }
- Daily limit = 50 calls, reset midnight IST
- Brevo warning when remaining == 5
"""

import os
import json
import random
import string
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import requests
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import EmailStr
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from upstash_redis.asyncio import Redis
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Brevo SDK
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

# ---------------------------------------------------------
# ENV + CONFIG
# ---------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL")
REDIS_TOKEN = os.getenv("REDIS_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "dev_api_db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
EXTERNAL_RAG_URL = os.getenv("EXTERNAL_RAG_URL", "https://mednexusrag-production.up.railway.app/ask")
DAILY_LIMIT_DEFAULT = int(os.getenv("DAILY_API_LIMIT", "50"))

missing = [k for k, v in {
    "REDIS_URL": REDIS_URL,
    "REDIS_TOKEN": REDIS_TOKEN,
    "MONGO_URI": MONGO_URI,
    "GEMINI_API_KEY": GEMINI_API_KEY
}.items() if v is None]

if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------
# Init Services
# ---------------------------------------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

dev_users_col = db["dev_users"]
api_keys_col = db["api_keys"]
api_usage_col = db["api_usage_logs"]
global_qa_col = db["global_qa"]

dev_users_col.create_index("email", unique=True)
api_keys_col.create_index("api_key", unique=True)
api_usage_col.create_index("api_key")
global_qa_col.create_index("normalized_question", unique=True)

# Redis
try:
    redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
    REDIS_AVAILABLE = True
except:
    redis = None
    REDIS_AVAILABLE = False

# Embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Brevo
BREVO_AVAILABLE = False
brevo_client = None
if BREVO_API_KEY:
    cfg = sib_api_v3_sdk.Configuration()
    cfg.api_key["api-key"] = BREVO_API_KEY
    brevo_client = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(cfg))
    BREVO_AVAILABLE = True

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def now_utc(): return datetime.utcnow().replace(tzinfo=timezone.utc)
def now_ist(): return datetime.now(tz=IST)

def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = "".join(c for c in t if c.isalnum() or c.isspace())
    return " ".join(t.split())

def generate_api_key(length=17):
    chars = string.ascii_letters + string.digits
    return "".join(random.SystemRandom().choice(chars) for _ in range(length))

def cosine_similarity(a, b):
    a = np.array(a); b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

async def redis_incr_daily(api_key, limit):
    today = now_ist().strftime("%Y-%m-%d")
    redis_key = f"rate:{api_key}:{today}"

    midnight = datetime.combine(
        now_ist().date() + timedelta(days=1),
        datetime.min.time()
    ).astimezone(IST)

    ttl = int((midnight - now_ist()).total_seconds())

    if not REDIS_AVAILABLE:
        api_keys_col.update_one(
            {"api_key": api_key},
            {"$inc": {f"daily.{today}.count": 1, "total_calls": 1}},
            upsert=True
        )
        doc = api_keys_col.find_one({"api_key": api_key})
        count = doc.get("daily", {}).get(today, {}).get("count", 0)
        rem = max(0, limit - count)
        return {"allowed": count <= limit, "calls_today": count, "remaining": rem, "reset_at": midnight.isoformat()}

    try:
        calls = await redis.incr(redis_key)
        if calls == 1:
            await redis.expire(redis_key, ttl)
    except:
        api_keys_col.update_one(
            {"api_key": api_key},
            {"$inc": {f"daily.{today}.count": 1, "total_calls": 1}},
            upsert=True
        )
        doc = api_keys_col.find_one({"api_key": api_key})
        calls = doc.get("daily", {}).get(today, {}).get("count", 1)

    rem = max(0, limit - calls)
    return {"allowed": calls <= limit, "calls_today": calls, "remaining": rem, "reset_at": midnight.isoformat()}

async def find_cached_answer(question, thresh=0.85):
    norm = normalize_text(question)
    exact = global_qa_col.find_one({"normalized_question": norm})
    if exact:
        return {
            "answer": exact["answer"],
            "context": exact.get("context"),
            "source": exact.get("source"),
            "similarity": 1.0,
            "type": "exact"
        }

    docs = list(global_qa_col.find().limit(500))
    emb_user = embedding_model.encode([question])[0].tolist()

    best = None
    best_sim = 0.0

    for d in docs:
        emb = d.get("embedding")
        if not emb: continue
        sim = cosine_similarity(emb_user, emb)
        if sim > best_sim:
            best_sim = sim
            best = d

    if best and best_sim >= thresh:
        return {
            "answer": best["answer"],
            "context": best.get("context"),
            "source": best.get("source"),
            "similarity": best_sim,
            "type": "semantic"
        }

    return None

async def should_send_notification_today(api_key: str) -> bool:
    today = now_ist().strftime("%Y-%m-%d")

    if REDIS_AVAILABLE:
        key = f"notify:{api_key}:{today}"
        try:
            if await redis.get(key): return False
            midnight = datetime.combine(now_ist().date() + timedelta(days=1), datetime.min.time()).astimezone(IST)
            ttl = int((midnight - now_ist()).total_seconds())
            await redis.setex(key, ttl, "1")
            return True
        except:
            pass

    doc = api_keys_col.find_one({"api_key": api_key})
    last = doc.get("last_warning_date") if doc else None
    if last == today: return False

    api_keys_col.update_one({"api_key": api_key}, {"$set": {"last_warning_date": today}}, upsert=True)
    return True

def send_brevo(owner_email, api_key, used, remaining, limit, reset_at):
    if not BREVO_AVAILABLE: return

    subject = "MedNexus API — Only 5 calls remaining"
    html = f"""
    <div>
        <p>Your API key <b>{api_key}</b> is nearing its daily limit.</p>
        <ul>
           <li>Used today: {used}</li>
           <li>Remaining: {remaining}</li>
           <li>Daily limit: {limit}</li>
           <li>Resets at IST: {reset_at}</li>
        </ul>
    </div>
    """
    email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": owner_email}],
        sender={"name": "MedNexus", "email": "mrsadiq471@gmail.com"},
        subject=subject,
        html_content=html
    )
    try:
        brevo_client.send_transac_email(email)
    except:
        pass

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(title="Developer API Gateway - MedNexus", version="2.0")

# ---------------------------------------------------------
# Developer Register/Login
# ---------------------------------------------------------
@app.get("/dev/register")
def dev_register(email: EmailStr = Query(...), password: str = Query(...)):
    try:
        dev_users_col.insert_one({
            "email": email,
            "password": password,
            "created_at": now_utc()
        })
        return {"status": "success", "email": email}
    except DuplicateKeyError:
        raise HTTPException(400, "User already exists")

@app.get("/dev/login")
def dev_login(email: EmailStr = Query(...), password: str = Query(...)):
    user = dev_users_col.find_one({"email": email})
    if not user:
        raise HTTPException(404, "User not found")
    if user["password"] != password:
        raise HTTPException(401, "Invalid password")
    return {"status": "ok", "email": email}

# ---------------------------------------------------------
# API KEY MGMT
# ---------------------------------------------------------
@app.get("/dev/create-key")
def create_key(
    email: EmailStr = Query(...),
    name: str = Query("default"),
    length: int = Query(17)
):
    user = dev_users_col.find_one({"email": email})
    if not user:
        raise HTTPException(404, "Developer not found")

    if api_keys_col.find_one({"owner_email": email}):
        raise HTTPException(400, "Developer already has an API key")

    if length not in (13, 17): length = 17

    for _ in range(10):
        key = generate_api_key(length)
        if not api_keys_col.find_one({"api_key": key}):
            break

    api_keys_col.insert_one({
        "api_key": key,
        "name": name,
        "owner_email": email,
        "created_at": now_utc(),
        "daily_limit": DAILY_LIMIT_DEFAULT,
        "disabled": False,
        "total_calls": 0,
        "daily": {}
    })

    return {"status": "success", "api_key": key}

@app.get("/dev/revoke-key")
def revoke_key(api_key: str = Query(...), email: EmailStr = Query(...)):
    doc = api_keys_col.find_one({"api_key": api_key, "owner_email": email})
    if not doc:
        raise HTTPException(404, "Key not found or owner mismatch")

    api_keys_col.update_one({"api_key": api_key}, {"$set": {"disabled": True}})
    return {"status": "success"}

@app.get("/dev/regenerate-key")
def regenerate_key(email: EmailStr = Query(...), length: int = Query(17)):
    doc = api_keys_col.find_one({"owner_email": email})
    if not doc:
        raise HTTPException(404, "No key to regenerate")

    api_keys_col.delete_one({"owner_email": email})

    if length not in (13, 17): length = 17
    key = generate_api_key(length)

    api_keys_col.insert_one({
        "api_key": key,
        "name": "regenerated",
        "owner_email": email,
        "created_at": now_utc(),
        "daily_limit": doc.get("daily_limit", DAILY_LIMIT_DEFAULT),
        "total_calls": 0,
        "disabled": False,
        "daily": {}
    })

    return {"status": "success", "api_key": key}

# ---------------------------------------------------------
# /dev/ask — MAIN ENDPOINT
# ---------------------------------------------------------
@app.get("/dev/ask")
async def dev_ask(
    api_key: str = Query(...),
    question: str = Query(...),
    background_tasks: BackgroundTasks = None
):
    # Validate
    key_doc = api_keys_col.find_one({"api_key": api_key})
    if not key_doc: raise HTTPException(401, "Invalid API key")
    if key_doc.get("disabled"): raise HTTPException(403, "API key disabled")

    limit = key_doc.get("daily_limit", DAILY_LIMIT_DEFAULT)

    # Rate limit
    rate = await redis_incr_daily(api_key, limit)
    if not rate["allowed"]:
        raise HTTPException(429, f"Daily limit exceeded. {rate['calls_today']}/{limit}")

    # Warn at 5 remaining
    if rate["remaining"] == 5:
        if await should_send_notification_today(api_key):
            background_tasks.add_task(
                send_brevo,
                key_doc["owner_email"],
                api_key,
                rate["calls_today"],
                rate["remaining"],
                limit,
                rate["reset_at"]
            )

    start = time.monotonic()

    # Cache
    cache = await find_cached_answer(question)
    if cache:
        answer = cache["answer"]
        context = cache["context"]
        source = cache["source"]
        cache_hit = True
    else:
        # Gemini
        prompt = f"""
You are MedBot, a trusted AI medical assistant.
Question: "{question}"
Output JSON only: {{"medical_answer": "<text>"}}
"""
        try:
            res = gemini_model.generate_content(prompt)
            txt = res.text.strip()
            txt = txt.replace("```json","").replace("```","").strip()

            try:
                parsed = json.loads(txt)
                answer = parsed.get("medical_answer","").strip()
            except:
                answer = txt

        except Exception as e:
            answer = "Gemini error."

        # RAG
        context = None
        source = None
        try:
            r = requests.get(EXTERNAL_RAG_URL, params={"question": question}, timeout=12)
            if r.status_code == 200:
                j = r.json()
                context = j.get("answer")
                source = j.get("source")
        except:
            pass

        # Store in background
        try:
            emb = embedding_model.encode([question])[0].tolist()
            background_tasks.add_task(
                global_qa_col.update_one,
                {"normalized_question": normalize_text(question)},
                {"$setOnInsert": {
                    "question": question,
                    "normalized_question": normalize_text(question),
                    "answer": answer,
                    "context": context,
                    "source": source,
                    "embedding": emb,
                    "created_at": now_utc()
                }},
                True
            )
        except:
            pass

        cache_hit = False

    response_time_ms = int((time.monotonic() - start) * 1000)

    # Log usage
    usage = {
        "api_key": api_key,
        "question": question,
        "answer": answer,
        "context": context,
        "source": source,
        "cache_hit": cache_hit,
        "response_time_ms": response_time_ms,
        "timestamp_utc": now_utc(),
        "timestamp_ist": now_ist(),
    }
    background_tasks.add_task(api_usage_col.insert_one, usage)

    # Update totals
    def upd(k):
        today = now_ist().strftime("%Y-%m-%d")
        api_keys_col.update_one({"api_key": k},
            {"$inc": {"total_calls": 1, f"daily.{today}.count": 1}},
            upsert=True)

    background_tasks.add_task(upd, api_key)

    return {
        "answer": answer,
        "context": context,
        "source": source,
        "cache_hit": cache_hit,
        "response_time_ms": response_time_ms,
        "calls_today": rate["calls_today"],
        "remaining_today": rate["remaining"],
        "reset_at_ist": rate["reset_at"]
    }

# ---------------------------------------------------------
# Analytics + Logs
# ---------------------------------------------------------
@app.get("/dev/analytics")
def analytics(
    api_key: str = Query(...),
    date: Optional[str] = Query(None)
):
    doc = api_keys_col.find_one({"api_key": api_key})
    if not doc: raise HTTPException(404, "Key not found")

    dt = date or now_ist().strftime("%Y-%m-%d")
    try:
        start = datetime.strptime(f"{dt}T00:00:00+05:30", "%Y-%m-%dT%H:%M:%S%z")
        end = start + timedelta(days=1)
    except:
        raise HTTPException(400, "Invalid date format")

    calls = api_usage_col.count_documents({
        "api_key": api_key,
        "timestamp_ist": {"$gte": start, "$lt": end}
    })

    total = doc.get("total_calls", 0)
    remaining = max(0, doc.get("daily_limit", DAILY_LIMIT_DEFAULT) - calls)

    return {
        "api_key": api_key,
        "owner_email": doc.get("owner_email"),
        "date": dt,
        "calls_today": calls,
        "remaining_today": remaining,
        "total_calls": total
    }

@app.get("/dev/logs")
def logs(
    api_key: str = Query(...),
    date: Optional[str] = Query(None),
    limit: int = Query(100)
):
    if date:
        try:
            start = datetime.strptime(f"{date}T00:00:00+05:30", "%Y-%m-%dT%H:%M:%S%z")
            end = start + timedelta(days=1)
            q = {"api_key": api_key, "timestamp_ist": {"$gte": start, "$lt": end}}
        except:
            raise HTTPException(400, "Invalid date")
    else:
        q = {"api_key": api_key}

    docs = list(api_usage_col.find(q).sort("timestamp_utc", -1).limit(limit))

    return {
        "api_key": api_key,
        "count": len(docs),
        "logs": [{
            "question": d.get("question"),
            "answer_snippet": (d.get("answer") or "")[:200],
            "timestamp_ist": d.get("timestamp_ist").isoformat() if d.get("timestamp_ist") else None,
            "cache_hit": d.get("cache_hit"),
            "response_time_ms": d.get("response_time_ms"),
            "source": d.get("source")
        } for d in docs]
    }

# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/health")
async def health():
    rstat = "ok"
    if REDIS_AVAILABLE:
        try:
            await redis.ping()
        except:
            rstat = "error"

    try:
        client.server_info()
        mstat = "ok"
    except:
        mstat = "error"

    return {
        "status": "ok",
        "redis": rstat,
        "mongo": mstat,
        "brevo": "configured" if BREVO_AVAILABLE else "missing"
    }

# ---------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------
@app.on_event("startup")
async def on_start():
    print("Developer API Gateway Started")

@app.on_event("shutdown")
async def on_stop():
    client.close()
    print("Shutting down...")

# ---------------------------------------------------------
# Local Run
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
