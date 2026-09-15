"""
CPRS Web Application — Competitive Programming Recommendation System.

A FastAPI backend serving:
- User auth (register/login)
- Multi-platform handle management (CF, AtCoder, LeetCode)
- Cross-platform profile analysis with recent performance
- Personalized problem recommendations

Usage:
    cd cprs && python -m uvicorn app:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Cookie, Response
from fastapi.responses import HTMLResponse
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetchers.codeforces import fetch_user_submissions, fetch_user_info
from fetchers.atcoder import fetch_user_submissions as fetch_ac_submissions
from fetchers.leetcode import fetch_user_submissions as fetch_lc_submissions, fetch_user_profile as fetch_lc_profile
from models.recommender import RecommenderEngine, TopicMastery
from models.database import (
    create_user, authenticate, create_session, get_user_by_session,
    delete_session, set_handle, get_handles, remove_handle,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

dataset_path = DATA_DIR / "cprs_unified_tagged.json"
if not dataset_path.exists():
    dataset_path = DATA_DIR / "cprs_unified.json"

engine = RecommenderEngine(str(dataset_path))

app = FastAPI(title="CPRS", version="0.1.0")


# --- Auth helpers ---

def get_current_user(session: Optional[str] = Cookie(None, alias="cprs_session")):
    if not session:
        return None
    return get_user_by_session(session)


# --- Pages ---

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return html_path.read_text()


# --- Auth API ---

@app.post("/api/register")
async def register(response: Response, username: str = Query(...), password: str = Query(...)):
    if len(username) < 3 or len(password) < 4:
        raise HTTPException(400, "Username must be 3+ chars, password 4+ chars")
    user_id = create_user(username, password)
    if user_id is None:
        raise HTTPException(409, "Username already taken")
    token = create_session(user_id)
    response.set_cookie("cprs_session", token, httponly=True, max_age=30*24*3600)
    return {"ok": True, "username": username}


@app.post("/api/login")
async def login(response: Response, username: str = Query(...), password: str = Query(...)):
    user_id = authenticate(username, password)
    if user_id is None:
        raise HTTPException(401, "Invalid credentials")
    token = create_session(user_id)
    response.set_cookie("cprs_session", token, httponly=True, max_age=30*24*3600)
    return {"ok": True, "username": username}


@app.post("/api/logout")
async def logout(response: Response, cprs_session: Optional[str] = Cookie(None)):
    if cprs_session:
        delete_session(cprs_session)
    response.delete_cookie("cprs_session")
    return {"ok": True}


@app.get("/api/me")
async def me(cprs_session: Optional[str] = Cookie(None)):
    user = get_current_user(cprs_session)
    if not user:
        return {"logged_in": False}
    handles = get_handles(user["id"])
    return {"logged_in": True, "username": user["username"], "handles": handles}


# --- Handle management ---

@app.post("/api/handles")
async def update_handle(
    platform: str = Query(...),
    handle: str = Query(...),
    cprs_session: Optional[str] = Cookie(None),
):
    user = get_current_user(cprs_session)
    if not user:
        raise HTTPException(401, "Not logged in")
    if platform not in ("codeforces", "atcoder", "leetcode"):
        raise HTTPException(400, "Invalid platform")
    set_handle(user["id"], platform, handle)
    return {"ok": True, "platform": platform, "handle": handle}


@app.delete("/api/handles")
async def delete_handle(
    platform: str = Query(...),
    cprs_session: Optional[str] = Cookie(None),
):
    user = get_current_user(cprs_session)
    if not user:
        raise HTTPException(401, "Not logged in")
    remove_handle(user["id"], platform)
    return {"ok": True}


# --- Recommendations ---

def _build_mastery_list(profile):
    mastery_list = []
    for topic, m in profile.topic_mastery.items():
        if isinstance(m, dict):
            m = TopicMastery(**m)
        mastery_list.append({
            "topic": topic,
            "solved": m.problems_solved,
            "attempted": m.problems_attempted,
            "solve_rate": round(m.solve_rate, 3),
            "avg_difficulty": round(m.avg_difficulty, 3),
        })
    mastery_list.sort(key=lambda x: x["attempted"], reverse=True)
    return mastery_list[:20]


@app.get("/api/recommend")
async def recommend(
    handle: Optional[str] = Query(None, description="Codeforces handle (for quick mode)"),
    n: int = Query(20, ge=1, le=50),
    platforms: str = Query("all"),
    last_n: int = Query(50, description="Analyze last N problems for recent performance"),
    cprs_session: Optional[str] = Cookie(None),
):
    """
    Get recommendations. Two modes:
    1. Quick mode: pass ?handle=xxx (CF only, no login needed)
    2. Full mode: logged in user with multiple platform handles
    """
    user = get_current_user(cprs_session)
    profiles = []
    recent_performances = {}

    if user:
        handles = get_handles(user["id"])

        # Build profile from each platform
        if "codeforces" in handles:
            try:
                cf_info = fetch_user_info(handles["codeforces"])
                cf_subs = fetch_user_submissions(handles["codeforces"])
                cf_profile = engine.build_user_profile(
                    cf_subs, handles["codeforces"], rating=cf_info.get("rating")
                )
                profiles.append(cf_profile)
                recent_performances["codeforces"] = engine.analyze_recent_performance(
                    cf_subs, "codeforces", last_n
                )
            except Exception as e:
                logger.warning(f"Failed to fetch CF data for {handles['codeforces']}: {e}")

        if "atcoder" in handles:
            try:
                ac_subs = fetch_ac_submissions(handles["atcoder"])
                ac_profile = engine.build_atcoder_profile(ac_subs, handles["atcoder"])
                profiles.append(ac_profile)
                recent_performances["atcoder"] = engine.analyze_recent_performance(
                    ac_subs, "atcoder", last_n
                )
            except Exception as e:
                logger.warning(f"Failed to fetch AtCoder data for {handles['atcoder']}: {e}")

        if "leetcode" in handles:
            try:
                lc_info = fetch_lc_profile(handles["leetcode"])
                lc_subs = fetch_lc_submissions(handles["leetcode"])
                lc_profile = engine.build_leetcode_profile(
                    lc_subs, lc_info, handles["leetcode"]
                )
                profiles.append(lc_profile)
                recent_performances["leetcode"] = engine.analyze_recent_performance(
                    lc_subs, "leetcode", last_n
                )
            except Exception as e:
                logger.warning(f"Failed to fetch LC data for {handles['leetcode']}: {e}")

        if not profiles:
            raise HTTPException(400, "No valid platform handles configured. Add handles in settings.")

        profile = engine.merge_profiles(profiles)
        profile.handle = user["username"]

    elif handle:
        # Quick mode — CF only
        try:
            cf_info = fetch_user_info(handle)
        except Exception as e:
            raise HTTPException(404, f"User '{handle}' not found: {e}")
        try:
            cf_subs = fetch_user_submissions(handle)
        except Exception as e:
            raise HTTPException(500, f"Failed to fetch submissions: {e}")

        profile = engine.build_user_profile(cf_subs, handle, rating=cf_info.get("rating"))
        recent_performances["codeforces"] = engine.analyze_recent_performance(
            cf_subs, "codeforces", last_n
        )
    else:
        raise HTTPException(400, "Provide ?handle=xxx or log in with platform handles")

    platform_filter = None
    if platforms != "all":
        platform_filter = [p.strip() for p in platforms.split(",")]

    recs = engine.recommend(profile, n=n, platforms=platform_filter)

    return {
        "profile": {
            "handle": profile.handle,
            "rating": profile.rating,
            "platform": profile.platform,
            "total_solved": profile.total_solved,
            "total_attempted": profile.total_attempted,
            "solve_rate": round(profile.overall_solve_rate, 3),
            "difficulty_level": round(profile.difficulty_level, 3),
            "topic_mastery": _build_mastery_list(profile),
        },
        "recent_performance": {
            platform: {
                "last_n": rp.last_n,
                "solved": rp.solved,
                "attempted": rp.attempted,
                "solve_rate": round(rp.solve_rate, 3),
                "avg_difficulty": round(rp.avg_difficulty, 3),
                "topics_practiced": rp.topics_practiced,
                "trend": rp.trend,
            }
            for platform, rp in recent_performances.items()
        },
        "recommendations": [
            {
                "cprs_id": r.cprs_id,
                "platform": r.platform,
                "name": r.name,
                "url": r.url,
                "difficulty": round(r.difficulty_normalized, 3) if r.difficulty_normalized else None,
                "tags": r.tags,
                "score": round(r.score, 3),
                "reasons": r.reasons,
            }
            for r in recs
        ],
    }


@app.get("/api/stats")
async def dataset_stats():
    from collections import Counter
    plats = Counter(p["platform"] for p in engine.problems)
    return {
        "total_problems": len(engine.problems),
        "platforms": dict(plats),
        "with_tags": sum(1 for p in engine.problems if p.get("tags_unified")),
        "with_difficulty": sum(1 for p in engine.problems if p.get("difficulty_normalized") is not None),
        "unified_tags": len(engine.all_tags),
    }
