"""
CPRS Web Application — Competitive Programming Recommendation System.

A FastAPI backend serving:
- A public landing page describing the product
- User auth (register with platform handles / login)
- Profile management (name, email, avatar, handles)
- Cross-platform performance reports
- Personalized problem recommendations

Everything except the landing page, the auth pages and the subscribe form
requires a session cookie.

Usage:
    cd cprs && python -m uvicorn app:app --reload --port 8000
"""
import base64
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query, Cookie, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetchers.codeforces import (
    fetch_user_submissions, fetch_user_info, fetch_user_rating_history,
)
from fetchers.atcoder import (
    fetch_user_submissions as fetch_ac_submissions,
    fetch_user_contest_history as fetch_ac_contests,
)
from fetchers.leetcode import (
    fetch_user_submissions as fetch_lc_submissions,
    fetch_user_profile as fetch_lc_profile,
    fetch_user_contest_history as fetch_lc_contests,
)
from models import contests as contest_analysis
from models.collaborative import CollaborativeModel
from models.recommender import RecommenderEngine, TopicMastery
from models.database import (
    create_user, authenticate, create_session, get_user_by_session,
    delete_session, set_handle, get_handles, remove_handle,
    get_profile, update_profile, add_subscriber,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

PLATFORMS = ("codeforces", "atcoder", "leetcode")
PLATFORM_LABELS = {
    "codeforces": "Codeforces",
    "atcoder": "AtCoder",
    "leetcode": "LeetCode",
}

dataset_path = DATA_DIR / "cprs_unified_tagged.json"
if not dataset_path.exists():
    dataset_path = DATA_DIR / "cprs_unified.json"

engine = RecommenderEngine(str(dataset_path))

# Collaborative filtering, loaded once and applied by folding each request's
# user into the fixed item factors. Optional: without the artefact the app
# serves content-based recommendations exactly as before.
#
# One solve inside the item space is enough for CF to be worth using — the
# cold-start study measured nDCG@10 rising from 0.001 with no history to 0.135
# after a single observed solve. Below that CF has no opinion at all and the
# content scorer carries the request.
CF_MODEL_PATH = DATA_DIR / "cf_model.npz"
MIN_CF_SOLVES = 1

try:
    cf_model = CollaborativeModel.load(CF_MODEL_PATH) if CF_MODEL_PATH.exists() else None
except Exception as e:  # pragma: no cover - a corrupt artefact must not take the app down
    logger.warning(f"Could not load CF model from {CF_MODEL_PATH}: {e}")
    cf_model = None

if cf_model is None:
    logger.warning(
        "No CF model loaded — recommendations will be content-based only. "
        "Build one with: python scripts/train_cf.py"
    )


def _cf_affinity(solved_ids: set) -> tuple:
    """
    Collaborative-filtering scores for one user, via fold-in.

    Returns (affinity, info) where affinity is {cprs_id: score} over the
    problems CF can speak about, or None when it cannot speak at all.
    """
    if cf_model is None or not cf_model.problem_ids:
        return None, {"available": False, "reason": "no model loaded"}

    cols = [cf_model.column_of[pid] for pid in solved_ids if pid in cf_model.column_of]
    info = {
        "available": False,
        "matched_solves": len(cols),
        "item_space": len(cf_model.problem_ids),
        "catalogue": len(engine.problems),
    }
    if len(cols) < MIN_CF_SOLVES:
        info["reason"] = (
            "none of your solved problems are in the interaction dataset, so "
            "collaborative filtering has no signal for you yet"
        )
        return None, info

    scores = cf_model.score_folded(cols)
    info["available"] = True
    return dict(zip(cf_model.problem_ids, scores.tolist())), info

app = FastAPI(title="CPRS", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Platform fetch cache ---------------------------------------------------
# Profile, report and recommendation views all need the same submission
# history. Without a cache each page load re-hits the platform APIs, which for
# a heavy user is several seconds and a lot of needless upstream traffic.

_CACHE: dict = {}
_CACHE_TTL = 600  # seconds


def _fetch_platform(platform: str, handle: str) -> tuple:
    """Return (submissions, info) for a handle, memoised for _CACHE_TTL."""
    key = (platform, handle.lower())
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1], hit[2]

    if platform == "codeforces":
        info = fetch_user_info(handle)
        subs = fetch_user_submissions(handle)
    elif platform == "atcoder":
        info = {}
        subs = fetch_ac_submissions(handle)
        if not subs:
            raise ValueError("no submissions returned")
    elif platform == "leetcode":
        info = fetch_lc_profile(handle)
        subs = fetch_lc_submissions(handle)
    else:
        raise ValueError(f"unknown platform {platform}")

    _CACHE[key] = (time.time(), subs, info)
    return subs, info


def _build_profile(platform: str, handle: str):
    """Fetch and build a UserProfile for one platform."""
    subs, info = _fetch_platform(platform, handle)
    if platform == "codeforces":
        profile = engine.build_user_profile(subs, handle, rating=info.get("rating"))
    elif platform == "atcoder":
        profile = engine.build_atcoder_profile(subs, handle)
    else:
        profile = engine.build_leetcode_profile(subs, info, handle)
    return profile, subs


# --- Auth helpers -----------------------------------------------------------

def get_current_user(session: Optional[str] = Cookie(None, alias="cprs_session")):
    if not session:
        return None
    return get_user_by_session(session)


def require_user(session: Optional[str]):
    user = get_current_user(session)
    if not user:
        raise HTTPException(401, "Not logged in")
    return user


def _page(name: str) -> str:
    return (STATIC_DIR / name).read_text()


# --- Pages ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing():
    """Public marketing page. No product functionality here."""
    return _page("landing.html")


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _page("auth.html")


@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return _page("auth.html")


@app.get("/app", response_class=HTMLResponse)
async def app_page(cprs_session: Optional[str] = Cookie(None)):
    if not get_current_user(cprs_session):
        return RedirectResponse("/login?next=/app", status_code=302)
    return HTMLResponse(_page("app.html"))


@app.get("/profile", response_class=HTMLResponse)
async def profile_page(cprs_session: Optional[str] = Cookie(None)):
    if not get_current_user(cprs_session):
        return RedirectResponse("/login?next=/profile", status_code=302)
    return HTMLResponse(_page("profile.html"))


@app.get("/report", response_class=HTMLResponse)
async def report_page(cprs_session: Optional[str] = Cookie(None)):
    if not get_current_user(cprs_session):
        return RedirectResponse("/login?next=/report", status_code=302)
    return HTMLResponse(_page("report.html"))


@app.get("/contests", response_class=HTMLResponse)
async def contests_page(cprs_session: Optional[str] = Cookie(None)):
    if not get_current_user(cprs_session):
        return RedirectResponse("/login?next=/contests", status_code=302)
    return HTMLResponse(_page("contests.html"))


# --- Auth API ---------------------------------------------------------------

@app.post("/api/register")
async def register(response: Response, payload: dict = Body(...)):
    """
    Register a user and attach their platform handles in one step.

    Body: {username, password, email?, first_name?, last_name?,
           handles: {codeforces?, atcoder?, leetcode?}}
    """
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if len(username) < 3 or len(password) < 4:
        raise HTTPException(400, "Username must be 3+ characters and password 4+")

    user_id = create_user(
        username,
        password,
        email=(payload.get("email") or "").strip() or None,
        first_name=(payload.get("first_name") or "").strip() or None,
        last_name=(payload.get("last_name") or "").strip() or None,
    )
    if user_id is None:
        raise HTTPException(409, "That username is already taken")

    handles = payload.get("handles") or {}
    saved = []
    for platform in PLATFORMS:
        handle = (handles.get(platform) or "").strip()
        if handle:
            set_handle(user_id, platform, handle)
            saved.append(platform)

    token = create_session(user_id)
    response.set_cookie("cprs_session", token, httponly=True, max_age=30*24*3600)
    return {"ok": True, "username": username, "handles_saved": saved}


@app.post("/api/login")
async def login(response: Response, payload: dict = Body(...)):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    user_id = authenticate(username, password)
    if user_id is None:
        raise HTTPException(401, "Incorrect username or password")
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
    profile = get_profile(user["id"]) or {}
    return {
        "logged_in": True,
        "username": user["username"],
        "first_name": profile.get("first_name"),
        "last_name": profile.get("last_name"),
        "email": profile.get("email"),
        "avatar": profile.get("avatar"),
        "handles": profile.get("handles", {}),
    }


# --- Profile ----------------------------------------------------------------

MAX_AVATAR_BYTES = 800_000


@app.get("/api/profile")
async def read_profile(cprs_session: Optional[str] = Cookie(None)):
    user = require_user(cprs_session)
    profile = get_profile(user["id"])
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile


@app.put("/api/profile")
async def write_profile(
    payload: dict = Body(...), cprs_session: Optional[str] = Cookie(None)
):
    """Update name / email / avatar and the full set of platform handles."""
    user = require_user(cprs_session)

    avatar = payload.get("avatar")
    if avatar:
        if not avatar.startswith("data:image/"):
            raise HTTPException(400, "Avatar must be an image")
        # A data URL is ~4/3 the size of the bytes it encodes.
        b64 = avatar.split(",", 1)[-1]
        if len(base64.b64decode(b64 + "==", validate=False)) > MAX_AVATAR_BYTES:
            raise HTTPException(413, "Image too large — please use one under 800 KB")

    update_profile(
        user["id"],
        email=payload.get("email"),
        first_name=payload.get("first_name"),
        last_name=payload.get("last_name"),
        avatar=avatar,
    )

    if "handles" in payload:
        handles = payload["handles"] or {}
        for platform in PLATFORMS:
            handle = (handles.get(platform) or "").strip()
            if handle:
                set_handle(user["id"], platform, handle)
            else:
                remove_handle(user["id"], platform)

    return get_profile(user["id"])


# --- Handle management ------------------------------------------------------

@app.post("/api/handles")
async def update_handle(
    platform: str = Query(...),
    handle: str = Query(...),
    cprs_session: Optional[str] = Cookie(None),
):
    user = require_user(cprs_session)
    if platform not in PLATFORMS:
        raise HTTPException(400, "Invalid platform")
    set_handle(user["id"], platform, handle)
    return {"ok": True, "platform": platform, "handle": handle}


@app.delete("/api/handles")
async def delete_handle(
    platform: str = Query(...),
    cprs_session: Optional[str] = Cookie(None),
):
    user = require_user(cprs_session)
    remove_handle(user["id"], platform)
    return {"ok": True}


@app.post("/api/handles/verify")
async def verify_handle(platform: str = Query(...), handle: str = Query(...)):
    """Check a handle resolves before the user commits to it at registration."""
    if platform not in PLATFORMS:
        raise HTTPException(400, "Invalid platform")
    try:
        subs, info = _fetch_platform(platform, handle.strip())
    except Exception as e:
        logger.info(f"Handle check failed for {platform}:{handle} — {e}")
        return {"ok": False, "detail": f"No {PLATFORM_LABELS[platform]} user '{handle}'"}
    solved = len({
        s.get("problem", {}).get("name") or s.get("title") or str(i)
        for i, s in enumerate(subs)
    })
    return {"ok": True, "submissions": len(subs), "distinct": solved,
            "rating": info.get("rating") if isinstance(info, dict) else None}


# --- Subscribe (public) -----------------------------------------------------

@app.post("/api/subscribe")
async def subscribe(payload: dict = Body(...)):
    email = (payload.get("email") or "").strip()
    if "@" not in email or len(email) < 5:
        raise HTTPException(400, "Please enter a valid email address")
    add_subscriber(email, (payload.get("message") or "").strip() or None)
    return {"ok": True}


# --- Reports ----------------------------------------------------------------

def _mastery_rows(profile, limit: Optional[int] = None):
    rows = []
    for topic, m in profile.topic_mastery.items():
        if isinstance(m, dict):
            m = TopicMastery(**m)
        rows.append({
            "topic": topic,
            "solved": m.problems_solved,
            "attempted": m.problems_attempted,
            "solve_rate": round(m.solve_rate, 3),
            "avg_difficulty": round(m.avg_difficulty, 3),
            "max_difficulty": round(m.max_difficulty, 3),
        })
    rows.sort(key=lambda x: x["solved"], reverse=True)
    return rows[:limit] if limit else rows


@app.get("/api/report")
async def report(
    platform: str = Query("all", description="one platform, or 'all' to merge"),
    last_n: int = Query(50),
    cprs_session: Optional[str] = Cookie(None),
):
    """Performance report for the logged-in user, per platform or merged."""
    user = require_user(cprs_session)
    handles = get_handles(user["id"])
    if not handles:
        raise HTTPException(400, "No platform handles configured yet")

    wanted = PLATFORMS if platform == "all" else (platform,)
    if platform != "all" and platform not in PLATFORMS:
        raise HTTPException(400, f"Unknown platform '{platform}'")

    profiles, per_platform, errors = [], {}, {}
    for p in wanted:
        handle = handles.get(p)
        if not handle:
            continue
        try:
            prof, subs = _build_profile(p, handle)
        except Exception as e:
            logger.warning(f"Report fetch failed for {p}:{handle} — {e}")
            errors[p] = f"Could not load {PLATFORM_LABELS[p]} data for '{handle}'"
            continue
        profiles.append(prof)
        rp = engine.analyze_recent_performance(subs, p, last_n)

        # How much of this platform's history we could actually map onto the
        # unified problem set. LeetCode only exposes the 20 most recent
        # accepted submissions, and the newest problems postdate our dataset
        # snapshot, so topic coverage there is often partial or empty.
        matched = len(prof.solved_problem_ids)
        coverage = {
            "history_seen": len(subs),
            "matched_to_dataset": matched,
            "note": None,
        }
        if matched == 0:
            coverage["note"] = (
                f"{PLATFORM_LABELS[p]} exposes only the {len(subs)} most recent "
                "accepted submissions, and none of them appear in our problem "
                "snapshot yet — so totals are shown, but topic-level detail "
                "is not available for this account."
            )
        elif matched < len(subs) * 0.5:
            coverage["note"] = (
                f"Only {matched} of the {len(subs)} submissions "
                f"{PLATFORM_LABELS[p]} exposes matched our problem snapshot, so "
                "topic detail here is partial."
            )

        per_platform[p] = {
            "handle": handle,
            "rating": prof.rating,
            "total_solved": prof.total_solved,
            "total_attempted": prof.total_attempted,
            "solve_rate": round(prof.overall_solve_rate, 3),
            "difficulty_level": round(prof.difficulty_level, 3),
            "coverage": coverage,
            "topics": _mastery_rows(prof),
            "recent": {
                "last_n": rp.last_n,
                "solved": rp.solved,
                "attempted": rp.attempted,
                "solve_rate": round(rp.solve_rate, 3),
                "avg_difficulty": round(rp.avg_difficulty, 3),
                "topics_practiced": rp.topics_practiced,
                "trend": rp.trend,
            },
        }

    if not profiles:
        raise HTTPException(
            502, "; ".join(errors.values()) or "No data available for your handles"
        )

    merged = engine.merge_profiles(profiles) if len(profiles) > 1 else profiles[0]
    topics = _mastery_rows(merged)
    strengths = sorted(
        [t for t in topics if t["solved"] >= 3],
        key=lambda t: (t["solve_rate"], t["solved"]),
        reverse=True,
    )[:5]
    weaknesses = sorted(
        [t for t in topics if t["attempted"] >= 3],
        key=lambda t: (t["solve_rate"], -t["attempted"]),
    )[:5]

    return {
        "scope": platform,
        "connected": list(per_platform.keys()),
        "errors": errors,
        "summary": {
            "total_solved": merged.total_solved,
            "total_attempted": merged.total_attempted,
            "solve_rate": round(merged.overall_solve_rate, 3),
            "difficulty_level": round(merged.difficulty_level, 3),
            "topics_covered": len([t for t in topics if t["solved"] > 0]),
            "platforms": len(per_platform),
        },
        "topics": topics,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "per_platform": per_platform,
    }


# --- Contests ---------------------------------------------------------------

_CONTEST_CACHE: dict = {}


def _contest_records(platform: str, handle: str) -> list:
    """Fetch and normalise one platform's contest history, memoised."""
    key = (platform, handle.lower())
    hit = _CONTEST_CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    if platform == "codeforces":
        records = contest_analysis.from_codeforces(fetch_user_rating_history(handle))
    elif platform == "atcoder":
        records = contest_analysis.from_atcoder(fetch_ac_contests(handle))
    else:
        records = contest_analysis.from_leetcode(fetch_lc_contests(handle))

    _CONTEST_CACHE[key] = (time.time(), records)
    return records


@app.get("/api/contests")
async def contests(
    platform: str = Query("all"),
    cprs_session: Optional[str] = Cookie(None),
):
    """Contest performance and derived improvement points, per platform."""
    user = require_user(cprs_session)
    handles = get_handles(user["id"])
    if not handles:
        raise HTTPException(400, "No platform handles configured yet")

    if platform != "all" and platform not in PLATFORMS:
        raise HTTPException(400, f"Unknown platform '{platform}'")
    wanted = PLATFORMS if platform == "all" else (platform,)

    per_platform, errors = {}, {}
    for p in wanted:
        handle = handles.get(p)
        if not handle:
            continue
        try:
            records = _contest_records(p, handle)
        except Exception as e:
            logger.warning(f"Contest fetch failed for {p}:{handle} — {e}")
            errors[p] = f"Could not load {PLATFORM_LABELS[p]} contest history for '{handle}'"
            continue
        analysis = contest_analysis.analyse(records)
        analysis["handle"] = handle
        analysis["label"] = PLATFORM_LABELS[p]
        per_platform[p] = analysis

    if not per_platform:
        raise HTTPException(
            502, "; ".join(errors.values()) or "No contest history available"
        )

    return {
        "scope": platform,
        "errors": errors,
        "overall": contest_analysis.merge(per_platform),
        "per_platform": per_platform,
    }


# --- Recommendations --------------------------------------------------------

@app.get("/api/recommend")
async def recommend(
    handle: Optional[str] = Query(None, description="Public demo lookup, no login"),
    handle_platform: str = Query("codeforces"),
    n: int = Query(20, ge=1, le=50),
    platforms: str = Query("all"),
    last_n: int = Query(50),
    cprs_session: Optional[str] = Cookie(None),
):
    """
    Recommendations for the logged-in user, merged across their handles.

    `handle`/`handle_platform` remain as a public single-platform demo path
    (used by no UI page) so the API can be exercised without an account.
    `platforms` filters which platforms recommendations are drawn from.
    """
    user = get_current_user(cprs_session)
    profiles = []
    recent_performances = {}

    if user:
        handles = get_handles(user["id"])
        if not handles:
            raise HTTPException(400, "No platform handles configured. Add them in your profile.")

        errors = {}
        for p in PLATFORMS:
            if p not in handles:
                continue
            try:
                prof, subs = _build_profile(p, handles[p])
            except Exception as e:
                logger.warning(f"Failed to fetch {p} data for {handles[p]}: {e}")
                errors[p] = str(e)
                continue
            profiles.append(prof)
            recent_performances[p] = engine.analyze_recent_performance(subs, p, last_n)

        if not profiles:
            raise HTTPException(502, "Could not load data for any of your handles")

        profile = engine.merge_profiles(profiles) if len(profiles) > 1 else profiles[0]
        profile.handle = user["username"]

        # Contest results, where they exist, are a better estimate of how hard
        # a problem this user can handle than the difficulty of what they have
        # solved in practice. Let them set the difficulty target; solve history
        # keeps deciding which topics to aim at.
        calibrations = []
        for p in PLATFORMS:
            if p not in handles:
                continue
            try:
                records = _contest_records(p, handles[p])
            except Exception as e:
                logger.info(f"No contest calibration from {p}: {e}")
                continue
            cal = contest_analysis.difficulty_calibration(
                contest_analysis.analyse(records), p
            )
            if cal:
                calibrations.append(cal)

        calibration = contest_analysis.blend_calibrations(
            calibrations, profile.difficulty_level
        )
        profile.difficulty_level = calibration["level"]
        stretch = calibration["stretch"]
        cf_affinity, cf_info = _cf_affinity(profile.solved_problem_ids)

    elif handle:
        if handle_platform not in PLATFORMS:
            raise HTTPException(400, f"Unknown platform '{handle_platform}'")
        label = PLATFORM_LABELS[handle_platform]
        try:
            profile, subs = _build_profile(handle_platform, handle)
        except Exception as e:
            logger.warning(f"Quick lookup failed for {label} user '{handle}': {e}")
            raise HTTPException(
                404,
                f"Could not load {label} user '{handle}'. Check the handle is "
                f"spelled correctly and belongs to {label}.",
            )
        recent_performances[handle_platform] = engine.analyze_recent_performance(
            subs, handle_platform, last_n
        )
        calibration = contest_analysis.blend_calibrations([], profile.difficulty_level)
        stretch = calibration["stretch"]
        cf_affinity, cf_info = _cf_affinity(profile.solved_problem_ids)
    else:
        raise HTTPException(401, "Log in to get recommendations")

    platform_filter = None
    if platforms != "all":
        platform_filter = [p.strip() for p in platforms.split(",")]

    recs = engine.recommend(
        profile, n=n, platforms=platform_filter, difficulty_stretch=stretch,
        cf_affinity=cf_affinity,
    )

    return {
        "profile": {
            "handle": profile.handle,
            "rating": profile.rating,
            "platform": profile.platform,
            "total_solved": profile.total_solved,
            "total_attempted": profile.total_attempted,
            "solve_rate": round(profile.overall_solve_rate, 3),
            "difficulty_level": round(profile.difficulty_level, 3),
            "topic_mastery": _mastery_rows(profile, limit=20),
        },
        "calibration": calibration,
        "collaborative": cf_info,
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
