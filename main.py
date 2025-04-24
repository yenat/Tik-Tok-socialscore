# filename: main.py

import os
import json
import re
import math
import logging
import httpx
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, validator, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import sklearn
from sklearn._loss._loss import CyHalfSquaredError
import asyncio

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api.log")]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Social Score API",
    description="API for calculating social scores from TikTok profiles",
    version="1.0.0",
    docs_url="/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DEFAULT_MIN_SCORE = 300
DEFAULT_MAX_SCORE = 850
CALLBACK_TIMEOUT = 10
CALLBACK_RETRIES = 3
MAX_BIO_LENGTH = 500

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "tiktok_scoring_model.pkl")
SCALING_PARAMS_PATH = os.getenv("SCALING_PARAMS_PATH", "scaling_params.pkl")

# Fix missing attribute for sklearn error
if not hasattr(sklearn._loss._loss, '__pyx_unpickle_CyHalfSquaredError'):
    sklearn._loss._loss.__pyx_unpickle_CyHalfSquaredError = lambda *args: CyHalfSquaredError

# Load model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaling_params = joblib.load(SCALING_PARAMS_PATH)
        logger.info("Model and scaling parameters loaded")
        return model, scaling_params
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

try:
    model, scaling_params = load_model()
except HTTPException as e:
    raise e

class SocialMediaProfile(BaseModel):
    social_media: str
    username: str

    @validator('social_media')
    def validate_platform(cls, v):
        v = v.lower()
        if v not in ["tiktok", "facebook"]:  # Allow both but only process TikTok
            raise ValueError("Unsupported platform")
        return v

class ScoreRequest(BaseModel):
    fayda_number: str = Field(..., min_length=5)
    data: List[SocialMediaProfile]
    callbackUrl: Optional[HttpUrl] = None

class SocialScoreResponse(BaseModel):
    fayda_number: str
    socialscore: int
    trust_level: str
    score_breakdown: Dict[str, float]
    timestamp: str

# Helpers
def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0

def scale_score(raw_score: float) -> int:
    try:
        range_ = scaling_params['score_max'] - scaling_params['score_min']
        scaled = DEFAULT_MIN_SCORE + (DEFAULT_MAX_SCORE - DEFAULT_MIN_SCORE) * (
            (raw_score - scaling_params['score_min']) / range_
        )
        return min(DEFAULT_MAX_SCORE, max(DEFAULT_MIN_SCORE, int(scaled)))
    except:
        return DEFAULT_MIN_SCORE

def get_trust_level(score: int) -> str:
    if score >= 750: return "Very High"
    if score >= 650: return "High"
    if score >= 550: return "Medium"
    return "Low"

async def fetch_tiktok_data(username: str) -> Optional[Dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/html",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.tiktok.com/"
    }

    profile = {
        "username": username,
        "biography": "",
        "is_verified": False,
        "followers": 0,
        "following": 0,
        "likes": 0,
        "videos_count": 0
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # First try the mobile API endpoint
            mobile_url = f"https://m.tiktok.com/api/user/detail/?uniqueId={username}"
            try:
                mobile_res = await client.get(mobile_url, headers=headers)
                if mobile_res.status_code == 200:
                    data = mobile_res.json()
                    if 'userInfo' in data:
                        user = data['userInfo']['user']
                        stats = user.get('stats', {})
                        profile.update({
                            "is_verified": user.get("verified", False),
                            "followers": stats.get("followerCount", 0),
                            "following": stats.get("followingCount", 0),
                            "likes": stats.get("heartCount", 0),
                            "videos_count": stats.get("videoCount", 0)
                        })
            except Exception as e:
                logger.warning(f"Mobile API request failed: {str(e)}")

            # If we didn't get valid data from mobile API, try HTML parsing
            if profile["followers"] == 0 and profile["videos_count"] == 0:
                html_url = f"https://www.tiktok.com/@{username}"
                html_res = await client.get(html_url, headers=headers)
                if html_res.status_code == 200:
                    html = html_res.text
                    
                    # Try to extract from SIGI_STATE
                    if match := re.search(r'<script id="SIGI_STATE"[^>]*>(.*?)</script>', html):
                        try:
                            data = json.loads(match.group(1))
                            if 'UserModule' in data:
                                user = data['UserModule']['users'].get(username)
                                if user:
                                    profile.update({
                                        "is_verified": user.get("verified", False),
                                        "followers": user.get("followerCount", 0),
                                        "following": user.get("followingCount", 0),
                                        "likes": user.get("heartCount", 0),
                                        "videos_count": user.get("videoCount", 0)
                                    })
                        except json.JSONDecodeError as e:
                            logger.warning(f"SIGI_STATE parsing failed: {str(e)}")

                    # Additional fallback parsing if needed
                    if profile["followers"] == 0:
                        if match := re.search(r'"followerCount":(\d+)', html):
                            profile["followers"] = int(match.group(1))
                        if match := re.search(r'"heartCount":(\d+)', html):
                            profile["likes"] = int(match.group(1))
                        if match := re.search(r'"videoCount":(\d+)', html):
                            profile["videos_count"] = int(match.group(1))

            # Get bio from oEmbed if we don't have it yet
            if not profile["biography"]:
                oembed_url = f"https://www.tiktok.com/oembed?url=https://www.tiktok.com/@{username}"
                try:
                    oembed_res = await client.get(oembed_url, headers=headers)
                    if oembed_res.status_code == 200:
                        oembed_data = oembed_res.json()
                        profile["biography"] = oembed_data.get("author_name", "")
                except Exception as e:
                    logger.warning(f"oEmbed request failed: {str(e)}")

            # Consider profile valid if we got at least some data
            if (profile["followers"] > 0 or 
                profile["videos_count"] > 0 or 
                profile["likes"] > 0 or 
                profile["biography"]):
                return profile

            logger.warning(f"Incomplete data for username: {username}")
            return None

    except Exception as e:
        logger.error(f"Error fetching TikTok profile: {str(e)}")
        return None
def calculate_features(profile: Dict) -> Dict:
    followers = max(profile.get("followers", 0), 0)
    likes = max(profile.get("likes", 0), 0)
    posts = max(profile.get("videos_count", 0), 0)
    bio = profile.get("biography", "")

    return {
        "profile_score": min(100, (40 if profile.get("is_verified") else 0) + min(len(bio), MAX_BIO_LENGTH) * 0.06),
        "engagement_score": min(100, safe_divide(likes, followers) * 100),
        "network_score": min(100, math.log1p(followers) * 10),
        "activity_score": min(100, (posts * 0.1) + math.log1p(likes) * 0.5)
    }
@app.post("/calculate-score", response_model=SocialScoreResponse)
async def calculate_score(request: ScoreRequest, background_tasks: BackgroundTasks):
    # Find the first TikTok profile in the request
    tiktok_profile = next((p for p in request.data if p.social_media.lower() == "tiktok"), None)
    
    if not tiktok_profile:
        error_msg = "No TikTok profile found in request"
        error_response = {
            "status": "error",
            "message": error_msg,
            "received_profiles": [p.social_media for p in request.data],
            "supported_platform": "tiktok"
        }
        if request.callbackUrl:
            background_tasks.add_task(send_callback, str(request.callbackUrl), error_response)
        raise HTTPException(400, detail=error_response)

    # Log if other platforms were included but ignored
    other_profiles = [p for p in request.data if p.social_media.lower() != "tiktok"]
    if other_profiles:
        logger.info(f"Ignoring non-TikTok profiles: {[p.social_media for p in other_profiles]}")

    # Fetch TikTok profile data
    data = await fetch_tiktok_data(tiktok_profile.username)
    
    # Handle invalid TikTok profiles
    if not data:
        error_msg = f"No valid TikTok account found for username: @{tiktok_profile.username}"
        error_response = {
            "status": "error",
            "message": error_msg,
            "username": tiktok_profile.username,
            "suggestion": "Please check the username is correct and the account is public"
        }
        
        if request.callbackUrl:
            logger.info(f"Sending error callback for {tiktok_profile.username}")
            background_tasks.add_task(send_callback, str(request.callbackUrl), error_response)
        
        raise HTTPException(404, detail=error_response)

    # Calculate score
    try:
        features = calculate_features(data)
        df = pd.DataFrame([[features[k] for k in ['profile_score', 'engagement_score', 'network_score', 'activity_score']]],
                        columns=['profile_score', 'engagement_score', 'network_score', 'activity_score'])
        raw_score = float(model.predict(df)[0])
        score = scale_score(raw_score)
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        error_response = {
            "status": "error",
            "message": "Scoring calculation failed",
            "username": tiktok_profile.username,
            "error_details": str(e)
        }
        if request.callbackUrl:
            background_tasks.add_task(send_callback, str(request.callbackUrl), error_response)
        raise HTTPException(500, detail=error_response)

    # Prepare success response
    response = SocialScoreResponse(
        fayda_number=request.fayda_number,
        socialscore=score,
        trust_level=get_trust_level(score),
        score_breakdown={k: round(v, 2) for k, v in features.items()},
        timestamp=datetime.utcnow().isoformat()
    )

    # Send success callback if URL provided
    if request.callbackUrl:
        logger.info(f"Sending success callback for {tiktok_profile.username} to {request.callbackUrl}")
        background_tasks.add_task(
            send_callback, 
            str(request.callbackUrl), 
            response.dict()
        )

    return response


async def send_callback(url: str, data: Dict) -> bool:
    """Enhanced callback sender with detailed logging"""
    callback_id = f"callback_{datetime.utcnow().timestamp()}"
    logger.info(f"[{callback_id}] Initiating callback to {url}")
    
    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
            for attempt in range(1, CALLBACK_RETRIES + 1):
                try:
                    logger.info(f"[{callback_id}] Attempt {attempt}/{CALLBACK_RETRIES}")
                    response = await client.post(url, json=data)
                    response.raise_for_status()
                    logger.info(f"[{callback_id}] Callback successful")
                    return True
                except httpx.RequestError as e:
                    logger.warning(f"[{callback_id}] Request failed: {str(e)}")
                except httpx.HTTPStatusError as e:
                    logger.warning(f"[{callback_id}] HTTP error: {str(e)}")
                
                if attempt < CALLBACK_RETRIES:
                    await asyncio.sleep(1 * attempt)  # Exponential backoff
                    
    except Exception as e:
        logger.error(f"[{callback_id}] Callback failed completely: {str(e)}")
    
    return False
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/debug/tiktok/{username}")
async def debug(username: str):
    data = await fetch_tiktok_data(username)
    if not data:
        raise HTTPException(404, "TikTok profile not found")
    return data

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
