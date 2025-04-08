import os
import json
import re
import random
import asyncio
import math
import logging
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, validator, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from httpx import Timeout, Limits
import pickle
from sklearn._loss._loss import CyHalfSquaredError  # Import required for unpickling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Score Verification API",
    description="API for verifying social media accounts and calculating social scores",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VERIFICATION_CODE_EXPIRY_MINUTES = 15
MAX_VERIFICATION_ATTEMPTS = 5
DEFAULT_MIN_SCORE = 300
DEFAULT_MAX_SCORE = 850
CALLBACK_TIMEOUT = 10
CALLBACK_RETRIES = 3
MAX_BIO_LENGTH = 500

# Load Model with proper unpickling setup
MODEL_PATH = os.getenv('MODEL_PATH', 'tiktok_scoring_model.pkl')
SCALING_PARAMS_PATH = os.getenv('SCALING_PARAMS_PATH', 'scaling_params.pkl')

def custom_unpickler(file):
    """Custom unpickler that handles scikit-learn model loading"""
    def __pyx_unpickle_CyHalfSquaredError(_, __, ___):
        return CyHalfSquaredError
    
    # Create the unpickler
    unpickler = pickle.Unpickler(file)
    
    # Define the custom persistent_load function
    def persistent_load(pid):
        if pid[0] == '__pyx_unpickle_CyHalfSquaredError':
            return __pyx_unpickle_CyHalfSquaredError(*pid[1:])
        return unpickler.persistent_load(pid)
    
    # Assign the custom loader
    unpickler.persistent_load = persistent_load
    
    # Load and return the object
    return unpickler.load()

try:
    # Load model with custom unpickler
    with open(MODEL_PATH, 'rb') as f:
        model = custom_unpickler(f)
    
    # Load scaling params with custom unpickler
    with open(SCALING_PARAMS_PATH, 'rb') as f:
        scaling_params = custom_unpickler(f)
    
    logger.info("Model and scaling parameters loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Models (remain the same as before)
class SocialMediaProfile(BaseModel):
    social_media: str  # Remove the alias
    username: str

    @validator('social_media')
    def validate_platform(cls, v):
        v = v.lower()
        if v not in ["tiktok", "facebook"]:
            raise ValueError("Unsupported platform")
        return v

class VerificationRequest(BaseModel):
    fayda_number: str = Field(..., min_length=5)
    type: str
    data: List[SocialMediaProfile]
    callbackUrl: Optional[HttpUrl] = None

    @validator('type')
    def validate_type(cls, v):
        if v != "*SOCIAL_SCORE*":
            raise ValueError("Invalid type")
        return v

class VerificationResponse(BaseModel):
    verification_code: str
    instructions: str
    expires_in: str

class SocialScoreResponse(BaseModel):
    fayda_number: str
    socialscore: int
    trust_level: str
    score_breakdown: Dict[str, float]
    timestamp: str

class VerificationStatus(BaseModel):
    status: str
    message: Optional[str] = None
    verification_code: Optional[str] = None

# Storage
verification_storage = {}

# Helper Functions (remain the same as before)
def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def scale_score(raw_score: float) -> int:
    try:
        score_range = scaling_params['score_max'] - scaling_params['score_min']
        scaled = DEFAULT_MIN_SCORE + (DEFAULT_MAX_SCORE - DEFAULT_MIN_SCORE) * (
            (raw_score - scaling_params['score_min']) / score_range
        )
        return min(DEFAULT_MAX_SCORE, max(DEFAULT_MIN_SCORE, int(scaled)))
    except Exception:
        return DEFAULT_MIN_SCORE

def get_trust_level(score: int) -> str:
    if score >= 750: return "Very High"
    elif score >= 650: return "High"
    elif score >= 550: return "Medium"
    return "Low"

def generate_verification_code(identifier: str) -> str:
    code = str(random.randint(100000, 999999))
    verification_storage[identifier] = {
        "code": code,
        "expires": datetime.now() + timedelta(minutes=VERIFICATION_CODE_EXPIRY_MINUTES),
        "attempts": 0,
        "verified": False
    }
    return code

def get_verification_instructions(code: str, platform: str) -> str:
    platform = platform.lower()
    base = (
        f"=== {platform.upper()} VERIFICATION ===\n"
        f"1. Add to bio: {code}\n"
        f"2. Make account public\n"
        f"3. Save changes\n"
    )
    if platform == "tiktok":
        return base + "Location: Edit Profile > Bio"
    elif platform == "facebook":
        return base + "Location: Edit Profile > Intro"
    return base

async def fetch_social_media_data(platform: str, username: str) -> Optional[Dict]:
    platform = platform.lower()
    if platform == "tiktok":
        return await fetch_tiktok_data(username)
    elif platform == "facebook":
        return await fetch_facebook_data(username)
    return None

async def fetch_tiktok_data(username: str) -> Optional[Dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    
    try:
        timeout = Timeout(10.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"https://www.tiktok.com/@{username}",
                headers=headers,
                follow_redirects=True
            )
            response.raise_for_status()
            
            html = response.text
            match = re.search(r'"user":\s*({.+?})', html)
            if match:
                user_data = json.loads(match.group(1))
                return {
                    "username": username,
                    "biography": user_data.get("signature", ""),
                    "is_verified": user_data.get("verified", False),
                    "followers": user_data.get("followerCount", 0),
                    "following": user_data.get("followingCount", 0),
                    "likes": user_data.get("heartCount", 0),
                    "videos_count": user_data.get("videoCount", 0)
                }
    except Exception as e:
        logger.error(f"TikTok fetch error: {str(e)}")
    return None

async def fetch_facebook_data(username: str) -> Optional[Dict]:
    return {
        "username": username,
        "biography": "",
        "is_verified": False,
        "followers": 0,
        "following": 0,
        "likes": 0,
        "posts_count": 0
    }

async def check_bio_for_code(platform: str, username: str, fayda_number: str, code: str) -> bool:
    stored = verification_storage.get(fayda_number)
    if not stored or datetime.now() > stored["expires"]:
        return False
        
    profile_data = await fetch_social_media_data(platform, username)
    if not profile_data:
        return False

    bio = profile_data.get("biography", "")
    return code in "".join(c for c in bio if c.isdigit())

def calculate_features(profile: Dict) -> Dict:
    followers = max(profile.get('followers', 0), 0)
    likes = max(profile.get('likes', 0), 0)
    posts = max(profile.get('videos_count', profile.get('posts_count', 0)), 0)
    bio = profile.get('biography', "")
    
    return {
        'profile_score': min(100, (
            (40 if profile.get('is_verified') else 0) +
            (min(len(bio), MAX_BIO_LENGTH) * 0.06)
        )),
        'engagement_score': min(100, safe_divide(likes, followers) * 100),
        'network_score': min(100, math.log1p(followers) * 10),
        'activity_score': min(100, (min(posts, 1000) * 0.1) + (math.log1p(likes) * 0.5))
    }

async def send_callback(url: str, data: Dict) -> bool:
    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
            for _ in range(CALLBACK_RETRIES):
                try:
                    response = await client.post(url, json=data)
                    response.raise_for_status()
                    return True
                except Exception:
                    await asyncio.sleep(1)
    except Exception:
        pass
    return False

# Endpoints (remain the same as before)
@app.post("/request-verification", response_model=VerificationResponse)
async def request_verification(request: VerificationRequest):
    if not request.data:
        raise HTTPException(400, "At least one profile required")
    
    code = generate_verification_code(request.fayda_number)
    platform = request.data[0].social_media
    return VerificationResponse(
        verification_code=code,
        instructions=get_verification_instructions(code, platform),
        expires_in=f"{VERIFICATION_CODE_EXPIRY_MINUTES} minutes"
    )

@app.post("/verify-and-score", response_model=SocialScoreResponse)
async def verify_and_score(request: VerificationRequest, background_tasks: BackgroundTasks):
    stored = verification_storage.get(request.fayda_number)
    if not stored:
        raise HTTPException(403, "Request verification first")
    if datetime.now() > stored["expires"]:
        raise HTTPException(403, "Code expired")

    profile = next((p for p in request.data if p.social_media.lower() in ["tiktok", "facebook"]), None)
    if not profile:
        raise HTTPException(400, "Unsupported platform")

    if not await check_bio_for_code(
        profile.social_media,
        profile.username,
        request.fayda_number,
        stored["code"]
    ):
        raise HTTPException(403, "Verification failed")

    profile_data = await fetch_social_media_data(profile.social_media, profile.username)
    if not profile_data:
        raise HTTPException(503, "Could not fetch profile data")

    features = calculate_features(profile_data)
    raw_score = float(model.predict(pd.DataFrame([[
        features['profile_score'],
        features['engagement_score'],
        features['network_score'],
        features['activity_score']
    ]]))[0])

    response = SocialScoreResponse(
        fayda_number=request.fayda_number,
        socialscore=scale_score(raw_score),
        trust_level=get_trust_level(scale_score(raw_score)),
        score_breakdown={k: round(v, 2) for k, v in features.items()},
        timestamp=datetime.now().isoformat()
    )

    if request.callbackUrl:
        background_tasks.add_task(send_callback, str(request.callbackUrl), response.dict())

    return response

@app.get("/verification-status/{fayda_number}", response_model=VerificationStatus)
async def get_status(fayda_number: str):
    stored = verification_storage.get(fayda_number)
    if not stored:
        return VerificationStatus(status="not_found")
    if datetime.now() > stored["expires"]:
        return VerificationStatus(status="expired")
    return VerificationStatus(
        status="active",
        verification_code=stored["code"],
        message=f"Expires in {(stored['expires'] - datetime.now()).seconds // 60} minutes"
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)