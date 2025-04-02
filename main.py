import os
import json
import re
import random
import asyncio
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import httpx
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, HttpUrl, validator
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

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

# --- Configuration ---
VERIFICATION_CODE_EXPIRY_MINUTES = 15
MAX_VERIFICATION_ATTEMPTS = 5
VERIFICATION_RETRY_DELAY = 30  # seconds
DEFAULT_MIN_SCORE = 300
DEFAULT_MAX_SCORE = 850
CALLBACK_TIMEOUT = 10  # seconds
CALLBACK_RETRIES = 3
MAX_BIO_LENGTH = 500

# --- Load Model and Scaling Parameters ---
MODEL_PATH = os.getenv('MODEL_PATH', 'tiktok_scoring_model.pkl')
SCALING_PARAMS_PATH = os.getenv('SCALING_PARAMS_PATH', 'scaling_params.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaling_params = joblib.load(SCALING_PARAMS_PATH)
    logger.info("Model and scaling parameters loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

# --- Models ---
class SocialMediaProfile(BaseModel):
    name: str
    username: str

    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError("Platform name cannot be empty")
        return v.lower()

class VerificationRequest(BaseModel):
    nationalId: str
    socialMedia: List[SocialMediaProfile]
    callback: Optional[HttpUrl] = None

    @validator('nationalId')
    def national_id_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError("National ID cannot be empty")
        if len(v) < 5:
            raise ValueError("National ID too short")
        return v

class VerificationResponse(BaseModel):
    verification_code: str
    instructions: str
    expires_in: str

class SocialScoreResponse(BaseModel):
    nationalId: str
    socialscore: int
    trust_level: str
    score_breakdown: Dict[str, float]
    timestamp: str

class VerificationStatus(BaseModel):
    status: str
    message: Optional[str] = None
    verification_code: Optional[str] = None

# --- Storage with expiration ---
verification_storage = {}

# --- Helper Functions ---
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with zero handling"""
    try:
        return a / b if b != 0 else default
    except Exception:
        return default

def scale_score(raw_score: float) -> int:
    """Convert raw model output to 300-850 scale"""
    try:
        score_range = max(scaling_params['score_max'] - scaling_params['score_min'], 1e-6)
        scaled = DEFAULT_MIN_SCORE + (DEFAULT_MAX_SCORE - DEFAULT_MIN_SCORE) * (
            (raw_score - scaling_params['score_min']) / score_range
        )
        return min(DEFAULT_MAX_SCORE, max(DEFAULT_MIN_SCORE, int(round(scaled))))
    except Exception:
        logger.warning("Failed to scale score, using default minimum")
        return DEFAULT_MIN_SCORE

def get_trust_level(score: int) -> str:
    """Get trust level based on score"""
    if score >= 750: return "Very High"
    elif score >= 650: return "High"
    elif score >= 550: return "Medium"
    return "Low"

def generate_verification_code(identifier: str) -> str:
    """Generate and store code with timestamp"""
    code = str(random.randint(100000, 999999))
    verification_storage[identifier] = {
        "code": code,
        "expires": datetime.now() + timedelta(minutes=VERIFICATION_CODE_EXPIRY_MINUTES),
        "attempts": 0,
        "verified": False,
        "created_at": datetime.now()
    }
    logger.info(f"Generated code for {identifier}")
    return code

def get_verification_instructions(code: str, platform: str) -> str:
    """Generate platform-specific verification instructions"""
    platform = platform.lower()
    base_instructions = (
        f"=== {platform.upper()} VERIFICATION INSTRUCTIONS ===\n"
        f"1. Add this code to your profile bio: {code}\n"
        f"2. Ensure your account is public\n"
        f"3. Save changes\n"
        f"4. Return here to complete verification\n"
        f"NOTE: Changes may take 1-2 minutes to appear"
    )
    
    if platform == "tiktok":
        return (
            f"{base_instructions}\n"
            f"Location: Edit Profile > Bio"
        )
    elif platform == "facebook":
        return (
            f"{base_instructions}\n"
            f"Location: Edit Profile > Intro section"
        )
    return base_instructions

async def fetch_tiktok_data(username: str) -> Optional[Dict]:
    """Fetch TikTok user data with improved error handling"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.tiktok.com/",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            url = f"https://www.tiktok.com/@{username}"
            response = await client.get(url, headers=headers, follow_redirects=True)
            
            if response.status_code != 200:
                logger.error(f"TikTok fetch failed: HTTP {response.status_code}")
                return None

            html = response.text
            patterns = [
                r'"user":\s*({.+?}),\s*"stats":\s*({.+?})',
                r'"UserModule":\s*({.+?})'
            ]

            for pattern in patterns:
                match = re.search(pattern, html)
                if match:
                    try:
                        user_data = json.loads(match.group(1))
                        stats_data = json.loads(match.group(2)) if len(match.groups()) > 1 else {}
                        
                        return {
                            "account_id": user_data.get("id", f"uid_{username}"),
                            "username": username,
                            "biography": user_data.get("signature", ""),
                            "is_verified": user_data.get("verified", False),
                            "followers": stats_data.get("followerCount", 0),
                            "following": stats_data.get("followingCount", 0),
                            "likes": stats_data.get("heartCount", 0),
                            "videos_count": stats_data.get("videoCount", 0),
                            "profile_pic_url": user_data.get("avatarLarger", "")
                        }
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse TikTok data: {str(e)}")
                        continue

            logger.warning("No matching data pattern found in TikTok response")
            return None

        except httpx.RequestError as e:
            logger.error(f"TikTok request failed: {str(e)}")
            return None

async def fetch_facebook_data(username: str) -> Optional[Dict]:
    """Fetch Facebook user data (placeholder for actual implementation)"""
    logger.warning("Facebook data fetching not fully implemented")
    return {
        "username": username,
        "biography": "",
        "is_verified": False,
        "followers": 0,
        "following": 0,
        "likes": 0,
        "posts_count": 0
    }

async def fetch_social_media_data(platform: str, username: str) -> Optional[Dict]:
    """Fetch social media profile data with platform routing"""
    platform = platform.lower()
    if platform in ["tiktok", "ticktok"]:
        return await fetch_tiktok_data(username)
    elif platform == "facebook":
        return await fetch_facebook_data(username)
    logger.warning(f"Unsupported platform: {platform}")
    return None

async def check_bio_for_code(platform: str, username: str, national_id: str, expected_code: str) -> bool:
    """Verify account ownership by checking bio for code"""
    stored = verification_storage.get(national_id, {})  # ← Changed to use national_id
    
    if not stored:
        logger.warning(f"No verification data found for {national_id}")
        return False
        
    # Check attempts
    if stored.get("attempts", 0) >= MAX_VERIFICATION_ATTEMPTS:
        logger.warning(f"Max attempts reached for {national_id}")
        return False
        
    # Check expiration
    if datetime.now() > stored.get("expires", datetime.min):
        logger.warning(f"Code expired for {national_id}")
        return False

    # Rest of the function remains the same...
    profile_data = await fetch_social_media_data(platform, username)
    if not profile_data:
        logger.error(f"Failed to fetch {platform} data for {username}")
        return False

    bio = profile_data.get("biography", "") or profile_data.get("bio", "")
    cleaned_bio = "".join(c for c in bio if c.isdigit())
    
    if expected_code not in cleaned_bio:
        logger.warning(f"Code not found in bio for {username}")
        stored["attempts"] = stored.get("attempts", 0) + 1
        return False

    return True

    # Check bio for code
    bio = profile_data.get("biography", "") or profile_data.get("bio", "")
    cleaned_bio = "".join(c for c in bio if c.isdigit())
    
    if expected_code not in cleaned_bio:
        logger.warning(f"Code not found in bio for {username}")
        stored["attempts"] = stored.get("attempts", 0) + 1
        return False

    return True

def calculate_tiktok_features(profile: Dict) -> Dict:
    """Calculate TikTok-specific features with robust error handling"""
    try:
        followers = max(profile.get('followers', 0), 0)
        likes = max(profile.get('likes', 0), 0)
        videos = max(profile.get('videos_count', 0), 0)
        bio = profile.get('biography', '')
        
        # Engagement calculations
        engagement = safe_divide(likes, followers, 0)
        
        # Profile quality
        profile_score = (
            (40 if profile.get('is_verified') else 0) +
            (min(len(bio), MAX_BIO_LENGTH) * 0.06) +
            (30 if profile.get('profile_pic_url') else 0)
        )
        
        # Network strength
        network_score = (
            (math.log1p(followers) * 10 if followers > 0 else 0) +
            (safe_divide(followers, profile.get('following', 1) + 1) * 5)
        )
        
        # Activity level
        activity_score = (
            (min(videos, 1000) * 0.1) +
            (math.log1p(likes) * 0.5
        ))
        
        return {
            'profile_score': max(0, min(100, profile_score)),
            'engagement_score': max(0, min(100, engagement * 100)),
            'network_score': max(0, min(100, network_score)),
            'activity_score': max(0, min(100, activity_score))
        }
    except Exception as e:
        logger.error(f"Feature calculation failed: {str(e)}")
        return {
            'profile_score': 0,
            'engagement_score': 0,
            'network_score': 0,
            'activity_score': 0
        }

def calculate_generic_features(profile: Dict) -> Dict:
    """Calculate features for generic platforms"""
    try:
        followers = max(profile.get('followers', 0), 0)
        likes = max(profile.get('likes', 0), 0)
        posts = max(profile.get('posts_count', 0), 0)
        bio = profile.get('biography', profile.get('bio', ''))
        
        engagement = safe_divide(likes, followers, 0)
        
        return {
            'profile_score': (
                (40 if profile.get('is_verified') else 0) +
                (min(len(bio), MAX_BIO_LENGTH) * 0.06) +
                (30 if profile.get('profile_pic_url') else 0)
            ),
            'engagement_score': engagement * 100,
            'network_score': math.log1p(followers) * 10,
            'activity_score': (min(posts, 1000) * 0.1) + (math.log1p(likes) * 0.5)
        }
    except Exception as e:
        logger.error(f"Generic feature calculation failed: {str(e)}")
        return {
            'profile_score': 0,
            'engagement_score': 0,
            'network_score': 0,
            'activity_score': 0
        }

def calculate_social_score(platform: str, profile: Dict) -> Dict:
    """Calculate social score using the trained model"""
    try:
        features = (
            calculate_tiktok_features(profile) 
            if platform.lower() == "tiktok" 
            else calculate_generic_features(profile)
        )
        
        X = pd.DataFrame([[
            features['profile_score'],
            features['engagement_score'],
            features['network_score'],
            features['activity_score']
        ]], columns=['profile_score', 'engagement_score', 'network_score', 'activity_score'])
        
        raw_score = float(model.predict(X)[0])
        final_score = scale_score(raw_score)
        
        return {
            "score": final_score,
            "trust_level": get_trust_level(final_score),
            "score_breakdown": {
                "profile": round(features['profile_score'], 2),
                "engagement": round(features['engagement_score'], 2),
                "network": round(features['network_score'], 2),
                "activity": round(features['activity_score'], 2)
            }
        }
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        return {
            "score": DEFAULT_MIN_SCORE,
            "trust_level": "Low",
            "score_breakdown": {}
        }

async def send_callback_with_retry(url: str, data: Dict) -> bool:
    """Send callback with retry logic"""
    async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
        for attempt in range(CALLBACK_RETRIES):
            try:
                response = await client.post(url, json=data)
                response.raise_for_status()
                logger.info(f"Callback succeeded to {url} (attempt {attempt + 1})")
                return True
            except Exception as e:
                logger.warning(f"Callback attempt {attempt + 1} failed to {url}: {str(e)}")
                if attempt < CALLBACK_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return False

# --- API Endpoints ---
@app.post("/request-verification", response_model=VerificationResponse)
async def request_verification(request: VerificationRequest):
    """Generate and return a verification code"""
    if not request.socialMedia:
        raise HTTPException(status_code=400, detail="At least one social media profile required")
    
    # Use nationalId as primary identifier
    code = generate_verification_code(request.nationalId)
    platform = request.socialMedia[0].name
    instructions = get_verification_instructions(code, platform)
    
    return VerificationResponse(
        verification_code=code,
        instructions=instructions,
        expires_in=f"{VERIFICATION_CODE_EXPIRY_MINUTES} minutes"
    )

@app.post("/verify-and-score", response_model=SocialScoreResponse)
async def verify_and_score(
    request: VerificationRequest,
    background_tasks: BackgroundTasks
):
    """Verify account and calculate social score"""
    # Get stored verification data using nationalId
    stored = verification_storage.get(request.nationalId)
    if not stored:
        logger.error(f"No verification found for nationalId: {request.nationalId}")
        raise HTTPException(
            status_code=403, 
            detail="Please request verification first"
        )
    
    # Check expiration
    if datetime.now() > stored["expires"]:
        logger.warning(f"Code expired for nationalId: {request.nationalId}")
        raise HTTPException(
            status_code=403,
            detail="Verification code expired. Please request a new one."
        )
    
    # Find first supported platform
    supported_platforms = ["tiktok", "facebook"]
    profile = next(
        (p for p in request.socialMedia if p.name.lower() in supported_platforms),
        None
    )
    
    if not profile:
        logger.error(f"Unsupported platforms in request: {[p.name for p in request.socialMedia]}")
        raise HTTPException(
            status_code=400,
            detail=f"Supported platforms: {', '.join(supported_platforms)}"
        )
    
    # Verify ownership - CRITICAL CHANGE HERE
    if not await check_bio_for_code(
        platform=profile.name,
        username=profile.username,
        national_id=request.nationalId,  # Pass nationalId to lookup
        expected_code=stored["code"]
    ):
        attempts_left = MAX_VERIFICATION_ATTEMPTS - stored.get("attempts", 0)
        logger.warning(
            f"Verification failed for {profile.username}. "
            f"Attempts left: {attempts_left}"
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"{profile.name} verification failed. "
                f"Attempts left: {attempts_left}. "
                "Ensure: 1) Code is in bio 2) Account is public "
                "3) Changes are saved"
            )
        )
    
    # Mark as verified
    stored["verified"] = True
    logger.info(f"Successfully verified {profile.username}")

    # Get profile data
    profile_data = await fetch_social_media_data(profile.name, profile.username)
    if not profile_data:
        logger.error(f"Failed to fetch profile data for {profile.username}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch {profile.name} profile data. Try again later."
        )
    
    # Calculate score
    try:
        score_result = calculate_social_score(profile.name, profile_data)
        logger.info(f"Calculated score: {score_result['score']} for {request.nationalId}")
    except Exception as e:
        logger.error(f"Scoring failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Scoring service temporarily unavailable"
        )
    
    # Prepare response
    response = SocialScoreResponse(
        nationalId=request.nationalId,
        socialscore=score_result["score"],
        trust_level=score_result["trust_level"],
        score_breakdown=score_result["score_breakdown"],
        timestamp=datetime.now().isoformat()
    )
    
    # Async callback if provided
    if request.callback:
        logger.info(f"Queueing callback to {request.callback}")
        background_tasks.add_task(
            send_callback_with_retry,
            str(request.callback),
            response.dict()
        )
    
    return response

@app.get("/verification-status/{national_id}", response_model=VerificationStatus)
async def check_verification_status(national_id: str):
    """Check verification status"""
    stored = verification_storage.get(national_id)
    if not stored:
        return VerificationStatus(status="not_found", message="No verification request found")
    
    if datetime.now() > stored["expires"]:
        return VerificationStatus(status="expired", message="Verification code has expired")
    
    remaining = (stored["expires"] - datetime.now()).seconds // 60
    return VerificationStatus(
        status="active",
        message=f"Verification code expires in {remaining} minutes",
        verification_code=stored["code"]
    )

@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "active_verifications": len(verification_storage),
        "version": "1.0.0"
    }

# --- Background Tasks ---
async def cleanup_expired_codes():
    """Periodically clean up expired verification codes"""
    while True:
        try:
            now = datetime.now()
            expired = [
                id for id, data in verification_storage.items()
                if data["expires"] < now
            ]
            for id in expired:
                del verification_storage[id]
                logger.info(f"Cleaned up expired code for {id}")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
        await asyncio.sleep(3600)  # Run hourly

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(cleanup_expired_codes())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=None,
        access_log=False
    )