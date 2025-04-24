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
import time
import numpy as np
from sklearn._loss._loss import CyHalfSquaredError

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


from sklearn._loss._loss import CyHalfSquaredError
import sklearn._loss._loss
import joblib
import pickle
import numpy as np
from fastapi import HTTPException
import os

# Add the missing attribute to the module if it doesn't exist
if not hasattr(sklearn._loss._loss, '__pyx_unpickle_CyHalfSquaredError'):
    sklearn._loss._loss.__pyx_unpickle_CyHalfSquaredError = lambda *args: CyHalfSquaredError

def load_and_update_model():
    try:
        # Create backup of original model file
        backup_path = MODEL_PATH + '.bak'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copyfile(MODEL_PATH, backup_path)
            logger.info("Created backup of original model file")
        
        # Custom unpickler to handle version differences
        def custom_unpickler(file):
            try:
                # First try normal loading
                unpickler = joblib.numpy_pickle.NumpyUnpickler(file)
                return unpickler.load()
            except Exception as e:
                logger.warning(f"Standard unpickling failed, trying custom solution: {str(e)}")
                file.seek(0)  # Rewind file
                
                # Create unpickler with custom persistent_load
                original_persistent_load = joblib.numpy_pickle.NumpyUnpickler(file).persistent_load
                def persistent_load(saved_id):
                    if saved_id[0] == '__pyx_unpickle_CyHalfSquaredError':
                        return CyHalfSquaredError
                    return original_persistent_load(saved_id)
                
                unpickler = joblib.numpy_pickle.NumpyUnpickler(file)
                unpickler.persistent_load = persistent_load
                return unpickler.load()
        
        # Load the model with our custom unpickler
        model = joblib.load(MODEL_PATH, unpickler=custom_unpickler)
        
        # Verify the model
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded object is not a valid model")
            
        # Test prediction
        test_input = np.array([[50, 50, 50, 50]])
        try:
            prediction = model.predict(test_input)
            logger.info(f"Test prediction successful: {prediction}")
        except Exception as e:
            logger.error(f"Test prediction failed: {str(e)}")
            raise
        
        # Immediately save the model in current version's format
        updated_model_path = MODEL_PATH + '.updated'
        joblib.dump(model, updated_model_path)
        logger.info("Saved updated model in current version's format")
        
        # Replace original with updated version
        os.replace(updated_model_path, MODEL_PATH)
        logger.info("Replaced original model with updated version")
        
        # Load scaling parameters
        scaling_params = joblib.load(SCALING_PARAMS_PATH)
        logger.info("Scaling parameters loaded successfully")
        
        return model, scaling_params
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Try to restore from backup if exists
        if os.path.exists(backup_path):
            logger.info("Attempting to restore from backup")
            try:
                os.replace(backup_path, MODEL_PATH)
            except Exception as restore_error:
                logger.error(f"Failed to restore backup: {str(restore_error)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

# Load the model and scaling parameters
try:
    model, scaling_params = load_and_update_model()
except HTTPException:
    # Fallback to simple loading if the advanced method fails
    logger.warning("Falling back to simple model loading")
    try:
        model = joblib.load(MODEL_PATH)
        scaling_params = joblib.load(SCALING_PARAMS_PATH)
        logger.info("Simple model loading succeeded")
    except Exception as e:
        logger.error(f"Fallback loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail="All model loading attempts failed")
# Models
class SocialMediaProfile(BaseModel):
    social_media: str
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
        normalized = v.replace('*', '').upper()
        if normalized != "SOCIAL_SCORE":
            raise ValueError("Type must be SOCIAL_SCORE (with or without asterisks)")
        return v

class VerificationResponse(BaseModel):
    verification_code: str
    instructions: str
    expires_in: str

class SocialScoreResponse(BaseModel):
    fayda_number: str
    type: str = "SOCIAL_SCORE"
    socialscore: int  # Changed from score with alias
    trust_level: str
    score_breakdown: Dict[str, float]
    timestamp: str

class VerificationStatus(BaseModel):
    status: str
    message: Optional[str] = None
    verification_code: Optional[str] = None

# Storage
verification_storage = {}

# Helper Functions
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
    """Fetch TikTok profile data with multiple fallback methods"""
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
        ]),
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
            # 1. Get bio from oEmbed
            oembed_url = f"https://www.tiktok.com/oembed?url=https://www.tiktok.com/@{username}"
            oembed_res = await client.get(oembed_url, headers=headers)
            if oembed_res.status_code == 200:
                profile["biography"] = oembed_res.json().get("author_name", "")

            # 2. Try mobile API for metrics
            mobile_url = f"https://m.tiktok.com/api/user/detail/?uniqueId={username}"
            mobile_res = await client.get(mobile_url, headers=headers)
            if mobile_res.status_code == 200 and mobile_res.text.strip():
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

            # 3. Fallback to HTML parsing if still missing data
            if profile["followers"] == 0:
                html_url = f"https://www.tiktok.com/@{username}"
                html_res = await client.get(html_url, headers=headers)
                if html_res.status_code == 200:
                    html = html_res.text
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
                        except json.JSONDecodeError:
                            pass

            return profile
    except Exception as e:
        logger.error(f"Error fetching TikTok data: {str(e)}")
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
    """Check if the verification code exists in the user's profile signature"""
    stored = verification_storage.get(fayda_number)
    if not stored or datetime.now() > stored["expires"]:
        return False

    if platform.lower() == "tiktok":
        # Wait for bio to potentially update
        await asyncio.sleep(15)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                # Get the HTML page with cache-busting
                url = f"https://www.tiktok.com/@{username}?_t={int(time.time())}"
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    html = response.text
                    
                    # Method 1: Direct signature extraction from HTML
                    if match := re.search(r'"signature":"([^"]+)"', html):
                        signature = match.group(1).encode('utf-8').decode('unicode_escape')
                        clean_bio = re.sub(r'[^\d]', '', signature)
                        if code in clean_bio:
                            logger.info(f"Found code in HTML signature: {signature}")
                            return True
                    
                    # Method 2: SIGI_STATE fallback
                    if match := re.search(r'<script id="SIGI_STATE"[^>]*>(.*?)</script>', html):
                        try:
                            data = json.loads(match.group(1))
                            if 'UserModule' in data:
                                for user in data['UserModule']['users'].values():
                                    if 'signature' in user:
                                        signature = user['signature']
                                        clean_bio = re.sub(r'[^\d]', '', signature)
                                        if code in clean_bio:
                                            logger.info(f"Found code in SIGI_STATE signature: {signature}")
                                            return True
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"SIGI_STATE parse error: {str(e)}")
                
                logger.warning(f"Couldn't find code in bio for {username}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking bio: {str(e)}")
            return False
            
    return False
async def _check_bio_direct(self, client, username: str, code: str) -> bool:
    """Fallback to direct HTML parsing"""
    try:
        response = await client.get(
            f"https://www.tiktok.com/@{username}",
            headers={"User-Agent": self.user_agents[0]}
        )
        if response.status_code == 200:
            if match := re.search(r'"signature":"([^"]+)"', response.text):
                bio = match.group(1).encode().decode('unicode_escape')
                return code in re.sub(r'[^\d]', '', bio)
    except Exception:
        pass
    return False
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

# API Endpoints
@app.post("/request-verification", response_model=VerificationResponse)
async def request_verification(request: VerificationRequest):
    if not request.data:
        raise HTTPException(status_code=400, detail="At least one profile required")

    # Generate and store a new verification code
    code = str(random.randint(100000, 999999))
    verification_storage[request.fayda_number] = {
        "code": code,
        "expires": datetime.now() + timedelta(minutes=VERIFICATION_CODE_EXPIRY_MINUTES),
        "attempts": 0,
        "verified": False
    }

    platform = request.data[0].social_media
    return VerificationResponse(
        verification_code=code,
        instructions=get_verification_instructions(code, platform),
        expires_in=f"{VERIFICATION_CODE_EXPIRY_MINUTES} minutes"
    )

@app.post("/verify-and-score", response_model=SocialScoreResponse)
async def verify_and_score(request: VerificationRequest, background_tasks: BackgroundTasks):
    # Fetch stored verification data
    stored["verified"] = True
    stored = verification_storage.get(request.fayda_number)
    
    # Check if verification was requested first
    if not stored:
        raise HTTPException(status_code=403, detail="Request verification first")
    
    # Check if code expired
    if datetime.now() > stored["expires"]:
        raise HTTPException(status_code=403, detail="Verification code expired")
    
    # Find the first supported platform (only TikTok now)
    profile = next((p for p in request.data if p.social_media.lower() == "tiktok"), None)
    if not profile:
        raise HTTPException(status_code=400, detail="Unsupported platform")
    
    # Verify the code in bio
    verification_passed = await check_bio_for_code(
        profile.social_media,
        profile.username,
        request.fayda_number,
        stored["code"]
    )
    
    if not verification_passed:
        raise HTTPException(status_code=403, detail="Verification failed - code not found in bio")
    
    # Fetch profile data
    profile_data = await fetch_social_media_data(profile.social_media, profile.username)
    if not profile_data:
        raise HTTPException(status_code=503, detail="Could not fetch profile data")
    
    # Calculate features
    features = calculate_features(profile_data)
    
    try:
        # Prepare input features with proper column names
        input_df = pd.DataFrame([[
            features['profile_score'],
            features['engagement_score'],
            features['network_score'],
            features['activity_score']
        ]], columns=['profile_score', 'engagement_score', 'network_score', 'activity_score'])
        
        # Make prediction (with model loaded from our new loader)
        raw_score = float(model.predict(input_df)[0])
        
        # Scale the score
        scaled_score = scale_score(raw_score)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        # Provide more detailed error message
        error_detail = {
            "error": "Score calculation failed",
            "message": str(e),
            "input_features": features
        }
        raise HTTPException(status_code=500, detail=error_detail)
    
        # Prepare response using the field name (not alias)
    response_data = {
        "fayda_number": request.fayda_number,
        "socialscore": scaled_score,  # Use the alias name here
        "trust_level": get_trust_level(scaled_score),
        "score_breakdown": {k: round(v, 2) for k, v in features.items()},
        "timestamp": datetime.now().isoformat()
    }
    
    # Create response using dict unpacking
    response = SocialScoreResponse(**response_data)

    # Send callback if provided
    if request.callbackUrl:
        background_tasks.add_task(
            send_callback, 
            str(request.callbackUrl), 
            response.dict(by_alias=True)  # Ensure field aliases are used
        )

    # Mark as verified and store the score for future reference
    stored.update({
        "verified": True,
        "score": scaled_score,
        "score_timestamp": datetime.now().isoformat()
    })
    
    return response

@app.get("/verification-status/{fayda_number}")
async def get_status(fayda_number: str):
    stored = verification_storage.get(fayda_number)
    if not stored:
        return {"status": "not_found"}
    if datetime.now() > stored["expires"]:
        return {"status": "expired"}
    return {
        "status": "active",
        "verified": stored.get("verified", False),  # THIS IS CRUCIAL
        "verification_code": stored["code"],
        "expires_in": (stored["expires"] - datetime.now()).seconds
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/debug/tiktok/{username}")
async def debug_tiktok(username: str):
    data = await fetch_tiktok_data(username)
    if not data:
        raise HTTPException(404, "Could not fetch TikTok data")
    return data

@app.get("/debug/verify-bio/{platform}/{username}/{code}")
async def debug_verify_bio(platform: str, username: str, code: str):
    result = await check_bio_for_code(platform, username, "debug", code)
    return {"match": result}

@app.get("/debug/verification-storage")
async def debug_verification_storage():
    """Endpoint to check current verification codes"""
    return verification_storage

@app.get("/debug/check-bio/{username}")
async def debug_user_bio(username: str):
    """Endpoint to check the actual TikTok bio/signature"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
        "Accept": "text/html,application/xhtml+xml",
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Get with cache busting
            url = f"https://www.tiktok.com/@{username}?_t={int(time.time())}"
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                html = response.text
                result = {}
                
                # Check direct HTML signature
                if match := re.search(r'"signature":"([^"]+)"', html):
                    signature = match.group(1).encode('utf-8').decode('unicode_escape')
                    result['html_signature'] = signature
                    result['html_signature_clean'] = re.sub(r'[^\d]', '', signature)
                
                # Check SIGI_STATE
                if match := re.search(r'<script id="SIGI_STATE"[^>]*>(.*?)</script>', html):
                    try:
                        data = json.loads(match.group(1))
                        if 'UserModule' in data:
                            for user in data['UserModule']['users'].values():
                                if 'signature' in user:
                                    result['sigi_signature'] = user['signature']
                                    result['sigi_signature_clean'] = re.sub(r'[^\d]', '', user['signature'])
                                    break
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                return result if result else {"error": "No signature found"}
            
            return {"error": f"Status {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)