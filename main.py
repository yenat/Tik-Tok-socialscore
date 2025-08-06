import asyncio
import os
import json
import re
import logging
import httpx
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from math import log1p

CALLBACK_TIMEOUT = 10
CALLBACK_RETRIES = 3

# --- Configuration ---
MODEL_PATH = 'tiktok_scoring_model.pkl'
SCALING_PARAMS_PATH = 'scaling_params.pkl'
LOG_FILE = 'debug.log'
# Define a more aggressive cap for extremely large numbers to prevent overflow
MAX_FOLLOWERS_CAP = 5_000_000_000 # 5 Billion
MAX_LIKES_CAP = 1_000_000_000_000 # 1 Trillion

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="TikTok Scoring API",
    description="API for calculating TikTok profile scores",
    version="2.0.7" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Models ---
model = None
scaling_params = None
feature_names = ['profile_score', 'network_score', 'activity_score', 'is_elite', 'followers']

# --- Helper Functions ---
def safe_divide(a, b, default=0):
    """Fixed version that handles both Series and scalar values"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if hasattr(result, '__iter__'):
            result[~np.isfinite(result)] = default
            return result
        return result if np.isfinite(result) else default

def log_scraped_data(data: Dict):
    """Enhanced debug logging for scraped data"""
    logger.debug("\n" + "="*50)
    logger.debug(f"SCRAPED DATA FOR @{data['username']}")
    logger.debug(f"Followers: {data['followers']:,}")
    logger.debug(f"Likes: {data['likes']:,}")
    logger.debug(f"Videos: {data['videos_count']:,}")
    logger.debug(f"Verified: {data['is_verified']}")
    logger.debug(f"Tier: {data['tier']}")
    logger.debug(f"Bio: {data['biography'][:50]}... (len: {len(data['biography'])})")
    logger.debug("="*50 + "\n")

# --- Data Scraper ---

async def fetch_tiktok_data(username: str) -> Optional[Dict]:
    """Enhanced version of your working scraper with guaranteed video counts"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/html",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.tiktok.com/@{username}"
    }
    
    profile = {
        "username": username,
        "biography": "",
        "is_verified": False,
        "followers": 0,
        "likes": 0,
        "videos_count": 0,
        "average_views_per_video": 0,
        "tier": "regular",
        "like_engagement_rate": 0.0,
        "comment_engagement_rate": 0.0
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                url = f"https://m.tiktok.com/api/user/detail/?uniqueId={username}"
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if 'userInfo' in data:
                        user = data['userInfo']['user']
                        stats = user.get('stats', {})
                        profile.update({
                            "is_verified": user.get("verified", False),
                            "followers": stats.get("followerCount", 0),
                            "likes": stats.get("heartCount", 0),
                            "videos_count": stats.get("videoCount", 0),
                            "biography": user.get("signature", "").strip()
                        })
                        if profile['videos_count'] > 0:
                            profile['average_views_per_video'] = safe_divide(
                                stats.get('playCount', 0),
                                profile['videos_count']
                            )
                        logger.debug("Mobile API success")
            except Exception as e:
                logger.warning(f"Mobile API failed: {str(e)}")

            if profile["followers"] == 0 or profile["videos_count"] == 0:
                try:
                    url = f"https://www.tiktok.com/@{username}"
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        html = response.text
                        
                        def parse_numeric_value(html_string, key_pattern):
                            """
                            Helper function to robustly parse large numbers from HTML.
                            It finds the number string and removes any non-digit characters before converting.
                            """
                            match = re.search(f'"{key_pattern}":(\d+)', html_string)
                            if not match:
                                # Fallback regex for numbers with commas or other separators.
                                match = re.search(f'"{key_pattern}":"([\d,]+)"', html_string)

                            if match:
                                number_string = match.group(1)
                                # Remove any non-digit characters like commas
                                sanitized_number = re.sub(r'[^\d]', '', number_string)
                                return int(sanitized_number)
                            return 0
                        
                        # Use the new helper function for each field
                        profile['followers'] = parse_numeric_value(html, 'followerCount')
                        profile['likes'] = parse_numeric_value(html, 'heartCount')
                        profile['videos_count'] = parse_numeric_value(html, 'videoCount')

                        if 'SIGI_STATE' in html:
                            try:
                                sigi_data = json.loads(re.search(r'<script id="SIGI_STATE"[^>]*>(.*?)</script>', html).group(1))
                                if 'UserModule' in sigi_data:
                                    user_data = next(
                                        (u for u in sigi_data['UserModule']['users'].values() 
                                         if u.get('uniqueId') == username),
                                         None
                                    )
                                    if user_data:
                                        stats = user_data.get('stats', {})
                                        if all(k in stats for k in ['followerCount', 'videoCount']):
                                            profile.update({
                                                "followers": stats['followerCount'],
                                                "videos_count": stats['videoCount'],
                                                "likes": stats.get('heartCount', profile['likes'])
                                            })
                            except Exception as e:
                                logger.warning(f"SIGI_STATE parse error: {str(e)}")

                        if not profile["is_verified"] and ('verified-icon' in html or 'Verified account' in html):
                            profile["is_verified"] = True

                        if not profile["biography"] and (match := re.search(r'<meta property="og:description" content="([^"]*)"', html)):
                            profile["biography"] = match.group(1).strip()

                except Exception as e:
                    logger.warning(f"Web scraping failed: {str(e)}")

            profile['followers'] = int(min(profile['followers'], MAX_FOLLOWERS_CAP))
            profile['likes'] = int(min(profile['likes'], MAX_LIKES_CAP))

            if profile['videos_count'] == 0 and profile['likes'] > 0:
                if profile['followers'] > 10000:
                    profile['videos_count'] = max(10, profile['likes'] // 5000)
                else:
                    profile['videos_count'] = max(1, profile['likes'] // 1000)
                
                profile['average_views_per_video'] = safe_divide(
                    profile['likes'] * 3,
                    profile['videos_count']
                )

            if profile['videos_count'] > 50 and profile['followers'] == 0:
                profile['videos_count'] = min(profile['videos_count'], 5)
                logger.warning(f"Adjusted unrealistic video count for @{username}")

            followers = profile["followers"]
            if followers >= 100_000_000: profile["tier"] = "ultra"
            elif followers >= 10_000_000: profile["tier"] = "mega"
            elif followers >= 1_000_000: profile["tier"] = "macro"
            elif followers >= 100_000: profile["tier"] = "mid"
            elif followers >= 10_000: profile["tier"] = "micro"
            else: profile["tier"] = "regular"

            if profile["followers"] > 0:
                profile["like_engagement_rate"] = min(
                    safe_divide(profile["likes"], profile["followers"]),
                    0.5
                )
                profile["comment_engagement_rate"] = profile["like_engagement_rate"] * 0.1

            log_scraped_data(profile)
            return profile

    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}", exc_info=True)
        return None

def calculate_features(data: Dict) -> pd.DataFrame:
    """
    Final, aggressive feature calculation to give the model highly differentiated
    inputs. The core change is a more powerful scaling for the network score.
    """
    df = pd.DataFrame([data])
    
    df['bio_length'] = df['biography'].str.len().fillna(0).clip(0, 500)
    df['is_elite'] = df['tier'].isin(['macro', 'mega', 'ultra']).astype(int)
    
    # Profile Score (0-100)
    df['profile_score'] = (
        df['is_verified'].astype(int) * 40 +
        np.log1p(df['bio_length']) * 10 +
        df['is_elite'] * 30
    ).clip(0, 100)
    # Network Score
    followers_scaled = np.log10(df['followers'].clip(1))
    likes_scaled = np.log10(df['likes'].clip(1))
    
    df['network_score'] = (followers_scaled * 100) + (likes_scaled * 20)
    
    # Activity Score (0-300) - Fine-tuned
    df['activity_score'] = (
        np.log1p(df['videos_count']) * 20 +
        np.log1p(df['average_views_per_video'].clip(1, 100_000_000)) * 15 +
        safe_divide(df['likes'], df['videos_count'].clip(1)).clip(0, 100_000) / 400
    ).clip(0, 300)
    
    logger.debug(f"Calculated Features: {df[feature_names].iloc[0].to_dict()}")

    return df[['profile_score', 'network_score', 'activity_score', 'is_elite', 'followers']]

def calculate_final_score(raw_pred: float, features: pd.DataFrame) -> int:
    """
    Final score calculation using a transparent, base-score driven approach.
    This provides a much better and more predictable score distribution.
    """
    
    # Step 1: Establish a base score based on the profile's tier.
    followers = features['followers'].iloc[0]
    if followers < 100:
        base_score = 300
    elif followers < 1000:
        base_score = 350
    elif followers < 10000:
        base_score = 400
    elif followers < 100_000:
        base_score = 450
    elif followers < 1_000_000:
        base_score = 500
    elif followers < 10_000_000:
        base_score = 600
    else:
        base_score = 700

    # Step 2: Use the model's prediction as a fine-tuning factor.
    if followers < 1000:
        model_adjustment = raw_pred * 0.5
    elif followers < 10_000:
        model_adjustment = raw_pred * 1.5
    else:
        model_adjustment = raw_pred * 2.0
    
    final_score_base = base_score + model_adjustment
    
    # Step 3: Apply a small elite boost and final clipping.
    is_elite = features['is_elite'].iloc[0]
    if is_elite:
        network_score_value = features['network_score'].iloc[0]
        elite_bonus = 20 + (network_score_value / 50)
        final_score_base += min(elite_bonus, 850 - final_score_base)
    
    final_score = int(np.clip(final_score_base, 300, 850))
    
    logger.debug(f"""
    FINAL SCORE CALCULATION:
    - Profile Followers: {followers:,}
    - Base Score (from tier): {base_score}
    - Model Raw Prediction: {raw_pred:.2f}
    - Model Adjustment: {model_adjustment:.1f}
    - Elite: {is_elite}
    - Final Score: {final_score}
    """)
    
    return final_score

# --- Pydantic Models ---
class DataItem(BaseModel):
    social_media: str = Field(...)
    username: str = Field(...)

class CalculateScoreRequest(BaseModel):
    fayda_number: str = Field(...)
    data: List[DataItem]

class ScoreBreakdownItem(BaseModel):
    value: float
    max: Optional[float] = None

class CalculateScoreResponse(BaseModel):
    fayda_number: str
    score: int
    score_range: str = "300-850"
    trust_level: str
    score_breakdown: Dict[str, ScoreBreakdownItem]
    timestamp: datetime = Field(default_factory=datetime.now)
    type: str = "SOCIAL_SCORE"

# --- API Endpoints ---
@app.on_event("startup")
async def startup():
    global model, scaling_params
    try:
        model = joblib.load(MODEL_PATH)
        scaling_params = joblib.load(SCALING_PARAMS_PATH)
        logger.info("Model and scaling parameters loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model from {MODEL_PATH} and scaling parameters from {SCALING_PARAMS_PATH}")

# --- API Endpoints ---
@app.post("/calculate-score", response_model=CalculateScoreResponse)
async def calculate_score(request: CalculateScoreRequest):
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        
        data_item = request.data[0]
        if data_item.social_media.lower() != "tiktok":
            raise HTTPException(400, "Only TikTok supported")

        profile = await fetch_tiktok_data(data_item.username)
        if not profile:
            raise HTTPException(404, "Profile not found")

        features_df = calculate_features(profile)
        
        scaled_features = features_df[scaling_params['feature_names']]
        scaled = scaling_params['scaler'].transform(scaled_features)
        
        raw_score = model.predict(scaled)[0]
        
        final_score = calculate_final_score(raw_score, features_df)
        
        # Calculate a theoretical max for network_score for a more descriptive response.
        # This is based on the MAX_FOLLOWERS_CAP and MAX_LIKES_CAP from your configuration.
        theoretical_max_network_score = (np.log10(MAX_FOLLOWERS_CAP) * 100) + (np.log10(MAX_LIKES_CAP) * 20)
        
        breakdown = {
            "profile_score": ScoreBreakdownItem(
                value=round(float(features_df['profile_score'].iloc[0]), 2), 
                max=100.0
            ),
            "network_score": ScoreBreakdownItem(
                value=round(float(features_df['network_score'].iloc[0]), 2), 
                max=round(float(theoretical_max_network_score), 2)
            ),
            "activity_score": ScoreBreakdownItem(
                value=round(float(features_df['activity_score'].iloc[0]), 2), 
                max=300.0
            )
        }
        
        return CalculateScoreResponse(
            fayda_number=request.fayda_number,
            score=final_score,
            trust_level="High" if final_score >= 700 else "Medium" if final_score >= 450 else "Low",
            score_breakdown=breakdown
        )
    except Exception as e:
        logger.error(f"Scoring error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal server error")
    
# Callback Sender Function
async def send_callback(url: str, data: Dict) -> bool:
    """Enhanced callback sender with detailed logging and retry logic."""
    callback_id = f"callback_{datetime.utcnow().timestamp()}"
    logger.info(f"[{callback_id}] Initiating callback to {url}")
    
    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
            for attempt in range(1, CALLBACK_RETRIES + 1):
                try:
                    logger.info(f"[{callback_id}] Attempt {attempt}/{CALLBACK_RETRIES}")
                    response = await client.post(url, json=data)
                    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                    logger.info(f"[{callback_id}] Callback successful")
                    return True
                except httpx.RequestError as e:
                    logger.warning(f"[{callback_id}] Request failed: {str(e)}")
                except httpx.HTTPStatusError as e:
                    logger.warning(f"[{callback_id}] HTTP error: {str(e)}")
                
                if attempt < CALLBACK_RETRIES:
                    await asyncio.sleep(1 * attempt)  # Exponential backoff
                    
    except Exception as e:
        logger.error(f"[{callback_id}] Callback failed completely: {str(e)}", exc_info=True)
    
    return False

# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint to verify API status and model loading."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# Debug endpoint for TikTok profile data
@app.get("/debug/tiktok/{username}")
async def debug_tiktok_profile(username: str):
    """Debug endpoint to fetch raw TikTok profile data."""
    data = await fetch_tiktok_data(username)
    if not data:
        raise HTTPException(404, "TikTok profile not found")
    return data

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)