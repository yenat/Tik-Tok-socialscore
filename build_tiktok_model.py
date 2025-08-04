import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_divide(a, b, default=0):
    """Safe division with zero handling"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = default
    return result

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath)
    
    # Constants
    MAX_FOLLOWERS = 200_000_000
    MAX_VIDEOS = 20_000
    MAX_BIO = 500
    MAX_AVG_VIEWS = 1_000_000_000 

    # Feature Engineering
    df['is_elite'] = ((df['followers'] > 10_000_000) | 
                     (df['tier'].isin(['mega', 'ultra']))).astype(int)
    df['biography'] = df['biography'].fillna('')
    
    # Profile features (20% weight)
    df['profile_score'] = (
        df['is_verified'].astype(int) * 40 +
        df['biography'].str.len().clip(upper=MAX_BIO) * 0.06 +
        df['is_elite'] * 30
    )
    
    # Network features (40% weight)
    df['log_followers'] = np.log1p(df['followers'].clip(0, MAX_FOLLOWERS))
    df['network_score'] = (
        df['log_followers'] * 15 +
        np.log1p(df['likes']) * 2 +
        safe_divide(df['likes'], np.sqrt(df['followers'] + 1), default=0.0) * 0.1
    )
    
    # Activity features (40% weight)
    df['content_velocity'] = safe_divide(
        df['videos_count'].clip(0, MAX_VIDEOS),
        np.log1p(df['followers'] + 1000)
    )
    likes_multiplier = (df['videos_count'] > 0).astype(float)
    df['activity_score'] = (
        np.log1p(df['videos_count']) * 20 +
        np.log1p(df['likes']) * 10 * likes_multiplier +
        np.log1p(df['average_views_per_video'].clip(lower=1, upper=MAX_AVG_VIEWS)) * 8 + 
        df['content_velocity'] * 5
    )
    
    # --- TARGET VARIABLE WITH CORRESPONDING WEIGHTS---
    df['trustworthiness_raw'] = (
        0.2 * np.power(df['profile_score'], 0.9) +   # 20% weight
        0.4 * np.power(df['network_score'], 0.7) +   # 40% weight
        0.4 * np.power(df['activity_score'], 0.8)    # 40% weight
    )
    
    # Ensure trustworthiness_raw is not negative
    df['trustworthiness_raw'] = df['trustworthiness_raw'].clip(lower=0)

    # Elite boost (applied after component combination)
    df.loc[df['is_elite'] == 1, 'trustworthiness_raw'] = df['trustworthiness_raw'] * 1.8
    
    # Clip the raw trustworthiness
    df['trustworthiness_raw'] = df['trustworthiness_raw'].clip(lower=0, upper=10000)

    logger.info(f"Trustworthiness Raw - Min: {df['trustworthiness_raw'].min():.2f}")
    logger.info(f"Trustworthiness Raw - Max: {df['trustworthiness_raw'].max():.2f}")
    logger.info(f"Trustworthiness Raw - Mean: {df['trustworthiness_raw'].mean():.2f}")
    logger.info(f"Trustworthiness Raw - 99th Percentile: {df['trustworthiness_raw'].quantile(0.99):.2f}")

    plt.figure(figsize=(8, 6))
    sns.histplot(df['trustworthiness_raw'], bins=50, kde=True)
    plt.title('Distribution of trustworthiness_raw (Target Variable)')
    plt.savefig('trustworthiness_raw_distribution.png')
    plt.close()

    return df

def train_model(X, y):
    """Train and evaluate the model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    pipeline = make_pipeline(
        RobustScaler(), 
        GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_leaf=15,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
    )
    
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                               cv=5, scoring='neg_root_mean_squared_error')
    logger.info(f"Cross-validated RMSE: {-cv_scores.mean():.2f} (±{-cv_scores.std():.2f})")
    
    pipeline.fit(X_train, y_train)
    
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)
    
    logger.info("\n=== Model Performance ===")
    logger.info(f"Train RMSE: {root_mean_squared_error(y_train, train_preds):.2f}")
    logger.info(f"Test RMSE: {root_mean_squared_error(y_test, test_preds):.2f}")
    logger.info(f"Test R²: {r2_score(y_test, test_preds):.3f}")
    
    return pipeline.named_steps['gradientboostingregressor'], pipeline.named_steps['robustscaler'], X_test, y_test, train_preds, test_preds

def scale_scores(preds, elite_flags, min_raw_calc, max_raw_calc):
    """Scale scores to 300-850 range"""
    scaled = np.zeros_like(preds, dtype=float)
    if abs(max_raw_calc - min_raw_calc) < 1e-6:
        scaled.fill((300 + 850) / 2) if min_raw_calc > 0 else scaled.fill(300)
    else:
        clamped_preds = np.clip(preds, min_raw_calc, max_raw_calc)
        normalized_preds = (clamped_preds - min_raw_calc) / (max_raw_calc - min_raw_calc)
        normalized_preds_transformed = normalized_preds**0.95 
        scaled = 300 + 550 * normalized_preds_transformed
    return np.clip(scaled, 300, 850).round().astype(int)

def visualize_results(df, test_idx, scores, all_raw_preds):
    """Visualize score distributions"""
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df.loc[test_idx], x='tier', y='social_score',
                order=['regular', 'micro', 'mid', 'macro', 'mega', 'ultra'])
    plt.title("Final Social Score Distribution by Tier")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('score_distributions.png')
    plt.close()

def main():
    df = load_and_preprocess_data('augmented_tiktok_profiles_100k.csv')
    
    # Updated features (removed engagement_score)
    features = ['profile_score', 'network_score', 'activity_score', 'is_elite']
    X = df[features]
    y = df['trustworthiness_raw']
    
    model, scaler, X_test, y_test, train_preds, test_preds = train_model(X, y)
    
    all_raw_preds = np.concatenate((train_preds, test_preds))
    global_min_pred = np.percentile(all_raw_preds, 1)
    global_max_pred = np.percentile(all_raw_preds, 99)
    if global_max_pred == global_min_pred:
        global_max_pred += 0.01

    test_scaled_scores = scale_scores(test_preds, X_test['is_elite'], global_min_pred, global_max_pred)
    df.loc[X_test.index, 'social_score'] = test_scaled_scores
    
    visualize_results(df, X_test.index, test_scaled_scores, all_raw_preds)
    
    # Save artifacts
    joblib.dump(model, 'tiktok_scoring_model.pkl')
    joblib.dump({
        'raw_score_min': float(global_min_pred),
        'raw_score_max': float(global_max_pred),
        'scaler': scaler,
        'feature_names': features 
    }, 'scaling_params.pkl')

    df[['nickname', 'tier', 'social_score']].to_csv('tiktok_credit_scores.csv', index=False)
    logger.info("Model and scores saved successfully!")

if __name__ == "__main__":
    main()