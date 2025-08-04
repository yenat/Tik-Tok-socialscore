import pandas as pd
import numpy as np
import random
from faker import Faker
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Initialize Faker for realistic-looking nicknames and biographies
fake = Faker()

# Define comprehensive tier configurations
# Adjusted weights to ensure more "elite" examples, and refined engagement ranges
TIER_CONFIG = {
    'ultra': {
        'weight': 0.015, # Increased from 0.01 for more elite examples
        'followers_range': (100_000_001, 300_000_000), # Up to 300M followers
        'min_likes_per_follower_ratio': 0.005, # Higher floor for engagement (0.5%)
        'max_likes_per_follower_ratio': 0.015, # Tighter, higher range (1.5%)
        'videos_range': (1500, 25000), # Adjusted videos count
        'verified_prob': 1.0, # Always verified
        'avg_views_multiplier': 0.6, # Average views are 60% of followers for ultra
        'avg_comment_to_like_ratio': 0.07, # Slightly higher comment ratio
        'description': 'top global celebrities'
    },
    'mega': {
        'weight': 0.04, # Increased from 0.03 for more elite examples
        'followers_range': (10_000_001, 100_000_000), # Up to 100M followers
        'min_likes_per_follower_ratio': 0.008, # Higher floor for engagement (0.8%)
        'max_likes_per_follower_ratio': 0.02, # Tighter, higher range (2%)
        'videos_range': (700, 15000), # Adjusted
        'verified_prob': 0.98, # Very high probability of being verified
        'avg_views_multiplier': 0.9, # Average views are 90% of followers for mega
        'avg_comment_to_like_ratio': 0.12, # Slightly higher comment ratio
        'description': 'internet stars'
    },
    'macro': {
        'weight': 0.1,
        'followers_range': (1_000_001, 10_000_000), # 1M to 10M
        'min_likes_per_follower_ratio': 0.01,
        'max_likes_per_follower_ratio': 0.03,
        'videos_range': (300, 10000),
        'verified_prob': 0.7,
        'avg_views_multiplier': 1.0, # Views roughly equal to followers for macro
        'avg_comment_to_like_ratio': 0.15,
        'description': 'well-known creators'
    },
    'mid': {
        'weight': 0.2,
        'followers_range': (100_001, 1_000_000), # 100K to 1M
        'min_likes_per_follower_ratio': 0.02,
        'max_likes_per_follower_ratio': 0.05,
        'videos_range': (100, 5000),
        'verified_prob': 0.3,
        'avg_views_multiplier': 1.1, # Views slightly higher than followers for mid
        'avg_comment_to_like_ratio': 0.18,
        'description': 'rising stars'
    },
    'micro': {
        'weight': 0.25,
        'followers_range': (10_001, 100_000), # 10K to 100K
        'min_likes_per_follower_ratio': 0.03,
        'max_likes_per_follower_ratio': 0.08,
        'videos_range': (50, 2000),
        'verified_prob': 0.05,
        'avg_views_multiplier': 1.2, # Views higher than followers for micro
        'avg_comment_to_like_ratio': 0.2,
        'description': 'small creators'
    },
    'nano': {
        'weight': 0.4, # Highest weight for smallest accounts
        'followers_range': (1, 10_000), # 1 to 10K
        'min_likes_per_follower_ratio': 0.05,
        'max_likes_per_follower_ratio': 0.1,
        'videos_range': (1, 500),
        'verified_prob': 0.001, # Very low probability
        'avg_views_multiplier': 1.3, # Views significantly higher than followers for nano
        'avg_comment_to_like_ratio': 0.25,
        'description': 'new or inactive users'
    }
}

# List of example regions for diversity
REGIONS = ['US', 'BR', 'IN', 'ID', 'GB', 'DE', 'FR', 'AU', 'CA', 'MX', 'PH', 'VN', 'TR', 'RU', 'JP', 'KR', 'SA', 'AE', 'ZA', 'EG', 'AR', 'CL', 'CO', 'PE', 'MY', 'TH', 'PK', 'BD', 'NG', 'KE', 'TZ', 'UG', 'GH', 'CI', 'SN', 'CM', 'AO', 'MZ', 'SD', 'MA', 'DZ', 'LY', 'ET', 'MG', 'ML', 'BF', 'NE', 'RW', 'SO', 'ZM', 'ZW', 'MW', 'BJ', 'TG', 'SL', 'LR', 'GW', 'GA', 'CG', 'CD', 'CF', 'GQ', 'GM', 'BI', 'DJ', 'ER', 'KM', 'RE', 'SC', 'MU', 'CV', 'ST', 'TD', 'MR', 'DJ', 'SS', 'ZW']

def assign_tier_and_augment(num_profiles: int) -> list[dict]:
    """
    Generates a list of dictionaries, each representing a TikTok profile.
    Profiles are assigned a tier based on weights, and data is augmented accordingly.
    """
    profiles = []
    tiers = list(TIER_CONFIG.keys())
    weights = [TIER_CONFIG[tier]['weight'] for tier in tiers]

    for _ in range(num_profiles):
        # Randomly select a tier based on defined weights
        tier = random.choices(tiers, weights=weights, k=1)[0]
        config = TIER_CONFIG[tier]

        # Generate core profile attributes based on tier configuration
        followers = random.randint(*config['followers_range'])
        videos_count = random.randint(*config['videos_range'])
        is_verified = random.random() < config['verified_prob']
        region = random.choice(REGIONS)

        # Calculate likes and engagement based on followers and ratios
        likes_per_follower_ratio = random.uniform(config['min_likes_per_follower_ratio'], config['max_likes_per_follower_ratio'])
        likes = int(followers * likes_per_follower_ratio)

        # Simulate average views per video
        # Clamped at a reasonable max to avoid unrealistic outliers
        avg_views_per_video = int(followers * config['avg_views_multiplier'] * random.uniform(0.8, 1.2)) # Add some variability
        if videos_count > 0:
            avg_views_per_video = max(1, avg_views_per_video) # Ensure at least 1 view if videos exist

        # Calculate engagement rates (likes and comments)
        like_engagement_rate = safe_divide(likes, followers, default=0.0)
        comment_engagement_rate = like_engagement_rate * random.uniform(config['avg_comment_to_like_ratio'] * 0.8, config['avg_comment_to_like_ratio'] * 1.2) # Add variability

        # Generate nickname and biography using Faker
        nickname = fake.user_name()
        biography = fake.text(max_nb_chars=random.randint(50, 300)) # Varied bio length

        profile = {
            'nickname': nickname,
            'biography': biography,
            'followers': followers,
            'likes': likes,
            'videos_count': videos_count,
            'is_verified': is_verified,
            'awg_engagement_rate': (like_engagement_rate + comment_engagement_rate) / 2, # Average of likes and comments
            'comment_engagement_rate': comment_engagement_rate,
            'like_engagement_rate': like_engagement_rate,
            'region': region,
            'tier': tier,
            'tier_description': config['description'],
            'average_views_per_video': avg_views_per_video
        }
        profiles.append(profile)
    return profiles

def safe_divide(numerator, denominator, default=0):
    """Safely divides two numbers, handling division by zero."""
    return numerator / denominator if denominator != 0 else default

def generate_augmented_data(num_profiles: int = 100_000, output_filename: str = 'augmented_tiktok_profiles_100k.csv'):
    """
    Generates a specified number of augmented TikTok profiles and saves them to a CSV.
    """
    print(f"Generating {num_profiles} augmented TikTok profiles...")
    profiles_data = assign_tier_and_augment(num_profiles)
    df = pd.DataFrame(profiles_data)

    # Basic data validation and cleaning
    df.fillna(0, inplace=True) # Fill any potential NaNs from calculations

    # Ensure numerical columns are of correct type
    numerical_cols = ['followers', 'likes', 'videos_count', 'awg_engagement_rate',
                      'comment_engagement_rate', 'like_engagement_rate', 'average_views_per_video']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Generated data saved to {output_filename}")
    return df

def plot_tier_distribution(df: pd.DataFrame):
    """Plots the distribution of profiles across different tiers."""
    plt.figure(figsize=(10, 6))
    sns.countplot(y='tier', data=df, order=[tier for tier, _ in sorted(TIER_CONFIG.items(), key=lambda item: item[1]['followers_range'][0], reverse=True)])
    plt.title('Distribution of Profiles Across Tiers')
    plt.xlabel('Number of Profiles')
    plt.ylabel('Tier')
    plt.tight_layout()
    plt.savefig('tier_distribution.png')
    print("Tier distribution plot saved as tier_distribution.png")

def validate_data_quality(df: pd.DataFrame):
    """Performs basic data quality checks."""
    print("\n--- Data Quality Report ---")
    print(f"Total profiles generated: {len(df)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDescriptive statistics for numerical columns:")
    print(df[['followers', 'likes', 'videos_count', 'like_engagement_rate', 'average_views_per_video']].describe())
    print("\nVerification status distribution:")
    print(df['is_verified'].value_counts(normalize=True))
    print("\n--- End Data Quality Report ---")

def main():
    """Main function to generate and validate augmented data."""
    num_profiles = 100_000 # You can adjust this number
    output_file = 'augmented_tiktok_profiles_100k.csv'

    # Check if the file already exists, if so, ask to regenerate or use existing
    if os.path.exists(output_file):
        choice = input(f"'{output_file}' already exists. Do you want to (r)egenerate or (u)se existing? (r/u): ").lower()
        if choice == 'r':
            df_augmented = generate_augmented_data(num_profiles, output_file)
        elif choice == 'u':
            print(f"Using existing data from {output_file}.")
            df_augmented = pd.read_csv(output_file)
        else:
            print("Invalid choice. Exiting.")
            return
    else:
        df_augmented = generate_augmented_data(num_profiles, output_file)
    
    if df_augmented is not None:
        plot_tier_distribution(df_augmented)
        validate_data_quality(df_augmented)
        print("\nData augmentation process completed successfully.")

if __name__ == "__main__":
    main()