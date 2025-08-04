import pandas as pd
import numpy as np

def extract_selected_columns(input_csv_file: str, output_csv_file: str):
    desired_columns = [
        'biography',
        'is_verified',
        'followers',
        'following',
        'likes',
        'videos_count',
        'awg_engagement_rate',
        'comment_engagement_rate',
        'like_engagement_rate',
        'profile_pic_url_hd',
        'nickname',
        'region'
    ]

    try:
        df = pd.read_csv(input_csv_file)
        print(f"Loaded '{input_csv_file}' with columns: {df.columns.tolist()}")

        missing_columns = [col for col in desired_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        desired_columns = [col for col in desired_columns if col in df.columns]

        df_selected = df[desired_columns]

        for col in desired_columns:
            if col not in df_selected.columns:
                df_selected[col] = np.nan

        df_selected.to_csv(output_csv_file, index=False)
        print(f"Saved selected columns to '{output_csv_file}'")

    except Exception as e:
        print(f"Error in extraction: {e}")


def combine_tiktok_data(selected_features_path: str, ethiopian_users_path: str, output_path: str):
    try:
        df_selected = pd.read_csv(selected_features_path)
        df_ethiopian = pd.read_csv(ethiopian_users_path)

        all_columns = list(set(df_selected.columns) | set(df_ethiopian.columns))
        df_selected = df_selected.reindex(columns=all_columns, fill_value=np.nan)
        df_ethiopian = df_ethiopian.reindex(columns=all_columns, fill_value=np.nan)

        df_combined = pd.concat([df_selected, df_ethiopian], ignore_index=True)
        print(f"Combined dataset has {len(df_combined)} records.")

        # Remove unwanted columns if they slipped in
        for col in ['like_count', 'account_id', 'username']:
            if col in df_combined.columns:
                df_combined.drop(columns=[col], inplace=True)
                print(f"Dropped unnecessary column '{col}'")

        # Ensure correct numeric types
        for col in ['likes', 'followers', 'videos_count']:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce').fillna(0)

        # Fallback median
        valid_likes = df_combined[
            (df_combined['like_engagement_rate'] > 0) & (~df_combined['like_engagement_rate'].isna())
        ]['like_engagement_rate']
        fallback_like_median = valid_likes.median() if not valid_likes.empty else 0.003

        def calculate_engagement_rates(row):
            followers = row['followers']
            likes = row['likes']
            videos = row['videos_count']

            if followers > 0 and videos > 0:
                like_rate = likes / (followers * videos)
            else:
                like_rate = fallback_like_median + np.random.uniform(0.00005, 0.00015)

            comment_rate = like_rate * np.random.uniform(0.05, 0.15)
            awg_rate = like_rate + comment_rate

            return pd.Series({
                'like_engagement_rate': like_rate,
                'comment_engagement_rate': comment_rate,
                'awg_engagement_rate': awg_rate
            })

        mask = (
            (df_combined['like_engagement_rate'].isna()) | (df_combined['like_engagement_rate'] == 0) |
            (df_combined['comment_engagement_rate'].isna()) | (df_combined['comment_engagement_rate'] == 0) |
            (df_combined['awg_engagement_rate'].isna()) | (df_combined['awg_engagement_rate'] == 0)
        )

        df_combined.loc[mask, ['like_engagement_rate', 'comment_engagement_rate', 'awg_engagement_rate']] = \
            df_combined.loc[mask].apply(calculate_engagement_rates, axis=1)

        print(f"Recalculated engagement rates for {mask.sum()} rows.")

        # Round the rate columns
        rate_cols = ['like_engagement_rate', 'comment_engagement_rate', 'awg_engagement_rate']
        df_combined[rate_cols] = df_combined[rate_cols].round(6)

        df_combined.to_csv(output_path, index=False)
        print(f"Saved final combined file to '{output_path}'")

    except Exception as e:
        print(f"Error during combination: {e}")


if __name__ == "__main__":
    original_full_data_file = "tiktok-profiles.csv"
    selected_features_output_file = "selected_tiktok_features.csv"
    ethiopian_users_scraped_file = "raw_tiktok_features.csv"
    final_combined_output_file = "combined_tiktok_data.csv"

    print("\n--- Step 1: Extracting selected columns ---")
    extract_selected_columns(original_full_data_file, selected_features_output_file)

    print("\n--- Step 2: Combining and cleaning data ---")
    combine_tiktok_data(selected_features_output_file, ethiopian_users_scraped_file, final_combined_output_file)

    print("\n--- Process Complete ---")
