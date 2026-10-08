import pandas as pd
import os
from src.data.preprocess import load_all_users
from src.features.extract_features import extract_features_from_session

TRAINING_DIR = "data/raw/Mouse-Dynamics-Challenge-master-2/training_files"

def build_training_dataset(target_user, window_size=100):
    all_data = load_all_users(TRAINING_DIR)
    rows = []

    for user, sessions in all_data.items():
        label = 1 if user == target_user else 0
        for session_name, df in sessions.items():
            feats_df = extract_features_from_session(df, window_size=window_size)
            feats_df["label"] = label
            feats_df["user"] = user
            feats_df["session"] = session_name
            rows.append(feats_df)

    full_dataset = pd.concat(rows, ignore_index=True)
    return full_dataset

if __name__ == "__main__":
    dataset = build_training_dataset(target_user="user7")
    print(dataset.head())
    print(f"Total rows: {len(dataset)}")
    print(dataset["label"].value_counts())

    os.makedirs("data/processed", exist_ok=True)
    dataset.to_csv("data/processed/user7_dataset.csv", index=False)
    print("Saved to data/processed/user7_dataset.csv")