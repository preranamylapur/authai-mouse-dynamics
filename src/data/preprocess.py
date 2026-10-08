import pandas as pd
import os

TRAINING_DIR = "data/raw/Mouse-Dynamics-Challenge-master-2/training_files"

def load_session(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    df = df.sort_values("client timestamp").reset_index(drop=True)
    return df

def load_all_users(training_dir):
    all_data = {}
    for user in os.listdir(training_dir):
        user_path = os.path.join(training_dir, user)
        if not os.path.isdir(user_path):
            continue
        sessions = {}
        for filename in os.listdir(user_path):
            filepath = os.path.join(user_path, filename)
            sessions[filename] = load_session(filepath)
        all_data[user] = sessions
    return all_data

if __name__ == "__main__":
    data = load_all_users(TRAINING_DIR)
    for user, sessions in data.items():
        total_events = sum(len(df) for df in sessions.values())
        print(f"{user}: {len(sessions)} sessions, {total_events} total events")