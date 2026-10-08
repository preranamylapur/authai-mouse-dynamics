import pandas as pd
import os

def load_session(filepath):
    return pd.read_csv(filepath)

def load_user_sessions(user_folder):
    sessions = {}
    for filename in os.listdir(user_folder):
        filepath = os.path.join(user_folder, filename)
        sessions[filename] = load_session(filepath)
    return sessions

if __name__ == "__main__":
    user7_folder = "data/raw/Mouse-Dynamics-Challenge-master-2/training_files/user7"
    sessions = load_user_sessions(user7_folder)

    print(f"Number of sessions for user7: {len(sessions)}")
    for name, df in sessions.items():
        print(f"{name}: {len(df)} events")