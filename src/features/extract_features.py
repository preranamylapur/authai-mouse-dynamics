import pandas as pd
import numpy as np

def compute_window_features(window):
    x = window["x"].values
    y = window["y"].values
    t = window["client timestamp"].values

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    valid = dt > 0
    dx, dy, dt = dx[valid], dy[valid], dt[valid]

    distances = np.sqrt(dx**2 + dy**2)
    total_distance = distances.sum()

    velocities = distances / dt
    avg_velocity = velocities.mean() if len(velocities) > 0 else 0
    max_velocity = velocities.max() if len(velocities) > 0 else 0
    min_velocity = velocities.min() if len(velocities) > 0 else 0

    accelerations = np.diff(velocities) / dt[1:] if len(velocities) > 1 else np.array([0])
    avg_acceleration = accelerations.mean() if len(accelerations) > 0 else 0

    jerks = np.diff(accelerations) if len(accelerations) > 1 else np.array([0])
    avg_jerk = jerks.mean() if len(jerks) > 0 else 0

    angles = np.arctan2(dy, dx)
    angle_changes = np.diff(angles) if len(angles) > 1 else np.array([0])
    avg_direction_change = np.abs(angle_changes).mean() if len(angle_changes) > 0 else 0

    straight_line_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2) if len(x) > 1 else 0
    straightness_ratio = straight_line_dist / total_distance if total_distance > 0 else 0

    click_count = (window["state"] == "Pressed").sum()

    duration = t[-1] - t[0] if len(t) > 1 else 1e-6
    click_frequency = click_count / duration if duration > 0 else 0

    velocity_variance = velocities.var() if len(velocities) > 0 else 0

    return {
        "total_distance": total_distance,
        "avg_velocity": avg_velocity,
        "max_velocity": max_velocity,
        "min_velocity": min_velocity,
        "avg_acceleration": avg_acceleration,
        "avg_jerk": avg_jerk,
        "avg_direction_change": avg_direction_change,
        "straightness_ratio": straightness_ratio,
        "click_count": click_count,
        "click_frequency": click_frequency,
        "velocity_variance": velocity_variance,
        "duration": duration,
    }

def extract_features_from_session(df, window_size=100):
    features = []
    for start in range(0, len(df) - window_size, window_size):
        window = df.iloc[start:start + window_size]
        feats = compute_window_features(window)
        features.append(feats)
    return pd.DataFrame(features)

if __name__ == "__main__":
    from src.data.preprocess import load_session
    df = load_session("data/raw/Mouse-Dynamics-Challenge-master-2/training_files/user7/session_1060325796")
    features_df = extract_features_from_session(df)
    print(features_df.head())
    print(f"Total windows: {len(features_df)}")