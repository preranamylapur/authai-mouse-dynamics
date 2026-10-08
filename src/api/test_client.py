import pandas as pd
import requests
from src.data.preprocess import load_session

session_path = "data/raw/Mouse-Dynamics-Challenge-master-2/training_files/user7/session_1060325796"
df = load_session(session_path)

window = df.iloc[0:100]
events = window.to_dict(orient="records")

response = requests.post(
    "http://127.0.0.1:5000/authenticate",
    json={"events": events, "model": "user7"}
)

print("Status code:", response.status_code)
print("Response:", response.json())