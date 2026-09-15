import requests

url = "http://127.0.0.1:5000/predict"

player = {
    "age": 25,
    "height_cm": 180,
    "weight_kg": 75,
    "pace": 75,
    "shooting": 70,
    "passing": 72,
    "dribbling": 74,
    "defending": 50,
    "physic": 70,
    "attacking_crossing": 60,
    "attacking_finishing": 65,
    "skill_dribbling": 70,
    "skill_ball_control": 75,
    "power_shot_power": 70,
    "power_stamina": 75,
    "power_strength": 65,
    "mentality_aggression": 50,
    "mentality_interceptions": 30,
    "mentality_positioning": 60,
    "mentality_vision": 65
}

response = requests.post(url, json=player)

print("API Response:")
print(response.json())