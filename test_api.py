import requests
import json

def test_api():
    url = "http://localhost:8000/classify"
    data = {"purpose_text": "grocery shopping at walmart"}
    
    try:
        response = requests.post(url, json=data)
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()