import requests
import time
import os

BASE_URL = "http://localhost:8000"
PROJECT_ID = "test-project-123"

def test_api():
    print("--- Testing Upload Flow ---")
    with open("test.txt", "w") as f:
        f.write("This is a test document about the solar system. The sun is at the center.")
    
    with open("test.txt", "rb") as f:
        res = requests.post(f"{BASE_URL}/v1/data/upload/{PROJECT_ID}", files={"file": f})
        print("Upload Response:", res.status_code, res.json())
        assert res.status_code == 200

    print("\n--- Testing Process Flow ---")
    res = requests.post(f"{BASE_URL}/v1/data/process/{PROJECT_ID}", json={"chunk_size": 100, "overlap_size": 20, "do_reset": 1})
    print("Process Request Response:", res.status_code, res.json())
    assert res.status_code == 202
    
    print("Waiting for processing to complete...")
    for _ in range(20):
        time.sleep(2)
        res = requests.get(f"{BASE_URL}/v1/data/status/{PROJECT_ID}")
        data = res.json()
        print("Status:", data["status"], data["detail"])
        if data["is_ready"]:
            print("Processing complete!")
            break
        if data["is_failed"]:
            print("Processing failed!")
            break
            
    print("\n--- Testing RAG Flow ---")
    res = requests.post(f"{BASE_URL}/v1/nlp/agent-query", json={
        "query": "What is at the center of the solar system?",
        "project_id": PROJECT_ID
    })
    print("Query Response:", res.status_code)
    try:
        print(res.json())
    except:
        print(res.text)

if __name__ == "__main__":
    test_api()
