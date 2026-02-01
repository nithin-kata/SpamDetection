"""
Test script for the FastAPI deployment.
"""

import requests
import time
import subprocess
import sys
import os
from multiprocessing import Process

def start_api_server():
    """Start the API server in a separate process."""
    os.system("python deploy_api.py")

def test_api_endpoints():
    """Test all API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API Endpoints")
    print("=" * 40)
    
    # Wait for server to start
    print("Waiting for API server to start...")
    time.sleep(5)
    
    try:
        # Test health check
        print("\n1. Testing health check endpoint...")
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        health_data = response.json()
        print(f"✅ Health check: {health_data}")
        
        # Test single prediction
        print("\n2. Testing single prediction...")
        test_text = "This movie was absolutely fantastic! Great acting and storyline."
        response = requests.post(
            f"{base_url}/predict",
            json={"text": test_text}
        )
        assert response.status_code == 200
        prediction_data = response.json()
        print(f"✅ Prediction: {prediction_data['predicted_class']} ({prediction_data['confidence']:.2%})")
        
        # Test batch prediction
        print("\n3. Testing batch prediction...")
        test_texts = [
            "Great movie, loved it!",
            "Terrible film, waste of time.",
            "Average movie, nothing special."
        ]
        response = requests.post(
            f"{base_url}/batch_predict",
            json=test_texts
        )
        assert response.status_code == 200
        batch_data = response.json()
        print(f"✅ Batch prediction: {len(batch_data['predictions'])} results")
        
        # Test model info
        print("\n4. Testing model info...")
        response = requests.get(f"{base_url}/models")
        assert response.status_code == 200
        model_data = response.json()
        print(f"✅ Model info: {model_data}")
        
        print("\n🎉 All API tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running.")
        return False
    except AssertionError as e:
        print(f"❌ API test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 API Testing Script")
    print("=" * 50)
    
    # Check if API server is already running
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        print("✅ API server is already running")
        server_running = True
    except:
        print("🔄 Starting API server...")
        server_running = False
        
        # Start API server in background
        api_process = Process(target=start_api_server)
        api_process.start()
        
        # Wait for server to start
        time.sleep(10)
    
    # Run tests
    success = test_api_endpoints()
    
    # Clean up
    if not server_running:
        try:
            api_process.terminate()
            api_process.join()
        except:
            pass
    
    if success:
        print("\n✅ API testing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ API testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()