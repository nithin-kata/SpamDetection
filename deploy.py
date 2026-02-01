#!/usr/bin/env python3
"""
Deployment helper script for the Text Classification System.
Provides easy commands for different deployment scenarios.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"\n🚀 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_requirements():
    """Check if required files exist."""
    required_files = [
        "requirements.txt",
        "app.py",
        "src/preprocessing.py",
        "src/models.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def deploy_local():
    """Deploy locally using Streamlit."""
    print("🏠 Local Deployment")
    print("=" * 50)
    
    if not check_requirements():
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Run tests
    if not run_command("python test_system.py", "Running system tests"):
        print("⚠️  Tests failed, but continuing with deployment...")
    
    # Start Streamlit
    print("\n🌟 Starting Streamlit application...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    
    os.system("streamlit run app.py")
    return True

def deploy_docker():
    """Deploy using Docker."""
    print("🐳 Docker Deployment")
    print("=" * 50)
    
    if not check_requirements():
        return False
    
    # Build Docker image
    if not run_command("docker build -t text-classifier .", "Building Docker image"):
        return False
    
    # Run Docker container
    print("\n🌟 Starting Docker container...")
    print("The app will be available at http://localhost:8501")
    print("Press Ctrl+C to stop the container")
    
    os.system("docker run -p 8501:8501 text-classifier")
    return True

def deploy_api():
    """Deploy FastAPI REST API."""
    print("🔌 API Deployment")
    print("=" * 50)
    
    if not check_requirements():
        return False
    
    # Install API dependencies
    if not run_command("pip install fastapi uvicorn", "Installing API dependencies"):
        return False
    
    # Start API server
    print("\n🌟 Starting FastAPI server...")
    print("API will be available at http://localhost:8000")
    print("Swagger docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    os.system("python deploy_api.py")
    return True

def setup_heroku():
    """Setup files for Heroku deployment."""
    print("☁️  Heroku Setup")
    print("=" * 50)
    
    if not check_requirements():
        return False
    
    # Check if Heroku CLI is installed
    try:
        subprocess.run(["heroku", "--version"], check=True, capture_output=True)
        print("✅ Heroku CLI found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Heroku CLI not found. Please install it first:")
        print("https://devcenter.heroku.com/articles/heroku-cli")
        return False
    
    # Create Heroku app
    app_name = input("Enter your Heroku app name (or press Enter for auto-generated): ").strip()
    
    if app_name:
        command = f"heroku create {app_name}"
    else:
        command = "heroku create"
    
    if not run_command(command, "Creating Heroku app"):
        return False
    
    # Initialize git if needed
    if not Path(".git").exists():
        run_command("git init", "Initializing git repository")
        run_command("git add .", "Adding files to git")
        run_command('git commit -m "Initial commit"', "Creating initial commit")
    
    # Deploy to Heroku
    if not run_command("git push heroku main", "Deploying to Heroku"):
        print("If this is your first push, you might need to:")
        print("1. git remote add heroku <your-heroku-git-url>")
        print("2. git push heroku main")
        return False
    
    print("\n🎉 Heroku deployment complete!")
    run_command("heroku open", "Opening your app in browser")
    return True

def setup_streamlit_cloud():
    """Setup for Streamlit Cloud deployment."""
    print("☁️  Streamlit Cloud Setup")
    print("=" * 50)
    
    print("To deploy on Streamlit Cloud:")
    print("1. Push your code to GitHub")
    print("2. Go to https://share.streamlit.io")
    print("3. Connect your GitHub account")
    print("4. Select this repository")
    print("5. Set main file path: app.py")
    print("6. Click Deploy")
    
    # Check if git is initialized
    if not Path(".git").exists():
        setup_git = input("Initialize git repository? (y/n): ").lower().startswith('y')
        if setup_git:
            run_command("git init", "Initializing git repository")
            run_command("git add .", "Adding files to git")
            run_command('git commit -m "Initial commit"', "Creating initial commit")
            
            repo_url = input("Enter your GitHub repository URL (optional): ").strip()
            if repo_url:
                run_command(f"git remote add origin {repo_url}", "Adding remote origin")
                print("Now run: git push -u origin main")
    
    return True

def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Deploy Text Classification System")
    parser.add_argument(
        "deployment_type",
        choices=["local", "docker", "api", "heroku", "streamlit-cloud"],
        help="Type of deployment"
    )
    
    args = parser.parse_args()
    
    print("🚀 Text Classification System Deployment")
    print("=" * 60)
    
    success = False
    
    if args.deployment_type == "local":
        success = deploy_local()
    elif args.deployment_type == "docker":
        success = deploy_docker()
    elif args.deployment_type == "api":
        success = deploy_api()
    elif args.deployment_type == "heroku":
        success = setup_heroku()
    elif args.deployment_type == "streamlit-cloud":
        success = setup_streamlit_cloud()
    
    if success:
        print("\n🎉 Deployment completed successfully!")
    else:
        print("\n❌ Deployment failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()