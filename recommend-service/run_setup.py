import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Starting E-commerce Recommendation Service Setup")
    
    required_files = ['config.py', 'model_utils.py', 'app.py', 'requirements.txt', '.env']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file {file} not found")
            return False
    
    print("✅ All required files found")
    
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    if os.path.exists('aggregated_interactions.csv'):

        if not run_command("python train_model.py", "Training model"):
            return False
    else:
        print("⚠️ Warning: aggregated_interactions.csv not found. Make sure to add your data file.")
        print("💡 You can still test the service, but recommendations won't work without the trained model.")
    
    print("\n🎉 Setup complete! You can now run the service with:")
    print("python app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
