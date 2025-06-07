import os
import sys
import subprocess

def setup_local_environment():
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    
    # Determine the correct activation script
    if os.name == 'nt':  # Windows
        activate_script = os.path.join('venv', 'Scripts', 'activate')
        pip_path = os.path.join('venv', 'Scripts', 'pip')
    else:  # Unix/Linux/Mac
        activate_script = os.path.join('venv', 'bin', 'activate')
        pip_path = os.path.join('venv', 'bin', 'pip')
    
    # Install requirements
    print("Installing requirements...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'])
    
    # Create .env file
    print("Creating .env file...")
    with open('.env', 'w') as f:
        f.write('FLASK_ENV=development\n')
        f.write('TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n')
    
    print("\nSetup complete! To start the application:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    print("2. Run the application:")
    print("   python app.py")

if __name__ == '__main__':
    setup_local_environment() 