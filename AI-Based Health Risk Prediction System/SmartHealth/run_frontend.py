"""
SmartHealth AI - Frontend Quick Start
Choose between Streamlit and Flask interfaces
"""

import subprocess
import sys
import os
import webbrowser
import time

def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print("🏥 SmartHealth AI - Frontend Quick Start")
    print("="*70)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_streamlit():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        print(f"  ✓ Streamlit {streamlit.__version__}")
        return True
    except ImportError:
        print("  ✗ Streamlit (not installed)")
        return False

def check_flask():
    """Check if Flask is installed"""
    try:
        import flask
        print(f"  ✓ Flask {flask.__version__}")
        return True
    except ImportError:
        print("  ✗ Flask (not installed)")
        return False

def install_package(package_name):
    """Install a package"""
    print(f"\nInstalling {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✓ {package_name} installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        return False

def run_streamlit():
    """Run Streamlit application"""
    print("\n" + "="*70)
    print("Starting Streamlit Application...")
    print("="*70)
    print("\n📍 Streamlit will open in your browser at: http://localhost:8501")
    print("💡 Tip: Press Ctrl+C to stop the server\n")
    
    time.sleep(2)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_streamlit.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit stopped.")
    except Exception as e:
        print(f"\n❌ Error running Streamlit: {e}")

def run_flask():
    """Run Flask API server"""
    print("\n" + "="*70)
    print("Starting Flask API Server...")
    print("="*70)
    print("\n📍 API Server running at: http://localhost:5000")
    print("🌐 Open index.html in your browser")
    print("💡 Tip: Press Ctrl+C to stop the server\n")
    
    time.sleep(2)
    
    try:
        subprocess.run([sys.executable, "api_flask.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Flask server stopped.")
    except Exception as e:
        print(f"\n❌ Error running Flask: {e}")

def main():
    """Main menu"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Required dependencies missing!")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "-"*70)
    print("Checking frontend options...\n")
    
    streamlit_available = check_streamlit()
    flask_available = check_flask()
    
    print("\n" + "-"*70)
    print("\nChoose a frontend:\n")
    
    options = []
    
    if streamlit_available:
        print("1. 🚀 Streamlit (Recommended - Easiest to use)")
        print("   - Single file, no backend needed")
        print("   - Interactive sliders and inputs")
        print("   - Responsive design")
        print("   - Command: streamlit run app_streamlit.py\n")
        options.append(('streamlit', run_streamlit))
    
    if flask_available:
        print("2. 🌐 Flask + HTML Frontend")
        print("   - RESTful API backend")
        print("   - Modern web interface")
        print("   - Batch prediction support")
        print("   - Commands:")
        print("     - Backend: python api_flask.py")
        print("     - Frontend: Open index.html\n")
        options.append(('flask', run_flask))
    
    if not streamlit_available and not flask_available:
        print("❌ No frontend packages installed!")
        print("\nInstall options:")
        
        response = input("\nInstall Streamlit? (y/n): ").lower().strip()
        if response == 'y':
            if install_package('streamlit>=1.30.0'):
                run_streamlit()
        
        response = input("\nInstall Flask? (y/n): ").lower().strip()
        if response == 'y':
            if install_package('flask>=3.0.0') and install_package('flask-cors>=4.0.0'):
                run_flask()
        
        return
    
    print("0. ❌ Exit\n")
    
    choice = input("Enter your choice (0-{}): ".format(len(options))).strip()
    
    try:
        choice_num = int(choice)
        
        if choice_num == 0:
            print("\n👋 Goodbye!")
            sys.exit(0)
        elif 1 <= choice_num <= len(options):
            app_name, run_func = options[choice_num - 1]
            run_func()
        else:
            print("\n❌ Invalid choice!")
    except ValueError:
        print("\n❌ Invalid input! Please enter a number.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
