@echo off
echo PatchVision Windows Setup

REM Check Python version
python --version
if errorlevel 1 (
    echo Python not found. Please install Python 3.9 or later.
    pause
    exit /b 1
)

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

REM Install development dependencies (optional)
pip install pytest pytest-cov

echo Setup complete!
echo To activate virtual environment: venv\Scripts\activate
pause