@echo off
echo ============================================
echo  Gestura - One-Time Setup
echo ============================================

echo.
echo [1/4] Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Installing Python backend dependencies...
pip install --upgrade pip
pip install -r backend\requirements.txt

echo.
echo [3/4] Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo [4/4] Installing React frontend dependencies...
cd frontend
npm install
cd ..

echo.
echo ============================================
echo  Setup complete!
echo  Next steps:
echo    1. Download WLASL dataset from Kaggle
echo    2. Run: preprocess.bat --wlasl_dir YOUR_PATH
echo    3. Run: train.bat
echo    4. Run: start.bat
echo ============================================
pause
