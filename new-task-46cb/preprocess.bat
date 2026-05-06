@echo off
echo ============================================
echo  Gestura - Dataset Preprocessing
echo ============================================
call venv\Scripts\activate.bat

if "%1"=="" (
    echo ERROR: Provide the WLASL dataset path.
    echo Usage: preprocess.bat C:\path\to\wlasl_videos
    pause
    exit /b 1
)

python ml\preprocess_wlasl.py --wlasl_dir %1 --max_classes 300
echo.
echo Preprocessing done. Run train.bat next.
pause
