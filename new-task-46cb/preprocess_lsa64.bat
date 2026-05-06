@echo off
echo ============================================
echo  Gestura - LSA64 Dataset Preprocessing
echo ============================================
call venv\Scripts\activate.bat

if "%1"=="" (
    echo ERROR: Provide path to LSA64 dataset folder.
    echo Usage: preprocess_lsa64.bat C:\Users\jemun\.zenflow\worktrees\new-task-46cb\final_dataset
    pause
    exit /b 1
)

python ml\preprocess_lsa64.py --lsa64_dir %1
echo.
echo Done. Now run: train.bat
pause
