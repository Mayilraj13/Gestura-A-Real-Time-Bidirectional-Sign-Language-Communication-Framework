@echo off
echo ============================================
echo  Gestura - Model Training (GPU: RTX 2050)
echo ============================================
call venv\Scripts\activate.bat
python ml\train_model.py
echo.
echo Training done. Model saved to backend\model\
pause
