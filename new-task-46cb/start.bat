@echo off
echo ============================================
echo  Gestura - Starting Application
echo ============================================

echo Starting FastAPI backend on port 8000...
start "Gestura Backend" cmd /k "call venv\Scripts\activate.bat && cd backend && python main.py"

timeout /t 3 /nobreak >nul

echo Starting React frontend on port 3000...
start "Gestura Frontend" cmd /k "cd frontend && npm start"

echo.
echo ============================================
echo  App running at: http://localhost:3000
echo  API docs at:    http://localhost:8000/docs
echo ============================================
