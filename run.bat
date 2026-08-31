@echo off
REM ============================================================
REM Lithium-Ion Battery SoH Prediction - Windows Run Script
REM ============================================================

echo.
echo === Battery SoH Prediction Project ===
echo.

IF NOT EXIST venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) ELSE (
    echo [1/3] Virtual environment already exists, skipping creation.
)

echo [2/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul
pip install -r requirements.txt

echo.
echo [3/3] Running the SoH prediction pipeline...
echo.
python main.py

echo.
echo ===============================================
echo  Done. Check the outputs\ folder for results.
echo ===============================================
pause
