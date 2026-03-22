@echo off
title BIST Tahmin Sistemi v5.0
cd /d "%~dp0"
echo.
echo ========================================
echo   BIST Tahmin Sistemi v5.0 Basliyor...
echo ========================================
echo.

REM Python 3.11 venv kontrol
if exist ".venv311\Scripts\activate.bat" (
    call .venv311\Scripts\activate.bat
    echo venv311 aktif edildi
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo venv aktif edildi
) else (
    echo HATA: Sanal ortam bulunamadi!
    echo Lutfen kurulum yapiniz.
    pause
    exit /b 1
)

echo.
echo Streamlit baslatiliyor...
echo Tarayici: http://localhost:8765
echo.
python -m streamlit run bist_streamlit_app.py --server.port 8765 --server.address localhost
pause
