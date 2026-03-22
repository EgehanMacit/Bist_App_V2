@echo off
title BIST Tahmin - Kurulum
cd /d "%~dp0"
echo.
echo ========================================
echo   BIST Tahmin Sistemi - Kurulum
echo ========================================
echo.

REM Python 3.12 kontrol
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo HATA: Python 3.12 bulunamadi!
    echo https://www.python.org/downloads/release/python-31213/ adresinden indirip kurun.
    echo Kurulumda "Add Python to PATH" kutusunu isaretleyin!
    pause
    exit /b 1
)

echo Python 3.12 bulundu.
echo.

REM Eski venv varsa sil
if exist ".venv" (
    echo Eski .venv siliniyor...
    rmdir /s /q .venv
)

echo Sanal ortam olusturuluyor...
py -3.12 -m venv .venv

echo Aktif ediliyor...
call .venv\Scripts\activate.bat

echo Paketler yukleniyor (5-10 dakika)...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo   KURULUM TAMAMLANDI!
echo   CALISTIR.bat ile baslatabilirsiniz.
echo ========================================
pause