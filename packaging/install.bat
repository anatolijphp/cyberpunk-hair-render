@echo off
rem Installs the hair dither fix. Usage: install.bat ["C:\path\to\Cyberpunk 2077"]
setlocal
set "GAME=%~1"
if "%GAME%"=="" set /p "GAME=Cyberpunk 2077 folder: "
set "BIN=%GAME%\bin\x64"
if not exist "%BIN%\Cyberpunk2077.exe" (echo Not a Cyberpunk 2077 folder: %GAME% & exit /b 1)
if not exist "%BIN%\amd_fidelityfx_loader_dx12_orig.dll" move "%BIN%\amd_fidelityfx_loader_dx12.dll" "%BIN%\amd_fidelityfx_loader_dx12_orig.dll" >nul
copy /y "%~dp0files\amd_fidelityfx_loader_dx12.dll" "%BIN%\" >nul
if not exist "%BIN%\hairfix.ini" copy "%~dp0files\hairfix.ini" "%BIN%\" >nul
echo Installed into %BIN% (an existing hairfix.ini was kept)
