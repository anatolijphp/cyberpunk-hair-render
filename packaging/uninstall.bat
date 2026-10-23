@echo off
rem Restores the original FidelityFX loader. Usage: uninstall.bat ["C:\path\to\Cyberpunk 2077"]
setlocal
set "GAME=%~1"
if "%GAME%"=="" set /p "GAME=Cyberpunk 2077 folder: "
set "BIN=%GAME%\bin\x64"
if not exist "%BIN%\amd_fidelityfx_loader_dx12_orig.dll" (echo Nothing to do: no amd_fidelityfx_loader_dx12_orig.dll in %BIN% & exit /b 0)
move /y "%BIN%\amd_fidelityfx_loader_dx12_orig.dll" "%BIN%\amd_fidelityfx_loader_dx12.dll" >nul
echo Original loader restored; hairfix.ini / hairfix.log were left in %BIN%
