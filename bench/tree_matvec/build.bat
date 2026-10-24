@echo off
rem build the standalone PTQ1_0 wide-column probe
setlocal
cd /d "%~dp0"
call "F:\qwen27b\src\code\build.local.bat"
if defined VCVARS call "%VCVARS%"
"F:\super\tools\cuda\bin\nvcc.exe" --use-local-env -O3 -arch=sm_89 -std=c++17 -Xcompiler="/O2 /EHsc" -o pt_wide.exe pt_wide.cu
echo BUILD_EXIT=%ERRORLEVEL%
endlocal
