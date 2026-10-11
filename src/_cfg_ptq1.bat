@echo off
rem Configure a clean build dir for the pr-ptq1-mmv experiment.
rem Mirrors the flags of the working build-win-kvmc (Release, sm_89, CUDA, FA, graphs).
rem NCCL off to avoid a configure-time dependency we don't need on a single GPU.
setlocal
call F:\super\tools\vcvars.bat
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
cd /d F:\super\llm\forks\llamAmpere
"C:\Program Files\CMake\bin\cmake.exe" -B build-ptq1 -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON ^
  -DGGML_CUDA_NCCL=OFF ^
  -DLLAMA_CURL=OFF ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF
echo CFG_EXIT=%ERRORLEVEL%
