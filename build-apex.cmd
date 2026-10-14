@echo off
rem Build the Strata engine for APEX models (CUDA, single or multiple architectures).
rem Usage: build-apex.cmd [arch list]   e.g. build-apex.cmd 75;86   (default 75;86)
rem Needs: Visual Studio 2022+ with C++ and CUDA workloads, CUDA Toolkit 13.x, Python 3.10+.

setlocal enabledelayedexpansion
set "ARCHS=%~1"
if "%ARCHS%"=="" set "ARCHS=75;86"

rem -- find the newest Visual Studio via vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere not found: install Visual Studio 2022+ with the C++ workload & exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" (
  echo No Visual Studio with the C++ workload found & exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

rem -- find the newest CUDA toolkit
set "CUDABIN="
for /f "delims=" %%d in ('dir /b /ad /o-n "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA" 2^>nul') do (
  if not defined CUDABIN if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%%d\bin\nvcc.exe" set "CUDABIN=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\%%d\bin"
)
if "%CUDABIN%"=="" (
  echo No CUDA Toolkit found: install CUDA 13.x & exit /b 1
)
echo CUDA: %CUDABIN%
set "PATH=%CUDABIN%;%PATH%"

where ninja >nul 2>nul || set "PATH=%PATH%;%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\Cmake\Ninja"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=%ARCHS% -DSTRATA_ENABLE_CUDA=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF || exit /b 1
ninja -C build strata || exit /b 1
echo.
echo Done: build\strata.exe  (architectures %ARCHS%)
