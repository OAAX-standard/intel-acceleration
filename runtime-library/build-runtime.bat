REM Disable command echoing for cleaner output
@echo off

REM Enable delayed environment variable expansion
setlocal enabledelayedexpansion

REM Change working directory to the directory of this script (handles drives too)
cd /d %~dp0

REM Set BUILD_DIR variable to a 'build' subdirectory in the current directory
set "BUILD_DIR=%cd%\build"

REM Set ARTIFACTS_DIR variable to an 'artifacts' subdirectory in the current directory
set "ARTIFACTS_DIR=%cd%\artifacts"

REM Create the build directory if it does not exist
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Remove the artifacts directory and all its contents if it exists
if exist "%ARTIFACTS_DIR%" rmdir /s /q "%ARTIFACTS_DIR%"

REM Create a new, empty artifacts directory
mkdir "%ARTIFACTS_DIR%"

REM Read runtime version from parent directory's VERSION file
set "ROOT_DIR=%cd%\.."
set "VERSION_FILE=%ROOT_DIR%\VERSION"
if not exist "%VERSION_FILE%" (
	echo VERSION file not found at "%VERSION_FILE%"
	exit /b 1
)
for /f "usebackq delims=" %%A in ("%VERSION_FILE%") do (
	set "RUNTIME_VERSION=%%A"
	goto :got_version
)
:got_version
if not defined RUNTIME_VERSION (
	echo Failed to read runtime version from "%VERSION_FILE%"
	exit /b 1
)
echo Building runtime version: %RUNTIME_VERSION%

REM Change to the build directory and save the previous directory on the stack
pushd "%BUILD_DIR%"

REM Delete all files in the build directory quietly (ignore errors/output)
del /q * >nul 2>&1

REM Run CMake to generate build files using the parent directory as the source
cmake .. -DRUNTIME_VERSION="%RUNTIME_VERSION%"

REM If CMake failed, exit the script with error
if errorlevel 1 exit /b 1

REM Build the project in Release configuration
cmake --build . --config Release

REM If build failed, exit the script with error
if errorlevel 1 exit /b 1

REM Print message about build completion
echo Build complete. The following shared libraries were created:

REM List all DLL files in the Release directory (bare format)
dir /b Release\*

REM Print message about copying DLLs
echo Copying shared libraries to artifacts directory...

REM Create a Windows subdirectory in artifacts if it doesn't exist
if not exist "%ARTIFACTS_DIR%\Windows" mkdir "%ARTIFACTS_DIR%\Windows"

REM Copy all DLLs from Release to the artifacts Windows directory
copy Release\*.dll "%ARTIFACTS_DIR%\Windows\"

REM Create a gzipped tarball of the DLLs in the Windows artifacts directory
tar czf "%ARTIFACTS_DIR%\runtime-library-X86_64-Windows.tar.gz" -C "%ARTIFACTS_DIR%\Windows" *.dll

REM Print confirmation message
echo Shared libraries for Windows have been copied to "%ARTIFACTS_DIR%\Windows\"

REM Restore previous directory from the stack
popd

REM End local environment changes
endlocal
