# PowerShell script to compile BLT C++ native modules into .dll, .a, and .exe targets.
param(
    [switch]$Clean,
    [switch]$BuildAll = $true,
    [string]$Compiler = ""
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$BUILD_DIR = Join-Path $PROJECT_ROOT "build\csrc"
$BIN_DIR = Join-Path $BUILD_DIR "bin"
$OBJ_DIR = Join-Path $BUILD_DIR "obj"
$INC_DIR = Join-Path $SCRIPT_DIR "include"
$SRC_DIR = Join-Path $SCRIPT_DIR "src"
$CLI_DIR = Join-Path $SCRIPT_DIR "cli"
$BLT_CSRC_DIR = Join-Path $PROJECT_ROOT "blt\csrc"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Building BLT Native Engine (.dll, .a, .exe)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($Clean) {
    Write-Host "[INFO] Cleaning build directories..." -ForegroundColor Yellow
    if (Test-Path $BUILD_DIR) { Remove-Item -Recurse -Force $BUILD_DIR }
    Write-Host "[INFO] Clean complete." -ForegroundColor Green
}

New-Item -ItemType Directory -Force -Path $BIN_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $OBJ_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $BLT_CSRC_DIR | Out-Null

# Compiler detection
$CXX = $Compiler
if (-not $CXX) {
    if (Get-Command "g++" -ErrorAction SilentlyContinue) {
        $CXX = "g++"
    } elseif (Get-Command "clang++" -ErrorAction SilentlyContinue) {
        $CXX = "clang++"
    } else {
        Write-Error "No suitable C++ compiler (g++ or clang++) found in PATH."
    }
}

$AR = "ar"
if (-not (Get-Command $AR -ErrorAction SilentlyContinue)) {
    $AR = "llvm-ar"
    if (-not (Get-Command $AR -ErrorAction SilentlyContinue)) {
        Write-Warning "Archiver 'ar' not found in PATH; static .a creation might fail."
    }
}

Write-Host "[INFO] Using C++ compiler: $CXX" -ForegroundColor Yellow
& $CXX --version | Select-Object -First 1 | Write-Host -ForegroundColor Gray

# Compile flags
$CXX_FLAGS = @("-O3", "-std=c++17", "-Wall", "-I$INC_DIR", "-DBLT_BUILD_DLL")

$SRC_FILES = @(
    "story_dedup",
    "rolling_hash",
    "boundary_rules",
    "streaming_patcher",
    "batch_packing"
)

$OBJ_FILES = @()

# 1. Compile Core Object Files (.o)
Write-Host "`n[STEP 1/4] Compiling Core C++ Source Files..." -ForegroundColor Cyan
foreach ($src in $SRC_FILES) {
    $srcPath = Join-Path $SRC_DIR "$src.cpp"
    $objPath = Join-Path $OBJ_DIR "$src.o"
    $OBJ_FILES += $objPath

    Write-Host "  -> Compiling $src.cpp -> $src.o" -ForegroundColor Gray
    & $CXX $CXX_FLAGS -c $srcPath -o $objPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Compilation failed for $src.cpp"
    }
}

# Compile consolidated C API wrapper
$cApiObj = Join-Path $OBJ_DIR "blt_c_api.o"
Write-Host "  -> Compiling blt_c_api.cpp -> blt_c_api.o" -ForegroundColor Gray
& $CXX $CXX_FLAGS -c (Join-Path $SRC_DIR "blt_c_api.cpp") -o $cApiObj
$ALL_CORE_OBJS = $OBJ_FILES + $cApiObj

# 2. Build Static Library (.a)
Write-Host "`n[STEP 2/4] Archiving Static Library (.a)..." -ForegroundColor Cyan
$STATIC_LIB = Join-Path $BIN_DIR "libblt_native.a"
if (Test-Path $STATIC_LIB) { Remove-Item -Force $STATIC_LIB }

Write-Host "  -> Creating $STATIC_LIB" -ForegroundColor Gray
& $AR rcs $STATIC_LIB $ALL_CORE_OBJS
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Archiver returned exit code $LASTEXITCODE"
} else {
    Write-Host "  [OK] Successfully created libblt_native.a" -ForegroundColor Green
}

# 3. Build Dynamic Shared Library (.dll)
Write-Host "`n[STEP 3/4] Linking Dynamic Shared Library (.dll)..." -ForegroundColor Cyan
$DLL_FILE = Join-Path $BIN_DIR "blt_native.dll"
$IMPLIB_FILE = Join-Path $BIN_DIR "libblt_native.dll.a"

Write-Host "  -> Linking $DLL_FILE" -ForegroundColor Gray
$linkArgs = @("-shared", "-O3") + $ALL_CORE_OBJS + @("-o", $DLL_FILE, "-Wl,--out-implib,$IMPLIB_FILE")
& $CXX $linkArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "DLL linking failed for blt_native.dll"
} else {
    Write-Host "  [OK] Successfully created blt_native.dll" -ForegroundColor Green
    Copy-Item -Force $DLL_FILE (Join-Path $BLT_CSRC_DIR "blt_native.dll")
    Write-Host "  [OK] Copied blt_native.dll to blt/csrc/ for Python runtime discovery" -ForegroundColor Green
}

# 4. Build Standalone CLI Executables (.exe)
Write-Host "`n[STEP 4/4] Building Standalone Executables (.exe)..." -ForegroundColor Cyan

$CLI_APPS = @(
    @{ Name = "blt_native"; Source = "blt_native_cli.cpp" },
    @{ Name = "blt_dedup"; Source = "dedup_cli.cpp" },
    @{ Name = "blt_rolling_hash_bench"; Source = "rolling_hash_bench.cpp" },
    @{ Name = "blt_boundary_rules_test"; Source = "boundary_rules_test.cpp" },
    @{ Name = "blt_streaming_patcher_cli"; Source = "streaming_patcher_cli.cpp" }
)

foreach ($app in $CLI_APPS) {
    $appName = $app.Name
    $appSrc = Join-Path $CLI_DIR $app.Source
    $appExe = Join-Path $BIN_DIR "$appName.exe"

    Write-Host "  -> Building $appName.exe from $($app.Source)" -ForegroundColor Gray
    $cliArgs = @("-O3", "-std=c++17", "-I$INC_DIR", $appSrc, $STATIC_LIB, "-o", $appExe)
    & $CXX $cliArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to link $appName.exe with static library, attempting direct object link..."
        $cliArgsFallback = @("-O3", "-std=c++17", "-I$INC_DIR", $appSrc) + $ALL_CORE_OBJS + @("-o", $appExe)
        & $CXX $cliArgsFallback
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build $appName.exe"
        }
    }
    Write-Host "  [OK] Successfully created $appName.exe" -ForegroundColor Green
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Native C++ Build Completed Successfully!" -ForegroundColor Green
Write-Host " Outputs located in: $BIN_DIR" -ForegroundColor Green
Write-Host "   - Dynamic Library: blt_native.dll" -ForegroundColor Yellow
Write-Host "   - Static Library:  libblt_native.a" -ForegroundColor Yellow
Write-Host "   - Executables:     blt_native.exe, blt_dedup.exe, etc." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
