# Build BLT Native 64-bit DLL using MSVC (Visual Studio 2022 / Community)
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $ProjectRoot "build\csrc\bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    # Fallback to standard 2022 path if different
    $found = Get-ChildItem "C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $vcvars = $found.FullName } else { Write-Error "Could not find vcvars64.bat" }
}

Write-Host "[INFO] Using MSVC environment from: $vcvars" -ForegroundColor Cyan

$sources = @(
    "csrc\src\story_dedup.cpp",
    "csrc\src\rolling_hash.cpp",
    "csrc\src\boundary_rules.cpp",
    "csrc\src\streaming_patcher.cpp",
    "csrc\src\batch_packing.cpp",
    "csrc\src\blt_c_api.cpp"
) -join " "

$cmd = "call `"$vcvars`" && cl /O2 /std:c++17 /EHsc /LD /DBLT_BUILD_DLL /Icsrc\include $sources /Fe:build\csrc\bin\blt_native.dll /link /IMPLIB:build\csrc\bin\libblt_native.lib"

cmd.exe /c $cmd
if ($LASTEXITCODE -ne 0) {
    Write-Error "MSVC compilation failed with exit code $LASTEXITCODE"
}

Remove-Item -Force story_dedup.obj, rolling_hash.obj, boundary_rules.obj, streaming_patcher.obj, batch_packing.obj, blt_c_api.obj -ErrorAction SilentlyContinue

Write-Host "[SUCCESS] 64-bit blt_native.dll successfully built!" -ForegroundColor Green
