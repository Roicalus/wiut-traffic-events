# Dev setup for a laptop (Windows). Run from the repo folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# This file is intentionally ASCII-only: Windows PowerShell 5.1 reads UTF-8
# files without BOM as ANSI, which breaks non-ASCII text and quotes.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location $PSScriptRoot

function Run([string]$cmd) {
    Write-Host ">> $cmd"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}

# Python: prefer the py launcher (installed with Python from python.org)
if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py -3" } else { $py = "python" }
Run "$py -c `"import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'; print(sys.version)`""

if (-not (Test-Path ".venv")) { Run "$py -m venv .venv" }
$vpy = ".\.venv\Scripts\python.exe"
Run "$vpy -m pip install --upgrade pip"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host ">> NVIDIA GPU found: installing torch with CUDA"
    Run "$vpy -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128"
} else {
    Write-Host ">> No NVIDIA GPU: CPU torch will be used (fine for rule development)"
}
Run "$vpy -m pip install -r requirements.txt"
Run "$vpy -m pip install pytest"

# Old environments may still have opencv-python-headless (earlier requirements.txt); it
# shadows the GUI build and breaks label_tool / define_zones. Keep only opencv-python.
& $vpy -m pip uninstall -y opencv-python-headless opencv-python | Out-Null
Run "$vpy -m pip install `"opencv-python>=4.8,<5`""

# model weights + sha256 check
$weights = @{
    "yolo11s.pt" = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
    "yolo11n.pt" = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
}
foreach ($name in $weights.Keys) {
    $path = Join-Path "weights" $name
    if (-not (Test-Path $path)) {
        Write-Host ">> downloading ${name}"
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/${name}" -OutFile $path
    }
    $hash = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $weights[$name]) {
        throw "${name}: wrong sha256. Delete $path and run setup.ps1 again."
    }
    Write-Host "   ${name} OK"
}

foreach ($d in @("samples", "labels", "cache", "debug")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Run "$vpy -m pytest tests -q"
& $vpy tools\check_env.py
Write-Host ""
Write-Host "Done. In every new terminal run first:  .\.venv\Scripts\Activate.ps1"
