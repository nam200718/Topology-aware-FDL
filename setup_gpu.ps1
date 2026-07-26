# Setup AMD DirectML GPU Environment for Topology-aware-FDL
$ErrorActionPreference = "Stop"

Write-Host "Setting up Python 3.12 Virtual Environment for AMD GPU (DirectML)..." -ForegroundColor Green

if (Test-Path ".venv312") {
    Write-Host ".venv312 already exists." -ForegroundColor Yellow
} else {
    Write-Host "Creating .venv312 with Python 3.12..."
    py -3.12 -m venv .venv312
}

Write-Host "Upgrading pip and installing PyTorch 2.3.1 + DirectML..." -ForegroundColor Green
.\.venv312\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv312\Scripts\python.exe -m pip install "numpy<2.0.0" "scipy<1.14"
.\.venv312\Scripts\python.exe -m pip install torch==2.3.1 torchvision==0.18.1 torch-directml
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`nTesting GPU Acceleration..." -ForegroundColor Green
.\.venv312\Scripts\python.exe -c "import torch, torch_directml; print('DirectML Available:', torch_directml.is_available()); print('Device:', torch_directml.device()); t = torch.ones(3, 3, device=torch_directml.device()); print('Test tensor on GPU:', t)"

Write-Host "`nSetup complete! Activate the environment with:" -ForegroundColor Green
Write-Host "  .\.venv312\Scripts\Activate.ps1" -ForegroundColor Cyan
