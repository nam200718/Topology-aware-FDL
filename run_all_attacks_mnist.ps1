$ErrorActionPreference = "Stop"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   TESTING 3 TYPES OF ATTACKS (MNIST)     " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Running attack: GRADIENT ASCENT (Enhancing gradient)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_sweep.yaml --attack gradient_ascent

Write-Host "`n[2/3] Running attack: SIGN FLIP (Flipping gradient)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_sweep.yaml --attack sign_flip

Write-Host "`n[3/3] Running attack: RANDOM NOISE (Adding random noise)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_sweep.yaml --attack random_noise

Write-Host "`n=========================================================" -ForegroundColor Green
Write-Host " 🎉 Finished all tests! Check the output folder!" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
