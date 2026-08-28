$ErrorActionPreference = "Continue"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   TESTING 4 TYPES OF ATTACKS (CIFAR-10 + ResNet9)       " -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "CẢNH BÁO: Tác vụ này sẽ ngốn khá nhiều thời gian (Treo qua đêm)!" -ForegroundColor Red

Write-Host "`n[1/4] Running attack: LABEL FLIP (Data poisoning)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_ablation.yaml --attack label_flip

Write-Host "`n[2/4] Running attack: GRADIENT ASCENT (Magnitude Inflation)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_ablation.yaml --attack gradient_ascent

Write-Host "`n[3/4] Running attack: SIGN FLIP (Flipping gradient)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_ablation.yaml --attack sign_flip

Write-Host "`n[4/4] Running attack: RANDOM NOISE (High variance noise)..." -ForegroundColor Yellow
python scripts/run_defense_experiment.py --config configs/defense_ablation.yaml --attack random_noise

Write-Host "`n=========================================================" -ForegroundColor Green
Write-Host " 🎉 Finished ALL CIFAR-10 TESTS! Check the output folder!" -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
