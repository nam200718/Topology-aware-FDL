# Queue completion watchdog: waits for the driver PID, then auto-runs the
# latency probe and result summarizers, and drops a completion marker.
param(
    [int]$TargetPid = 3564,
    [string]$Python = ".venv312/Scripts/python.exe"
)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)

try { Wait-Process -Id $TargetPid -ErrorAction Stop } catch { }
Write-Output "[watchdog] queue process exited at $(Get-Date -Format s)"

Write-Output "[watchdog] running latency probe..."
& $Python scripts/profile_fedala_cfl_latency.py 2>&1 |
    Tee-Object -FilePath "outputs/latency_probe.txt"

Write-Output "[watchdog] rendering remaining results..."
& $Python scripts/render_remaining_results.py 2>&1 |
    Tee-Object -FilePath "outputs/remaining_results.txt"

Write-Output "[watchdog] rendering Table III rows (seed 42)..."
& $Python scripts/render_table3_rows.py --seed 42 2>&1 |
    Tee-Object -FilePath "outputs/table3_rows_seed42.txt"

"QUEUE COMPLETE $(Get-Date -Format s)" | Set-Content "outputs/QUEUE_COMPLETE.marker"
Write-Output "[watchdog] marker written. Done."
