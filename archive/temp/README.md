# temp/ — Temporary Implementation Files

Code tạm thời cho Post-Midterm tasks. Sẽ merge vào codebase chính
sau khi teammate hoàn thành hierarchical ensemble refactoring.

## Tại sao dùng temp/?
Tránh git conflict với teammate đang sửa src/config.py, src/core/*, src/topologies/*.

## Cách chạy
python temp/run_partition_demo.py --sweep
python temp/run_baseline_comparison.py
python -m pytest temp/test_*.py -v
