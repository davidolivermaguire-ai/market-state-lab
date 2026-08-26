@echo off
title market-state-lab - full sweep
cd /d "%~dp0"
echo ============================================================
echo   market-state-lab: 9 methods x 7 assets, common date range
echo   This takes roughly 12-15 minutes. Leave this window open.
echo ============================================================
echo.
py -m msl.cli run -c configs/trend_mixed.yaml --refresh
echo.
echo Done. Results are in the results\ folder.
pause
