@echo off
title market-state-lab - current state
cd /d "%~dp0"
py -m msl.cli state -c configs/trend_indices.yaml --refresh
echo.
pause
