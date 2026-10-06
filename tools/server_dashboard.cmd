@echo off
rem Pump Detector dashboard from the server: SSH tunnel (laptop 8766 -> server 8766) + browser.
rem Double-click. Keep this window open while you use http://localhost:8766/ ; closing it closes the tunnel.
rem "port 8766 in use": a local dashboard.py or another tunnel is still running - close it first.
title Pump Detector - tunnel to server (keep open)
start "" /b cmd /c "timeout /t 4 >nul & start http://localhost:8766/"
echo Tunnel to 45.76.90.13 is open. Dashboard: http://localhost:8766/  (close this window to stop)
ssh -p 2222 -i D:\keys\pd.pem -N -L 8766:127.0.0.1:8766 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes ubuntu@45.76.90.13
echo.
echo Tunnel closed. Press any key.
pause >nul
