@echo off
rem Build "PD Watchlist.exe" (window with the dashboard ticker list) into dist\
rem Needs Python 3.14 on PATH. First run creates .venv-watch and installs watch_app\requirements.txt.
cd /d "%~dp0.."
if not exist .venv-watch\Scripts\python.exe (
  python -m venv .venv-watch || goto :fail
  .venv-watch\Scripts\python.exe -m pip install -r watch_app\requirements.txt || goto :fail
)
.venv-watch\Scripts\python.exe watch_app\pd_watch.py --make-icon || goto :fail
.venv-watch\Scripts\pyinstaller.exe --noconfirm --onefile --windowed --name "PD Watchlist" --icon "%CD%\watch_app\pd_watch.ico" --add-data "%CD%\watch_app\pd_watch.ico;." --distpath dist --workpath build\watch --specpath build\watch watch_app\pd_watch.py || goto :fail
echo.
echo Done: dist\PD Watchlist.exe
goto :eof
:fail
echo Build failed.
exit /b 1
