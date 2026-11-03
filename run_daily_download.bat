@echo off
rem Windows Task Scheduler runs this each weekday evening. Output goes to logs\options_download.log.
cd /d "%~dp0"
if not exist logs mkdir logs
echo ===== %date% %time% >> logs\options_download.log
venv\Scripts\python.exe download_options.py >> logs\options_download.log 2>&1
exit /b %errorlevel%
