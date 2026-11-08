@echo off
echo ======================================================================
echo Generating Participant_Independent_sEMG_HGR_Paper.docx
echo ======================================================================
cd /d "%~dp0"
python build_docx.py
echo.
echo Process complete. Check for Participant_Independent_sEMG_HGR_Paper.docx
echo ======================================================================
pause
