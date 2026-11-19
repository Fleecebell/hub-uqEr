@echo off
rem Run this assignment with the py312 interpreter (sklearn/torch live there).
rem Usage: run.cmd tfidf\classify_tfidf.py
rem        run.cmd bert\finetune.py --model macbert --pool max
rem NOTE: keep this file ASCII-only -- cmd.exe reads .cmd as GBK and
rem       UTF-8 Chinese comments would break parsing.
setlocal
set "PY=D:\Miniconda3\envs\py312\python.exe"
if not exist "%PY%" (
  echo [x] py312 not found: %PY%
  echo     Edit PY in run.cmd to your own conda env path.
  exit /b 1
)
"%PY%" %*
exit /b %ERRORLEVEL%
