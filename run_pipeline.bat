@echo off
REM Sales Forecasting System - Quick Commands
REM Run commands from: C:\Users\dp686\Desktop\sales-forecasting

echo.
echo ========================================
echo  Sales Forecasting System
echo ========================================
echo.

if "%1"=="setup" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo [SETUP] Complete!
    goto :end
)

if "%1"=="pipeline" (
    echo [PIPELINE] Running complete data science pipeline...
    call .venv\Scripts\activate.bat
    echo   1. Building dataset...
    python build_dataset.py
    echo   2. Creating features...
    python build_features.py
    echo   3. Training models...
    python train_simple.py
    echo   4. Evaluating...
    python evaluate_model.py
    echo   5. Generating insights...
    python explain_simple.py
    echo [PIPELINE] Complete!
    goto :end
)

if "%1"=="summary" (
    echo [SUMMARY] Project Status Report
    call .venv\Scripts\activate.bat
    python SUMMARY.py
    goto :end
)

if "%1"=="train" (
    echo [TRAIN] Training models...
    call .venv\Scripts\activate.bat
    python train_simple.py
    goto :end
)

if "%1"=="evaluate" (
    echo [EVALUATE] Model evaluation...
    call .venv\Scripts\activate.bat
    python evaluate_model.py
    goto :end
)

if "%1"=="explain" (
    echo [EXPLAIN] Feature importance analysis...
    call .venv\Scripts\activate.bat
    python explain_simple.py
    goto :end
)

if "%1"=="sample" (
    echo [SAMPLE] Sampling data to 10%%...
    call .venv\Scripts\activate.bat
    python sample_data.py
    goto :end
)

if "%1"=="rebuild" (
    echo [REBUILD] Full rebuild from raw data...
    call .venv\Scripts\activate.bat
    python build_dataset.py
    python build_features.py
    goto :end
)

if "%1"=="jupyter" (
    echo [JUPYTER] Starting Jupyter Lab...
    call .venv\Scripts\activate.bat
    jupyter lab
    goto :end
)

if "%1"=="dashboard" (
    echo [DASHBOARD] Starting Streamlit app...
    call .venv\Scripts\activate.bat
    streamlit run app/app.py
    goto :end
)

if "%1"=="help" (
    echo.
    echo AVAILABLE COMMANDS:
    echo.
    echo   run_pipeline.bat setup       - Setup virtual environment
    echo   run_pipeline.bat pipeline    - Run complete pipeline
    echo   run_pipeline.bat summary     - Show project summary
    echo   run_pipeline.bat train       - Train models only
    echo   run_pipeline.bat evaluate    - Evaluate models only
    echo   run_pipeline.bat explain     - Feature importance analysis
    echo   run_pipeline.bat sample      - Sample data to 10%%
    echo   run_pipeline.bat rebuild     - Rebuild from raw data
    echo   run_pipeline.bat jupyter     - Start Jupyter Lab
    echo   run_pipeline.bat dashboard   - Start Streamlit dashboard
    echo   run_pipeline.bat help        - Show this help
    echo.
    goto :end
)

REM Default: show help
echo Run: run_pipeline.bat help
echo Or: run_pipeline.bat [setup^|pipeline^|train^|evaluate^|explain^|summary]

:end
echo.
pause
