<#
Run the Streamlit dashboard with sensible defaults.
This wrapper exists to make it easy to start the app with auto-reload enabled.
#>
param(
    [int]
    $Port = 8502
)

Write-Host "Starting Streamlit dashboard on port $Port (runOnSave enabled if supported by Streamlit)"

# Activate venv if present
if (Test-Path -Path .venv\Scripts\Activate.ps1) {
    Write-Host "Activating .venv virtual environment..."
    . .\.venv\Scripts\Activate.ps1
}

# Start Streamlit; runOnSave will also be used if present in .streamlit/config.toml
python -m streamlit run .\dashboard.py --server.port $Port
