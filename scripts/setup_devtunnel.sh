#!/bin/bash

# Script to setup and run dashboard with devtunnel
# This script will:
# 1. Install devtunnel if not already available
# 2. Configure the tunnel with GitHub authentication
# 3. Start the Streamlit dashboard on port 8300
# 4. Create a private tunnel accessible only to GitHub users

set -e

echo "Setting up devtunnel for dashboard..."

# Check if devtunnel is installed
if ! command -v devtunnel &> /dev/null; then
    echo "Installing devtunnel..."
    
    # Download devtunnel for Linux x64
    DEVTUNNEL_URL="https://aka.ms/TunnelsCliDownload/linux-x64"
    
    # Try to download
    if ! curl -L "$DEVTUNNEL_URL" -o /tmp/devtunnel; then
        echo "ERROR: Failed to download devtunnel. Network restrictions may be preventing access."
        echo "Please manually download devtunnel from: https://aka.ms/TunnelsCliDownload/linux-x64"
        echo "Or visit: https://docs.tunnel.dev/ for installation instructions"
        exit 1
    fi
    
    # Make executable and move to path
    chmod +x /tmp/devtunnel
    sudo mv /tmp/devtunnel /usr/local/bin/
    
    echo "devtunnel installed successfully"
fi

# Verify installation
devtunnel --version

echo "Logging in to devtunnel with GitHub..."
# Login with GitHub (requires interactive authentication on first run)
devtunnel user login --provider github || {
    echo "Please run: devtunnel user login --provider github"
    echo "Then re-run this script"
    exit 1
}

echo "Creating tunnel on port 8300 with GitHub authentication..."
# Create a tunnel with GitHub authentication
# -a github: Require GitHub authentication
# -p 8300: Port to forward
devtunnel host -p 8300 -a github &
TUNNEL_PID=$!

echo "Starting Streamlit dashboard on port 8300..."
cd "$(dirname "$0")/.."
streamlit run dashboard/dashboard.py &
STREAMLIT_PID=$!

echo ""
echo "Dashboard is running!"
echo "devtunnel PID: $TUNNEL_PID"
echo "Streamlit PID: $STREAMLIT_PID"
echo ""
echo "To stop the services:"
echo "  kill $TUNNEL_PID $STREAMLIT_PID"
echo ""

# Wait for both processes
wait
