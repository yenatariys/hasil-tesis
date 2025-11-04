#!/bin/bash

# Comprehensive verification and setup script for dashboard with devtunnel
# This script checks all prerequisites and guides through the setup process

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Dashboard DevTunnel Setup Verification${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found"
    echo "Please install Python 3.8+"
    exit 1
fi

# Check pip
echo -n "Checking pip... "
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC} pip not found"
    echo "Please install pip"
    exit 1
fi

# Check required Python packages
echo -n "Checking Python dependencies... "
if pip3 list | grep -q streamlit; then
    echo -e "${GREEN}✓${NC} Streamlit installed"
else
    echo -e "${YELLOW}!${NC} Streamlit not found"
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check data files
echo -n "Checking data files... "
if [ -f "data/lex_labeled_review_app.csv" ] && [ -f "data/lex_labeled_review_play.csv" ]; then
    echo -e "${GREEN}✓${NC} Data files present"
elif [ -f "lex_labeled_review_app.csv" ] && [ -f "lex_labeled_review_play.csv" ]; then
    echo -e "${GREEN}✓${NC} Data files present (root directory)"
else
    echo -e "${YELLOW}!${NC} Data files not found"
    echo "Dashboard may not work without data files"
fi

# Check Streamlit config
echo -n "Checking Streamlit configuration... "
if [ -f ".streamlit/config.toml" ]; then
    if grep -q "port = 8300" .streamlit/config.toml; then
        echo -e "${GREEN}✓${NC} Port 8300 configured"
    else
        echo -e "${YELLOW}!${NC} Port not set to 8300"
    fi
else
    echo -e "${RED}✗${NC} Config file not found"
fi

# Check devtunnel
echo -n "Checking devtunnel... "
if command -v devtunnel &> /dev/null; then
    DEVTUNNEL_VERSION=$(devtunnel --version 2>&1 | head -1)
    echo -e "${GREEN}✓${NC} $DEVTUNNEL_VERSION"
    DEVTUNNEL_INSTALLED=true
else
    echo -e "${RED}✗${NC} Not installed"
    DEVTUNNEL_INSTALLED=false
fi

echo ""
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Setup Options${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

if [ "$DEVTUNNEL_INSTALLED" = false ]; then
    echo -e "${YELLOW}DevTunnel is not installed.${NC}"
    echo ""
    echo "To install devtunnel, run:"
    echo "  curl -sL https://aka.ms/DevTunnelCliInstall | bash"
    echo ""
    echo "Or download manually from:"
    echo "  https://aka.ms/TunnelsCliDownload/linux-x64"
    echo ""
    
    read -p "Would you like to install devtunnel now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing devtunnel..."
        curl -sL https://aka.ms/DevTunnelCliInstall | bash
        
        # Check if installation succeeded
        if command -v devtunnel &> /dev/null; then
            echo -e "${GREEN}✓${NC} DevTunnel installed successfully"
            DEVTUNNEL_INSTALLED=true
        else
            echo -e "${RED}✗${NC} Installation failed. Please install manually."
            exit 1
        fi
    else
        echo ""
        echo "You can run the dashboard locally without devtunnel:"
        echo "  streamlit run dashboard/dashboard.py"
        echo ""
        echo "To use devtunnel later, install it and run this script again."
        exit 0
    fi
fi

# Check devtunnel authentication
echo ""
echo "Checking devtunnel authentication..."
if devtunnel user show &> /dev/null; then
    USER_INFO=$(devtunnel user show 2>&1 | grep -E "User:|Email:" | head -1)
    echo -e "${GREEN}✓${NC} Authenticated: $USER_INFO"
else
    echo -e "${YELLOW}!${NC} Not authenticated"
    echo ""
    echo "Logging in with GitHub..."
    devtunnel user login --provider github
    
    if devtunnel user show &> /dev/null; then
        echo -e "${GREEN}✓${NC} Authentication successful"
    else
        echo -e "${RED}✗${NC} Authentication failed"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}Starting Dashboard and Tunnel${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check if port 8300 is available
if lsof -Pi :8300 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}!${NC} Port 8300 is already in use"
    echo ""
    read -p "Kill the process and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:8300 | xargs kill -9
        echo "Process killed"
    else
        echo "Please free port 8300 and run this script again"
        exit 1
    fi
fi

# Start dashboard in background
echo "Starting Streamlit dashboard on port 8300..."
streamlit run dashboard/dashboard.py --server.headless=true &
STREAMLIT_PID=$!

# Wait for dashboard to start
echo -n "Waiting for dashboard to initialize"
for i in {1..10}; do
    if curl -s http://localhost:8300 > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Create devtunnel
echo "Creating devtunnel with GitHub authentication..."
echo ""

# Create tunnel with output capture
TUNNEL_OUTPUT=$(mktemp)
devtunnel host -p 8300 -a github --allow-anonymous false 2>&1 | tee "$TUNNEL_OUTPUT" &
TUNNEL_PID=$!

# Wait for tunnel URL to appear
echo -n "Waiting for tunnel URL"
for i in {1..20}; do
    if grep -q "https://" "$TUNNEL_OUTPUT" 2>/dev/null; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Extract and display URL
TUNNEL_URL=$(grep -o 'https://[^ ]*\.devtunnels\.ms[^ ]*' "$TUNNEL_OUTPUT" | head -1)

if [ -n "$TUNNEL_URL" ]; then
    echo ""
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}✓ Dashboard is running!${NC}"
    echo -e "${GREEN}=====================================${NC}"
    echo ""
    echo -e "${BLUE}Tunnel URL:${NC}"
    echo -e "${GREEN}$TUNNEL_URL${NC}"
    echo ""
    echo -e "${BLUE}Local URLs:${NC}"
    echo "  http://localhost:8300"
    echo ""
    echo -e "${BLUE}Access:${NC}"
    echo "  - GitHub authentication required"
    echo "  - Share the tunnel URL with authorized users"
    echo ""
    echo -e "${BLUE}Process IDs:${NC}"
    echo "  Streamlit: $STREAMLIT_PID"
    echo "  Tunnel: $TUNNEL_PID"
    echo ""
    echo -e "${BLUE}To stop the services:${NC}"
    echo "  kill $STREAMLIT_PID $TUNNEL_PID"
    echo ""
    echo -e "${BLUE}Logs:${NC}"
    echo "  Tunnel: $TUNNEL_OUTPUT"
    echo ""
    echo "Press Ctrl+C to stop all services"
    echo ""
    
    # Save URL to file
    echo "$TUNNEL_URL" > devtunnel_url.txt
    echo "Tunnel URL saved to: devtunnel_url.txt"
    
    # Wait for user interrupt
    trap "echo ''; echo 'Stopping services...'; kill $STREAMLIT_PID $TUNNEL_PID 2>/dev/null; rm -f $TUNNEL_OUTPUT; echo 'Done.'; exit 0" INT TERM
    wait
else
    echo -e "${RED}✗${NC} Failed to create tunnel"
    echo "Check the output above for errors"
    kill $STREAMLIT_PID 2>/dev/null
    rm -f "$TUNNEL_OUTPUT"
    exit 1
fi
