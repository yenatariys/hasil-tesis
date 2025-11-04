# DevTunnel Setup for Dashboard

## Overview
This guide explains how to run the dashboard with devtunnel to make it accessible to GitHub-authenticated users.

## Prerequisites
- devtunnel CLI installed
- GitHub account for authentication
- Python 3.8+ with requirements installed

## Installation

### Step 1: Install DevTunnel

#### Linux / macOS
```bash
curl -sL https://aka.ms/DevTunnelCliInstall | bash
```

#### Windows (PowerShell)
```powershell
Invoke-WebRequest -Uri https://aka.ms/DevTunnelCliInstall -OutFile install-devtunnel.ps1
.\install-devtunnel.ps1
```

#### Manual Download
Download from: https://aka.ms/TunnelsCliDownload/linux-x64 (Linux)
Download from: https://aka.ms/TunnelsCliDownload/win-x64 (Windows)

### Step 2: Login with GitHub
```bash
devtunnel user login --provider github
```

This will open a browser window for GitHub authentication.

### Step 3: Start the Dashboard and Tunnel

#### Option A: Use the provided script (Recommended)
```bash
cd /path/to/hasil-tesis
./scripts/setup_devtunnel.sh
```

#### Option B: Manual setup
```bash
# Terminal 1: Start the dashboard
cd /path/to/hasil-tesis
streamlit run dashboard/dashboard.py

# Terminal 2: Create and host the tunnel
devtunnel host -p 8300 -a github --allow-anonymous false
```

### Step 4: Access the Dashboard
After running the tunnel command, you'll receive a URL like:
```
https://xxxxxxxx-8300.use.devtunnels.ms
```

Share this URL with users who have GitHub accounts. They will need to authenticate with GitHub to access the dashboard.

## Configuration

### Port Configuration
The dashboard is configured to run on port 8300 (see `.streamlit/config.toml`).

### Authentication Options
- `--allow-anonymous false`: Requires GitHub authentication (default)
- `-a github`: Use GitHub as the authentication provider

### Advanced Options
```bash
# Create a persistent tunnel
devtunnel create -p 8300

# List existing tunnels
devtunnel list

# Host a specific tunnel
devtunnel host <tunnel-id>

# Delete a tunnel
devtunnel delete <tunnel-id>
```

## Troubleshooting

### Issue: "devtunnel: command not found"
- Ensure devtunnel is installed and in your PATH
- Try: `export PATH=$PATH:~/.local/bin`

### Issue: "Access denied"
- Make sure you're logged in: `devtunnel user login --provider github`
- Verify your GitHub account has necessary permissions

### Issue: "Port already in use"
- Check if Streamlit is already running: `ps aux | grep streamlit`
- Kill the process or use a different port

### Issue: Network restrictions in CI/CD
- DevTunnel requires access to Microsoft Azure services
- In restricted environments (like GitHub Actions), you may need to:
  - Whitelist *.devtunnels.ms and *.windows.net domains
  - Use alternative tunneling solutions
  - Run the setup on a local machine or unrestricted environment

## Security Considerations

1. **GitHub Authentication**: The tunnel requires GitHub authentication, ensuring only authorized users can access
2. **HTTPS**: All traffic is encrypted via HTTPS
3. **Temporary URLs**: Tunnel URLs are temporary and can be revoked
4. **Access Logs**: DevTunnel provides access logs for monitoring

## Additional Resources
- [DevTunnel Documentation](https://docs.tunnel.dev/)
- [GitHub CLI Documentation](https://cli.github.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Notes
- The tunnel URL is temporary and will expire after the tunnel process is stopped
- For production deployments, consider using Azure App Service, AWS, or other hosting platforms
- This setup is ideal for development, demos, and temporary sharing
