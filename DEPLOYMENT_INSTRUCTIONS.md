# Dashboard Deployment with DevTunnel - Complete Instructions

## Current Status

✅ **Dashboard Configuration**: Updated to run on port 8300
✅ **Streamlit Setup**: Successfully tested and running
⚠️  **DevTunnel Installation**: Requires manual setup due to network restrictions

## Quick Start (Local Machine)

### 1. Install Dependencies
```bash
cd hasil-tesis
pip install -r requirements.txt
```

### 2. Install DevTunnel
```bash
# Linux/macOS
curl -sL https://aka.ms/DevTunnelCliInstall | bash

# Windows PowerShell
iwr https://aka.ms/DevTunnelCliInstall | iex
```

### 3. Login with GitHub
```bash
devtunnel user login --provider github
```

### 4. Start Everything
```bash
# Use the automated script
./scripts/setup_devtunnel.sh

# OR manually in two terminals:

# Terminal 1: Start Streamlit
streamlit run dashboard/dashboard.py

# Terminal 2: Create tunnel
devtunnel host -p 8300 -a github --allow-anonymous false
```

### 5. Share the URL
DevTunnel will display a URL like:
```
https://abc123xyz-8300.use.devtunnels.ms
```

Only users authenticated with GitHub can access this URL.

## Configuration Details

### Streamlit Configuration
File: `.streamlit/config.toml`
```toml
[server]
runOnSave = true
port = 8300
```

### DevTunnel Authentication Options

#### GitHub-only access (Recommended)
```bash
devtunnel host -p 8300 -a github --allow-anonymous false
```

#### Microsoft Account access
```bash
devtunnel host -p 8300 -a aad --allow-anonymous false
```

#### Multiple providers
```bash
devtunnel host -p 8300 -a github -a aad --allow-anonymous false
```

## Advanced Usage

### Create Persistent Tunnel
```bash
# Create tunnel
devtunnel create --name hasil-tesis-dashboard -p 8300

# List tunnels
devtunnel list

# Host the tunnel
devtunnel host hasil-tesis-dashboard -a github
```

### Monitoring and Logs
```bash
# View tunnel logs
devtunnel logs

# Check tunnel status
devtunnel show <tunnel-id>
```

## Alternative Deployment Options

If DevTunnel is not suitable for your use case, consider these alternatives:

### 1. GitHub Codespaces
```bash
# Forward port in Codespace
gh codespace ports forward 8300:8300
```

### 2. Streamlit Cloud
- Push to GitHub
- Connect at share.streamlit.io
- Configure authentication in Streamlit Cloud settings

### 3. Docker + Cloud Run
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.port=8300"]
```

### 4. Azure Web Apps
- Deploy directly from GitHub
- Configure authentication with Azure AD

### 5. Heroku
```bash
echo "web: streamlit run dashboard/dashboard.py --server.port=\$PORT" > Procfile
git push heroku main
```

## Security Best Practices

1. **Always use authentication**: Never expose the dashboard publicly without auth
2. **HTTPS only**: Ensure all traffic is encrypted
3. **Monitor access**: Review access logs regularly
4. **Rotate tunnels**: For sensitive data, create new tunnels periodically
5. **Limit permissions**: Only give access to users who need it

## Troubleshooting

### DevTunnel Not Installing
**Issue**: Cannot download devtunnel in restricted network

**Solutions**:
1. Use a different network without restrictions
2. Download manually from https://aka.ms/TunnelsCliDownload/linux-x64
3. Contact your IT department to whitelist:
   - *.devtunnels.ms
   - *.windows.net
   - aka.ms

### Dashboard Not Starting
**Issue**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

**Issue**: Port already in use

**Solution**:
```bash
# Find and kill the process
lsof -ti:8300 | xargs kill -9

# Or use a different port
streamlit run dashboard/dashboard.py --server.port=8301
```

### Authentication Failed
**Issue**: Cannot login to devtunnel

**Solution**:
```bash
# Clear cached credentials
devtunnel user logout
devtunnel user login --provider github

# Ensure browser can access GitHub
# Check if 2FA is required
```

### Tunnel Disconnects
**Issue**: Tunnel closes unexpectedly

**Solution**:
```bash
# Run in a screen session
screen -S dashboard
./scripts/setup_devtunnel.sh
# Ctrl+A, D to detach

# Or use nohup
nohup devtunnel host -p 8300 -a github &
```

## CI/CD Considerations

### GitHub Actions
DevTunnel installation in GitHub Actions requires:
1. Whitelisting Microsoft Azure domains
2. Using a self-hosted runner with network access
3. Or using GitHub Codespaces as the runtime

### Example Workflow
```yaml
name: Deploy Dashboard
on: workflow_dispatch

jobs:
  deploy:
    runs-on: self-hosted  # or ubuntu-latest with network access
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Install DevTunnel
        run: curl -sL https://aka.ms/DevTunnelCliInstall | bash
      - name: Start Tunnel
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          devtunnel user login --provider github --token $GITHUB_TOKEN
          devtunnel host -p 8300 -a github &
          streamlit run dashboard/dashboard.py
```

## Getting Help

- **DevTunnel Docs**: https://docs.tunnel.dev/
- **Streamlit Docs**: https://docs.streamlit.io/
- **GitHub Issues**: Open an issue in this repository
- **Stack Overflow**: Tag your question with `devtunnel` and `streamlit`

## Next Steps

1. ✅ Dashboard configured for port 8300
2. ✅ Setup scripts created
3. 📝 Install devtunnel on your local machine
4. 🚀 Run the dashboard and create tunnel
5. 🔗 Share the devtunnel URL with authorized users

---

**Note**: This setup requires manual execution on a machine with proper network access. The GitHub Actions environment has network restrictions that prevent automatic devtunnel installation.
