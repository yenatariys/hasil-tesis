# 🚀 Quick Start: Dashboard with DevTunnel

## One-Command Setup

```bash
./scripts/verify_and_run.sh
```

This script will:
1. ✅ Verify all prerequisites
2. ✅ Install devtunnel if needed
3. ✅ Authenticate with GitHub
4. ✅ Start the dashboard on port 8300
5. ✅ Create a private tunnel
6. ✅ Display your tunnel URL

## Manual 3-Step Setup

### Step 1: Install DevTunnel
```bash
curl -sL https://aka.ms/DevTunnelCliInstall | bash
```

### Step 2: Authenticate
```bash
devtunnel user login --provider github
```

### Step 3: Run
```bash
./scripts/setup_devtunnel.sh
```

## Your Tunnel URL

After running the setup, you'll get a URL like:
```
https://abc123xyz-8300.use.devtunnels.ms
```

**Share this URL** with users who need access. They must:
1. Have a GitHub account
2. Be logged into GitHub
3. Click the link and authenticate

## Local Access Only

If you just want to run locally without devtunnel:
```bash
pip install -r requirements.txt
streamlit run dashboard/dashboard.py
```

Access at: http://localhost:8300

## Troubleshooting

### "Command not found: devtunnel"
```bash
# Install devtunnel
curl -sL https://aka.ms/DevTunnelCliInstall | bash

# Or download manually
wget https://aka.ms/TunnelsCliDownload/linux-x64 -O devtunnel
chmod +x devtunnel
sudo mv devtunnel /usr/local/bin/
```

### "Port 8300 already in use"
```bash
# Kill the process using the port
lsof -ti:8300 | xargs kill -9

# Or use a different port
streamlit run dashboard/dashboard.py --server.port=8301
devtunnel host -p 8301 -a github
```

### "Authentication failed"
```bash
# Logout and login again
devtunnel user logout
devtunnel user login --provider github
```

### "Network error" or "Cannot resolve host"
You may be behind a firewall. Try:
- Use a different network
- Contact IT to whitelist: `*.devtunnels.ms`, `*.windows.net`
- Use a VPN

## What's Been Configured

- ✅ Port: 8300
- ✅ Authentication: GitHub only
- ✅ Anonymous access: Disabled
- ✅ HTTPS: Enabled
- ✅ Auto-start: Scripts ready

## Need More Help?

- 📘 Full guide: [DEPLOYMENT_INSTRUCTIONS.md](DEPLOYMENT_INSTRUCTIONS.md)
- 🔧 Setup details: [DEVTUNNEL_SETUP.md](DEVTUNNEL_SETUP.md)
- 📊 Status report: [DEVTUNNEL_STATUS.md](DEVTUNNEL_STATUS.md)

## Security Notes

🔒 **Your tunnel is secure:**
- GitHub authentication required
- HTTPS encrypted
- Private (not publicly listed)
- You control who has the link

⚠️ **Best practices:**
- Don't share the URL publicly
- Rotate the tunnel periodically for sensitive data
- Monitor access logs
- Revoke tunnel when done

## Next Steps

1. Run: `./scripts/verify_and_run.sh`
2. Copy the tunnel URL
3. Share with authorized users
4. They login with GitHub and access the dashboard

**That's it!** 🎉
