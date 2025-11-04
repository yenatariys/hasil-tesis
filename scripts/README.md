# Scripts Directory

This directory contains automation scripts for running the dashboard with DevTunnel.

## Available Scripts

### 1. `verify_and_run.sh` (Recommended)
**Purpose**: Comprehensive setup with automatic verification

**Features**:
- ✅ Checks all prerequisites
- ✅ Installs devtunnel if missing
- ✅ Handles authentication
- ✅ Starts dashboard automatically
- ✅ Creates tunnel with GitHub auth
- ✅ Displays and saves tunnel URL
- ✅ Interactive troubleshooting

**Usage**:
```bash
./scripts/verify_and_run.sh
```

**When to use**: First time setup or when you want full automation

---

### 2. `setup_devtunnel.sh`
**Purpose**: Quick setup assuming prerequisites are met

**Features**:
- Installs devtunnel
- Authenticates with GitHub
- Starts dashboard and tunnel
- Displays tunnel URL

**Usage**:
```bash
./scripts/setup_devtunnel.sh
```

**When to use**: When you want a simpler script or already have devtunnel installed

---

## Quick Start

```bash
# Navigate to repository root
cd hasil-tesis

# Run the comprehensive setup
./scripts/verify_and_run.sh

# Copy the tunnel URL that appears
```

---

## Script Comparison

| Feature | verify_and_run.sh | setup_devtunnel.sh |
|---------|-------------------|-------------------|
| Prerequisite checking | ✅ Yes | ❌ No |
| Interactive installation | ✅ Yes | ⚠️ Automatic |
| Error handling | ✅ Extensive | ⚠️ Basic |
| URL extraction | ✅ Automatic | ⚠️ Manual |
| Status display | ✅ Colored output | ⚠️ Basic |
| Save URL to file | ✅ Yes | ❌ No |
| Port conflict handling | ✅ Interactive | ❌ No |
| Recommended for | First use | Repeat use |

---

## What These Scripts Do

1. **Install DevTunnel** (if needed)
   - Downloads from Microsoft
   - Installs to `/usr/local/bin/`
   - Makes executable

2. **Authenticate**
   - Opens browser for GitHub login
   - Saves credentials
   - One-time setup

3. **Start Dashboard**
   - Runs Streamlit on port 8300
   - Headless mode (no browser opens)
   - Background process

4. **Create Tunnel**
   - Connects to DevTunnel service
   - Configures GitHub authentication
   - Generates secure URL
   - Forwards port 8300

5. **Display Information**
   - Shows tunnel URL
   - Shows local URLs
   - Shows process IDs
   - Provides stop instructions

---

## Configuration

Both scripts use these settings:

| Setting | Value | Description |
|---------|-------|-------------|
| Port | 8300 | Dashboard port |
| Auth | github | Authentication provider |
| Anonymous | false | Require login |
| Headless | true | No browser opens |

---

## Output Files

### `devtunnel_url.txt`
- Created by `verify_and_run.sh`
- Contains your tunnel URL
- Located in repository root
- Recreated each run

**Example**:
```
https://7j8k9m0n-8300.use.devtunnels.ms
```

---

## Stopping the Services

### If running in foreground:
```bash
# Press Ctrl+C
```

### If running in background:
```bash
# The script will show PIDs, e.g.:
# Streamlit PID: 12345
# Tunnel PID: 12346

# Kill both processes:
kill 12345 12346
```

### If you forgot the PIDs:
```bash
# Kill streamlit
pkill -f streamlit

# Kill devtunnel
pkill -f devtunnel

# Or kill processes on port 8300
lsof -ti:8300 | xargs kill -9
```

---

## Troubleshooting

### Script won't run
```bash
# Make executable
chmod +x scripts/verify_and_run.sh
chmod +x scripts/setup_devtunnel.sh

# Check line endings (if edited on Windows)
dos2unix scripts/*.sh
```

### devtunnel won't install
- Check internet connection
- Try different network
- Download manually: https://aka.ms/TunnelsCliDownload/linux-x64

### Port already in use
```bash
# Find and kill the process
lsof -ti:8300 | xargs kill -9
```

### Authentication fails
```bash
# Reset authentication
devtunnel user logout
devtunnel user login --provider github
```

---

## Advanced Usage

### Custom port
Edit `.streamlit/config.toml`:
```toml
[server]
port = 8301  # Change to desired port
```

Then in script, change `-p 8300` to `-p 8301`

### Different authentication
```bash
# Use Microsoft account instead of GitHub
devtunnel host -p 8300 -a aad --allow-anonymous false
```

### Persistent tunnel
```bash
# Create named tunnel
devtunnel create --name my-dashboard -p 8300

# List tunnels
devtunnel list

# Host specific tunnel
devtunnel host my-dashboard -a github
```

### Background persistence
```bash
# Use screen
screen -S dashboard
./scripts/verify_and_run.sh
# Ctrl+A, D to detach

# Reattach later
screen -r dashboard
```

---

## Security Notes

These scripts:
- ✅ Require GitHub authentication
- ✅ Disable anonymous access
- ✅ Use HTTPS encryption
- ✅ Create private tunnels
- ✅ Don't expose credentials

**Best practices**:
- Don't commit `devtunnel_url.txt` (it's in `.gitignore`)
- Rotate tunnels for sensitive data
- Monitor who you share URLs with
- Stop tunnels when not in use

---

## Support

- See: `../GET_YOUR_DEVTUNNEL_LINK.md` for detailed guide
- See: `../QUICK_START.md` for quick reference
- See: `../IMPLEMENTATION_SUMMARY.md` for overview
- See: https://docs.tunnel.dev/ for DevTunnel docs

---

## Examples

### Example 1: First time setup
```bash
$ ./scripts/verify_and_run.sh
# Follow prompts
# Get tunnel URL
# Share with users
```

### Example 2: Quick restart
```bash
$ ./scripts/setup_devtunnel.sh
# Already authenticated
# URL appears in output
```

### Example 3: With screen
```bash
$ screen -S dashboard
$ ./scripts/verify_and_run.sh
# Ctrl+A, D to detach
$ screen -ls  # List sessions
$ screen -r dashboard  # Reattach later
```

---

**Note**: These scripts are designed for development and demo purposes. For production deployments, consider using proper hosting services.
