# 🚀 EduInfra Ghana — Deployment Guide

> **For:** Ing. Joseph K. Mensah (PE-GhIE) · Brainhauz Solutions  
> **Platform:** Streamlit Community Cloud (app) + Netlify (landing page)  
> **Time to deploy:** ~15 minutes

---

## Part A — Push to GitHub

### 1. Create a GitHub repository

1. Go to [github.com/new](https://github.com/new)
2. Name it `eduinfra-ghana` (or any name you prefer)
3. Set it to **Public** (required for free Streamlit Community Cloud)
4. Do **not** initialise with a README (you already have one)
5. Click **Create repository**

### 2. Check your .gitignore

Open `C:\Dev\eduinfra-ghana\.gitignore` and confirm these lines are present
(they are — the file was updated in Stage 5):

```
.env
.venv/
__pycache__/
*.pkl
*.pyc
.streamlit/secrets.toml
logs/*.log
```

> ⚠️ Never commit `.env`, `.venv/`, `.pkl` model files, or your `secrets.toml`.

### 3. Initialise and commit

Open PowerShell in `C:\Dev\eduinfra-ghana\` and run:

```powershell
git init
git add .
git commit -m "feat: EduInfra Ghana — full platform v2 submission"
```

### 4. Connect to GitHub and push

```powershell
git remote add origin https://github.com/[your-username]/eduinfra-ghana.git
git branch -M main
git push -u origin main
```

Replace `[your-username]` with your GitHub username.

### 5. Verify the push

Visit `https://github.com/[your-username]/eduinfra-ghana` and confirm all
files appear — especially `app.py`, `src/`, `data/schools_priority_ranked.csv`,
and `requirements.txt`. The `models/*.pkl` and `.venv/` folders should **not**
be present.

---

## Part B — Deploy on Streamlit Community Cloud

### 1. Sign in to Streamlit Cloud

Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
the same GitHub account you used above.

### 2. Create a new app

1. Click **"New app"**
2. Under **Repository**, select `[your-username]/eduinfra-ghana`
3. Under **Branch**, select `main`
4. Under **Main file path**, type: `app.py`
5. Leave the **App URL** as the suggested default (e.g., `eduinfra-ghana.streamlit.app`)
   or customise it — this becomes your public URL

### 3. Add your ANTHROPIC_API_KEY secret

Before clicking Deploy:

1. Click **"Advanced settings"**
2. In the **Secrets** text box, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-api03-..."
   ```
   Replace the value with your real Anthropic API key from
   [console.anthropic.com](https://console.anthropic.com)
3. Click **Save**

> 🔒 Secrets are encrypted and never visible to other users or in logs.

### 4. Deploy

Click **"Deploy!"**. Streamlit will:
- Install packages from `requirements.txt` (takes 3–5 minutes on first deploy)
- Launch the app at your chosen URL

Once the spinner stops, your app is live. Copy the URL — you'll need it next.

> **Tip:** If the first deployment fails due to a missing package, add it to
> `requirements.txt`, commit, and push — Streamlit will redeploy automatically.

---

## Part C — Update the Netlify redirect

### 1. Update netlify.toml

Open `C:\Dev\eduinfra-ghana\netlify\netlify.toml` and replace the placeholder
URL on the `to =` line with your real Streamlit URL:

```toml
[[redirects]]
from = "/app"
to = "https://YOUR-REAL-APP.streamlit.app"   # ← update this
status = 302
force = true
```

### 2. Update index.html

Open `C:\Dev\eduinfra-ghana\netlify\index.html` and find the Launch button:

```html
<a href="https://eduinfra-ghana.streamlit.app" class="launch-btn" ...>
```

Replace `https://eduinfra-ghana.streamlit.app` with your real URL.

Also update the GitHub link near the bottom:

```html
<a href="https://github.com/[user]/eduinfra-ghana" ...>
```

Replace `[user]` with your actual GitHub username.

### 3. Commit and push

```powershell
git add netlify/
git commit -m "chore: update netlify URLs with live app address"
git push
```

---

## Part D — Deploy to Netlify (drag-and-drop, no CLI needed)

### 1. Go to Netlify

Visit [netlify.com](https://www.netlify.com) and sign in (or create a free account).

### 2. Create a new site

On the dashboard, scroll to the bottom and find:

> **"Deploy manually — drag and drop your site folder here"**

### 3. Drag and drop

Open Windows Explorer and navigate to `C:\Dev\eduinfra-ghana\`.

Drag the **`netlify`** folder (the whole folder, not its contents) and drop it
onto the Netlify drag-and-drop zone.

Netlify will upload the files and give you a random URL like
`https://random-name-123.netlify.app` in about 10 seconds.

### 4. Set a custom domain (optional)

In the Netlify dashboard:
1. Click **"Domain settings"** → **"Add custom domain"**
2. Enter your domain (e.g., `eduinfra.brainhauz.com`)
3. Follow the DNS instructions for your domain registrar

### 5. Rename your Netlify site

In **Site settings → General → Site name**, rename it to `eduinfra-ghana`
so your URL becomes `https://eduinfra-ghana.netlify.app`.

---

## Quick Reference — Live URLs

| Service | URL |
|---|---|
| **Streamlit app** | `https://eduinfra-ghana.streamlit.app` *(update after deploy)* |
| **Netlify landing** | `https://eduinfra-ghana.netlify.app` *(update after deploy)* |
| **GitHub repo** | `https://github.com/[user]/eduinfra-ghana` |
| **Netlify /app redirect** | `https://eduinfra-ghana.netlify.app/app` → Streamlit |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| App crashes on startup | Check Streamlit Cloud logs → usually a missing package in `requirements.txt` |
| AI chat shows "Set ANTHROPIC_API_KEY" | Add the secret in Streamlit Cloud → App settings → Secrets |
| `shap` install fails on Cloud | Add `shap>=0.45.0` to `requirements.txt` and redeploy |
| Pipeline button errors | The pipeline needs the DHS `.DTA` file — upload it to `data/dhs/` and commit |
| Netlify shows 404 | Confirm `netlify/index.html` exists and you dragged the `netlify/` folder |

---

*EduInfra Ghana · Built by Ing. Joseph K. Mensah (PE-GhIE) · Brainhauz Solutions*
