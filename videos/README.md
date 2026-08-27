# 🎬 Promotional Videos & Automated Playwright Recording

This folder contains high-definition recorded video walkthroughs of the **AI Trading Bot** platform, formatted specifically for social media advertising campaigns on **Facebook** and **LinkedIn**.

> **Generated videos are not committed to git** (`videos/demo_tour/*.mp4`, `*.webm`
> are gitignored — they were previously tracked and bloated every clone by
> ~29 MB). Run the recording scripts below to (re)generate them locally, or
> host a copy externally (e.g. a GitHub Release asset or cloud storage) and
> link it here if you need a persistently shared version.

---

## 📁 Repository Quick Links

* 📄 **Marketing Ad Copy & All-Pages Storyboard (English & Français):** [`MARKETING_AD_SCENARIOS.md`](./MARKETING_AD_SCENARIOS.md)
* 📄 **Chart Features Guide & French Subtitle Scenario (Français):** [`CHART_WORKFLOW_SCENARIO_FR.md`](./CHART_WORKFLOW_SCENARIO_FR.md)
* 🛠️ **All-Pages Platform Walkthrough Recording Script:** [`scripts/record_demo_tour.py`](../scripts/record_demo_tour.py)
* 🛠️ **Chart Features & French Subtitles Recording Script:** [`scripts/record_chart_features.py`](../scripts/record_chart_features.py)
* 🎥 **Generated All-Pages MP4 Advertisement Video (~15.5 MB):** [`demo_tour/AI_Trading_Bot_Platform_Walkthrough_90s.mp4`](./demo_tour/AI_Trading_Bot_Platform_Walkthrough_90s.mp4)
* 🎥 **Generated Chart Features French MP4 Video:** [`demo_tour/AI_Trading_Bot_Chart_Features_FR.mp4`](./demo_tour/AI_Trading_Bot_Chart_Features_FR.mp4)

---

## 🚀 How to Record & Generate the Walkthrough Video

The platform includes an automated Python script powered by **Playwright** that launches a Chromium browser in Full-HD (1920x1080), navigates across all frontend modules with human-like scrolling and pausing, and automatically converts the recorded session into an optimized `.mp4` video via FFmpeg.

### 1. Prerequisites
Ensure you have the Python dependencies and system tools installed:
```bash
pip install playwright
playwright install chromium

# Verify ffmpeg is available (used for automatic WebM to MP4 conversion)
which ffmpeg
```

### 2. Launch the Development Servers
Before running the tour, ensure your Next.js frontend (and FastAPI backend) server is online. By default, the script looks for the frontend on port `3001` (or `3000` depending on your `.env` configuration):

```bash
# Start backend and frontend via the canonical Makefile
make dev
```
*(Verify your server is responding at http://localhost:3001 or http://localhost:3000)*

### 3. Execute the Automated Video Recording
Open a separate terminal unit and execute the Playwright script:

```bash
python3 scripts/record_demo_tour.py
```

> 💡 **Customizing the Port:** If your server runs on a custom URL or port, pass it via an environment variable:
> ```bash
> FRONTEND_URL=http://localhost:3000 python3 scripts/record_demo_tour.py
> ```
> 💡 **Headless vs. Visual Mode:** By default, the script runs with `headless=True` for rock-solid stability in terminal environments. If you want to visually watch the automated mouse moves and clicks on your screen while recording, open [`scripts/record_demo_tour.py`](../scripts/record_demo_tour.py#L58) and set `headless=False`.

---

## 📍 Automated Tour Walkthrough Roadmap (90 Seconds)

When executed, the script automatically traverses the following 7 scenes in real-time:
1. **`00:00` | Dashboard & Charts (`/`):** Smooth mouse hovering over live XAUUSD candlesticks and timeframe switching (M5, M15, H1).
2. **`00:15` | AI Strategies (`/strategies`):** Navigating to the AI PDF-to-Strategy Compiler and PDF dropzone.
3. **`00:30` | AI Self-Refinement (`/ai-reports`):** Scrolling through the 10-Trade Cycle evaluation and hyper-parameter optimization reports.
4. **`00:45` | Bots & Account Management (`/bots` -> `/bot-control`):** Showcasing multi-account cards and the risk-free **Paper Trading** mode toggle.
5. **`00:58` | Backtester & Analytics (`/backtest` -> `/analytics`):** Inspecting historical simulation runs, win-rate distributions (68.4%), and Sharpe Ratios.
6. **`01:12` | News Shield & Risk Caps (`/news` -> `/settings`):** Verifying automatic trading pauses before major high-impact announcements (CPI, NFP, Fed) and Daily Loss Limit circuit breakers.
7. **`01:25` | Outro (`/`):** Return to the primary XAUUSD live chart dashboard.

---

## 📢 Using Your Video in Ad Campaigns

Once the script finishes running, grab your optimized video at:
**`videos/demo_tour/AI_Trading_Bot_Platform_Walkthrough_90s.mp4`**

Pair this video directly with our tested long-form & short-form ad copy (available in both **English 🇬🇧** and **French 🇫🇷**) found in [`MARKETING_AD_SCENARIOS.md`](./MARKETING_AD_SCENARIOS.md) when publishing on **LinkedIn** (institutional/quant focus) or **Facebook** (proactive retail Gold/XAUUSD focus).
