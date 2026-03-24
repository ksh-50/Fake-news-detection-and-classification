<p align="center">
  <img src="screenshots/landing.png" width="700" alt="Text to Truth — Landing Page" />
</p>

<h1 align="center">🛡️ Text to Truth</h1>

<p align="center">
  <b>Paste a headline. Get the truth.</b><br>
  Built by <a href="https://github.com/Piyush01-tech">Team Text to Truth</a>
</p>

<p align="center">
  <a href="https://github.com/Piyush01-tech/Text-to-Truth"><img src="https://img.shields.io/badge/repo-Text--to--Truth-6C63FF?style=flat-square&logo=github" alt="GitHub Repo" /></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/streamlit-latest-FF4B4B?style=flat-square" alt="Streamlit" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" />
</p>

---

## What This Does

You paste any news headline. The app checks two things:

1. **ML Model** — trained on 44,000+ real and fake articles, it looks at the language patterns
2. **Live Web Search** — searches DuckDuckGo in real-time and checks if BBC, Reuters, AP, CNN, or any of 40+ trusted outlets are reporting the same story

If credible sources back it up → ✅ **Verified Real**

If nobody credible is covering it → ⚠️ **Likely Fake**

If it's unclear → 🔍 **Unverified**

---

## See It In Action

<table>
  <tr>
    <td align="center"><b>✅ Real headline detected</b></td>
    <td align="center"><b>⚠️ Fake headline caught</b></td>
  </tr>
  <tr>
    <td><img src="screenshots/verified-real.png" width="450" /></td>
    <td><img src="screenshots/likely-fake.png" width="450" /></td>
  </tr>
</table>

---

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── streamlit_app.py       → The main web app
│   ├── web_verify.py          → Web search + credibility engine
│   ├── train_model.py         → Model training script
│   ├── detect_fake_news.py    → CLI tool
│   ├── text_clean.py          → Text preprocessing
│   └── utils.py               → I/O helpers
├── data/                      → Training data
├── outputs/                   → Trained model files
└── requirements.txt
```

---

## How It Works Under The Hood

```
Headline → ML Model (TF-IDF + Logistic Regression) → fake probability
         → DuckDuckGo Search → relevance check → debunk detection → web score
         → Composite: 40% ML + 60% web evidence → final verdict
```

The web engine doesn't just check if a domain appears in results — it verifies that search results actually **match the headline content** and flags any results containing debunking language ("hoax", "false", "rumor", etc.).

---

## Tech Stack

Python · Streamlit · scikit-learn · DuckDuckGo Search · joblib

---

<p align="center">
  <b>MIT License</b> · Built with ❤️ by Team Text to Truth</a>
</p>
