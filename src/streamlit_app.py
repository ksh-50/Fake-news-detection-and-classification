import argparse
import sys
from pathlib import Path
import joblib
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_clean import clean_text
from web_verify import corroborate


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths():
    out = project_root() / "outputs"
    return {
        "pipeline": out / "pipeline.joblib",
        "model": out / "model.joblib",
        "vectorizer": out / "vectorizer.joblib",
    }


def load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path):
    try:
        if pipeline_path and pipeline_path.exists():
            return joblib.load(pipeline_path), None, None
        if model_path.exists() and vectorizer_path.exists():
            return None, joblib.load(model_path), joblib.load(vectorizer_path)
    except Exception as exc:
        st.error(f"**Failed to load model files:** {exc}")
        st.stop()
    return None, None, None


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp, .stApp * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: #0B0F19;
}

#MainMenu, header, footer, .stDeployButton { display: none !important; }

.brand-header {
    text-align: center;
    padding: 3rem 0 0.5rem;
}
.brand-name {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6C63FF;
    margin-bottom: 0.8rem;
}
.brand-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: #F1F1F4;
    margin: 0;
    letter-spacing: -1px;
    line-height: 1.15;
}
.brand-header .tagline {
    color: #5A5E6E;
    font-size: 0.92rem;
    font-weight: 400;
    margin-top: 0.6rem;
    line-height: 1.5;
}

.subtle-divider {
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, #6C63FF, #a78bfa);
    border-radius: 2px;
    margin: 1.8rem auto;
}

.verdict-container {
    text-align: center;
    padding: 2rem 0 1rem;
    animation: fadeUp 0.4s ease;
}
.verdict-emoji {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.verdict-label-real {
    font-size: 1.8rem;
    font-weight: 700;
    color: #34D399;
}
.verdict-label-fake {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F87171;
}
.verdict-subtitle {
    font-size: 0.9rem;
    color: #5A5E6E;
    margin-top: 0.3rem;
}

.track-container {
    max-width: 480px;
    margin: 1rem auto 0.4rem;
}
.track-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #5A5E6E;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.track {
    height: 6px;
    background: #1E2230;
    border-radius: 3px;
    overflow: hidden;
}
.track-fill-real {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #059669, #34D399);
    transition: width 0.6s ease;
}
.track-fill-fake {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #DC2626, #F87171);
    transition: width 0.6s ease;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 1.5rem 0;
    margin: 0.5rem 0;
}
.stat {
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F1F4;
}
.stat-label {
    font-size: 0.7rem;
    color: #4A4E5E;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.15rem;
}

.sources-section {
    text-align: center;
    padding: 0.5rem 0 1rem;
}
.sources-title {
    font-size: 0.7rem;
    color: #4A4E5E;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.6rem;
}
.source-pill {
    display: inline-block;
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.18);
    border-radius: 100px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    color: #A5A0FF;
    margin: 0.2rem;
    font-weight: 500;
}
.no-sources-msg {
    color: #5A5E6E;
    font-size: 0.82rem;
    font-style: italic;
}

.stButton > button {
    background: #6C63FF !important;
    color: white !important;
    border: none !important;
    padding: 0.7rem 3rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #5B52E6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(108, 99, 255, 0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

.stTextArea textarea {
    background: #12162A !important;
    border: 1px solid #1E2338 !important;
    border-radius: 10px !important;
    color: #E4E4E8 !important;
    font-size: 0.95rem !important;
    padding: 1rem 1.2rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea::placeholder {
    color: #3A3E4E !important;
}
.stTextArea textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.12) !important;
}

.custom-details {
    border: 1px solid #1E2338;
    border-radius: 8px;
    margin: 1rem 0;
    overflow: hidden;
}
.custom-details summary {
    cursor: pointer;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #5A5E6E;
    background: transparent;
    list-style: none;
    user-select: none;
}
.custom-details summary::-webkit-details-marker {
    display: none;
}
.custom-details summary::before {
    content: "▸ ";
    color: #4A4E5E;
}
.custom-details[open] summary::before {
    content: "▾ ";
}
.custom-details .details-content {
    background: #0E1221;
    border-top: 1px solid #1E2338;
    padding: 0.8rem 1rem;
}

div[data-testid="stExpander"] {
    display: none !important;
}

.detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    margin: 0.5rem 0;
}
.detail-table th {
    text-align: left;
    color: #4A4E5E;
    font-weight: 600;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #1E2338;
}
.detail-table td {
    color: #8A8E9E;
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid rgba(30,35,56,0.5);
}
.detail-table td:nth-child(2) {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    color: #A5A0FF;
}

.app-footer {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-top: 1px solid #1A1E2E;
    margin-top: 2rem;
}
.footer-brand {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3A3E4E;
}
.footer-brand span {
    color: #6C63FF;
}

section[data-testid="stSidebar"] {
    background: #0E1221 !important;
    border-right: 1px solid #1A1E2E !important;
}
.sidebar-status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.35rem 0;
    font-size: 0.82rem;
    color: #8A8E9E;
}
.dot-ok {
    width: 7px; height: 7px; border-radius: 50%;
    background: #34D399;
    box-shadow: 0 0 6px rgba(52,211,153,0.4);
}
.dot-err {
    width: 7px; height: 7px; border-radius: 50%;
    background: #F87171;
    box-shadow: 0 0 6px rgba(248,113,113,0.4);
}

.scanning-msg {
    text-align: center;
    color: #6C63FF;
    font-size: 0.85rem;
    font-weight: 500;
    animation: pulse 1.5s infinite;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
"""


def main():
    dp = default_paths()

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pipeline", default=str(dp["pipeline"]))
    ap.add_argument("--model", default=str(dp["model"]))
    ap.add_argument("--vectorizer", default=str(dp["vectorizer"]))
    args, _ = ap.parse_known_args()

    pipeline_path = Path(args.pipeline).resolve()
    model_path = Path(args.model).resolve()
    vectorizer_path = Path(args.vectorizer).resolve()

    st.set_page_config(
        page_title="Text to Truth · Fake News Detector",
        page_icon="🛡️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("#### System")
        for name, path in [("Pipeline", pipeline_path), ("Model", model_path), ("Vectorizer", vectorizer_path)]:
            ok = path.exists()
            dot = "dot-ok" if ok else "dot-err"
            st.markdown(f'<div class="sidebar-status-item"><div class="{dot}"></div>{name}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption("ML model analyzes linguistic patterns. Web search verifies against Reuters, AP, BBC, and 40+ outlets.")

    pipe, clf, vec = load_pipeline_or_parts(pipeline_path, model_path, vectorizer_path)
    if pipe is None and (clf is None or vec is None):
        st.error("**Model not found.** Train first:\n\n`python src/train_model.py --real data/True.csv --fake data/Fake.csv --outdir outputs`")
        st.stop()

    st.markdown("""
    <div class="brand-header">
        <div class="brand-name">From Text to Truth</div>
        <h1>Fake News Detector</h1>
        <div class="tagline">Paste any headline. We'll cross-check it against trusted sources worldwide.</div>
    </div>
    <div class="subtle-divider"></div>
    """, unsafe_allow_html=True)

    txt = st.text_area(
        "Enter headline or article",
        height=120,
        placeholder="Paste a news headline here…",
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([1.2, 1, 1.2])
    with col2:
        analyze = st.button("Analyze", use_container_width=True)

    if analyze and not txt.strip():
        st.warning("Please paste a headline or article text to analyze.")

    if analyze and txt.strip():
        s = clean_text(txt)
        if pipe is not None:
            ml_prob_fake = float(pipe.predict_proba([s])[0, 1])
        else:
            X = vec.transform([s])
            ml_prob_fake = float(clf.predict_proba(X)[0, 1])

        with st.spinner(""):
            st.markdown('<p class="scanning-msg">Checking credible sources…</p>', unsafe_allow_html=True)
            web_result = corroborate(txt, max_results=15)

        web_score = web_result.score

        if web_result.error and "Search failed" in (web_result.error or ""):
            composite_fake = ml_prob_fake
            web_weight = 0.0
        else:
            debunk_boost = min(0.3, web_result.debunk_hits * 0.15)
            # Web evidence weighted higher (70%) — ML model is less reliable on short headlines
            composite_real = 0.3 * (1.0 - ml_prob_fake) + 0.7 * web_score
            composite_fake = 1.0 - composite_real + debunk_boost
            web_weight = 0.7
            # Strong trusted-source override: 2+ credible sources is strong evidence
            n_trusted = len(web_result.sources_found)
            if n_trusted >= 3 and web_result.debunk_hits == 0:
                composite_fake = min(composite_fake, 0.20)
            elif n_trusted >= 2 and web_result.debunk_hits == 0:
                composite_fake = min(composite_fake, 0.30)

        composite_fake = max(0.0, min(1.0, composite_fake))
        confidence = abs(composite_fake - 0.5) * 2
        confidence_pct = confidence * 100
        is_fake = composite_fake >= 0.5

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        if is_fake:
            st.markdown(f"""
            <div class="verdict-container">
                <div class="verdict-emoji">⚠️</div>
                <div class="verdict-label-fake">Likely Fake</div>
                <div class="verdict-subtitle">This headline doesn't appear to be from credible sources</div>
            </div>
            """, unsafe_allow_html=True)
        elif confidence_pct < 15:
            st.markdown(f"""
            <div class="verdict-container">
                <div class="verdict-emoji">🔍</div>
                <div class="verdict-label-real" style="color:#FBBF24;">Unverified</div>
                <div class="verdict-subtitle">Not enough evidence to confirm or deny this headline</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-container">
                <div class="verdict-emoji">✅</div>
                <div class="verdict-label-real">Verified Real</div>
                <div class="verdict-subtitle">This headline is consistent with credible reporting</div>
            </div>
            """, unsafe_allow_html=True)

        fill_cls = "track-fill-fake" if is_fake else "track-fill-real"
        st.markdown(f"""
        <div class="track-container">
            <div class="track-label">
                <span>Confidence</span>
                <span>{confidence_pct:.0f}%</span>
            </div>
            <div class="track">
                <div class="{fill_cls}" style="width:{confidence_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat">
                <div class="stat-value">{ml_prob_fake:.0%}</div>
                <div class="stat-label">ML Score</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(web_result.sources_found)}</div>
                <div class="stat-label">Sources</div>
            </div>
            <div class="stat">
                <div class="stat-value">{confidence_pct:.0f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if web_result.sources_found:
            pills = "".join(f'<span class="source-pill">{s}</span>' for s in web_result.sources_found)
            st.markdown(f"""
            <div class="sources-section">
                <div class="sources-title">Corroborated by</div>
                {pills}
            </div>
            """, unsafe_allow_html=True)
        elif not web_result.error:
            st.markdown('<div class="sources-section"><div class="no-sources-msg">No credible sources found for this headline</div></div>', unsafe_allow_html=True)

        if web_result.debunk_hits > 0:
            debunk_names = ", ".join(web_result.debunk_sources) if web_result.debunk_sources else "multiple sources"
            st.markdown(f"""
            <div class="sources-section">
                <div style="color:#F87171; font-size:0.78rem; font-weight:500;">
                    ⚠ {web_result.debunk_hits} result(s) flagged as potential debunking / fact-check ({debunk_names})
                </div>
            </div>
            """, unsafe_allow_html=True)

        detail_caption = ""
        if web_result.total_results > 0:
            detail_caption = f'<div style="color:#3A3E4E; font-size:0.72rem; margin-top:0.6rem; text-align:center;">{web_result.total_results} searched · {web_result.relevant_results} relevant · {len(web_result.sources_found)} trusted · {web_result.debunk_hits} debunked</div>'
        if web_result.error:
            detail_caption += f'<div style="color:#F87171; font-size:0.72rem; margin-top:0.3rem; text-align:center;">⚠ {web_result.error}</div>'

        st.markdown(f"""
        <details class="custom-details">
            <summary>Signal breakdown</summary>
            <div class="details-content">
                <table class="detail-table">
                    <tr><th>Signal</th><th>Score</th><th>Weight</th></tr>
                    <tr><td>ML Model</td><td>{ml_prob_fake:.3f}</td><td>{1-web_weight:.0%}</td></tr>
                    <tr><td>Web Corroboration</td><td>{web_score:.2f}</td><td>{web_weight:.0%}</td></tr>
                    <tr><td style="color:#E4E4E8;font-weight:600;">Composite</td><td style="color:#E4E4E8;font-weight:600;">{composite_fake:.3f}</td><td>—</td></tr>
                </table>
                {detail_caption}
            </div>
        </details>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-footer">
        <div class="footer-brand">Built by <span>From Text to Truth</span></div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":

    
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            main()
        else:
            raise RuntimeError("Not in Streamlit context")
    except (ImportError, RuntimeError):
        import subprocess
        script = str(Path(__file__).resolve())
        print(f"Launching Streamlit server for {script} ...")
        subprocess.call([sys.executable, "-m", "streamlit", "run", script])
