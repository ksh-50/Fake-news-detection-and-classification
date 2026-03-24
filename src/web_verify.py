from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

TRUSTED_SOURCES: dict[str, str] = {
    "reuters.com": "Reuters",
    "apnews.com": "AP News",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "cnn.com": "CNN",
    "nytimes.com": "New York Times",
    "theguardian.com": "The Guardian",
    "washingtonpost.com": "Washington Post",
    "npr.org": "NPR",
    "aljazeera.com": "Al Jazeera",
    "abc.net.au": "ABC Australia",
    "abcnews.go.com": "ABC News",
    "cbsnews.com": "CBS News",
    "nbcnews.com": "NBC News",
    "usatoday.com": "USA Today",
    "politico.com": "Politico",
    "thehill.com": "The Hill",
    "france24.com": "France 24",
    "dw.com": "DW News",
    "pbs.org": "PBS",
    "forbes.com": "Forbes",
    "bloomberg.com": "Bloomberg",
    "economist.com": "The Economist",
    "time.com": "TIME",
    "wsj.com": "Wall Street Journal",
    "ft.com": "Financial Times",
    "independent.co.uk": "The Independent",
    "sky.com": "Sky News",
    "news.sky.com": "Sky News",
    "hindustantimes.com": "Hindustan Times",
    "ndtv.com": "NDTV",
    "timesofindia.indiatimes.com": "Times of India",
    "thehindu.com": "The Hindu",
    "indianexpress.com": "Indian Express",
    "snopes.com": "Snopes",
    "factcheck.org": "FactCheck.org",
    "politifact.com": "PolitiFact",
}

FACTCHECK_DOMAINS = {"snopes.com", "factcheck.org", "politifact.com"}

DEBUNK_KEYWORDS = re.compile(
    r"\b(fake|hoax|false|debunk|misinformation|disinformation|rumou?r|"
    r"not true|no truth|fact.?check|misleading|fabricat|satire|parody|"
    r"unverified|baseless|doctored|manipulated|scam|conspiracy|denied|"
    r"denies|no evidence|unfounded|didn.?t happen)\b",
    re.IGNORECASE,
)


@dataclass
class CorroborationResult:
    query: str = ""
    sources_found: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    debunk_sources: List[str] = field(default_factory=list)
    total_results: int = 0
    relevant_results: int = 0
    debunk_hits: int = 0
    score: float = 0.0
    error: str | None = None


def _domain_from_url(url: str) -> str:
    url = url.lower()
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else ""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _extract_keywords(headline: str) -> set[str]:
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "her", "was", "one", "our", "out", "has", "had", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "from", "have", "been",
        "with", "this", "that", "will", "would", "could", "should", "into",
        "than", "them", "then", "they", "what", "when", "where", "which",
        "about", "after", "being", "their", "there", "these", "those",
        "very", "just", "also", "more", "some", "such", "only",
        "says", "said", "according", "report", "reports", "news",
    }
    words = set(re.findall(r"[a-z]{3,}", headline.lower()))
    return words - stop


def _relevance_score(headline: str, title: str, body: str) -> float:
    keywords = _extract_keywords(headline)
    if not keywords:
        return 0.0
    combined = _normalize(title + " " + body)
    matched = sum(1 for kw in keywords if kw in combined)
    return matched / len(keywords)


def _has_debunk_signals(title: str, body: str) -> bool:
    return bool(DEBUNK_KEYWORDS.search(title + " " + body))


def corroborate(headline: str, *, max_results: int = 20) -> CorroborationResult:
    result = CorroborationResult(query=headline)

    try:
        from ddgs import DDGS
    except ImportError:
        result.error = "ddgs library not installed – pip install ddgs"
        return result

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(headline, max_results=max_results))
    except Exception as exc:
        result.error = f"Search failed: {exc}"
        result.score = 0.5
        return result

    result.total_results = len(hits)
    seen_sources: set[str] = set()

    for hit in hits:
        url = hit.get("href", "") or hit.get("link", "")
        title = hit.get("title", "")
        body = hit.get("body", "") or hit.get("snippet", "")
        domain = _domain_from_url(url)

        relevance = _relevance_score(headline, title, body)
        if relevance < 0.25:
            continue

        result.relevant_results += 1

        is_debunk = _has_debunk_signals(title, body)
        is_factcheck = any(fc in domain for fc in FACTCHECK_DOMAINS)

        matched_source = None
        for trusted_domain, source_name in TRUSTED_SOURCES.items():
            if trusted_domain in domain and source_name not in seen_sources:
                matched_source = source_name
                seen_sources.add(source_name)
                break

        if is_debunk or is_factcheck:
            result.debunk_hits += 1
            if matched_source:
                result.debunk_sources.append(matched_source)
        elif matched_source:
            result.sources_found.append(matched_source)
            result.source_urls.append(url)

    n_corr = len(result.sources_found)
    if n_corr == 0:
        # No trusted sources, but relevant results hint it's a real story
        if result.relevant_results >= 5:
            base_score = 0.30
        elif result.relevant_results >= 2:
            base_score = 0.20
        else:
            base_score = 0.10
    elif n_corr == 1:
        base_score = 0.55
    elif n_corr == 2:
        base_score = 0.75
    elif n_corr == 3:
        base_score = 0.85
    else:
        base_score = min(1.0, 0.85 + 0.05 * (n_corr - 3))

    if result.debunk_hits > 0:
        base_score = max(0.0, base_score - min(0.6, result.debunk_hits * 0.25))

    word_count = len(headline.split())
    if word_count <= 3:
        base_score *= 0.6
    elif word_count <= 5:
        base_score *= 0.8

    if result.total_results > 0 and result.relevant_results == 0:
        base_score = 0.05

    result.score = max(0.0, min(1.0, base_score))
    return result
