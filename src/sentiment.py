"""
News sentiment analysis for SPY/QQQ using free RSS feeds.
No API key needed — uses Google News, Yahoo Finance, and MarketWatch RSS.

Outputs a sentiment score from -1.0 (extreme fear) to +1.0 (extreme greed)
that feeds into the ML predictor and screener as an additional feature.
"""

import feedparser
import re
import numpy as np
from dataclasses import dataclass
from textblob import TextBlob
from datetime import datetime, timedelta


# Free RSS feeds for market news (no API key required)
FEED_URLS = {
    "google_spy": "https://news.google.com/rss/search?q=SPY+stock+market&hl=en-US&gl=US&ceid=US:en",
    "google_qqq": "https://news.google.com/rss/search?q=QQQ+nasdaq+stock&hl=en-US&gl=US&ceid=US:en",
    "google_market": "https://news.google.com/rss/search?q=stock+market+today&hl=en-US&gl=US&ceid=US:en",
    "google_fed": "https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en-US&gl=US&ceid=US:en",
}

# Keywords that amplify sentiment for options trading
BULLISH_KEYWORDS = [
    "rally", "surge", "breakout", "bullish", "record high", "buy",
    "upgrade", "beat expectations", "strong earnings", "recovery",
    "rate cut", "dovish", "stimulus", "jobs growth", "rebound",
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "selloff", "bearish", "correction", "recession",
    "downgrade", "miss expectations", "weak earnings", "layoffs",
    "rate hike", "hawkish", "inflation", "tariff", "crisis", "default",
]

FEAR_KEYWORDS = [
    "fear", "panic", "volatility", "uncertainty", "risk", "warning",
    "collapse", "contagion", "black swan", "bubble",
]


@dataclass
class HeadlineSentiment:
    title: str
    source: str
    polarity: float      # -1.0 to 1.0 (TextBlob)
    keyword_score: float  # -1.0 to 1.0 (keyword matching)
    combined: float       # weighted average
    age_hours: float


@dataclass
class MarketSentiment:
    headlines_analyzed: int
    avg_polarity: float
    avg_keyword_score: float
    composite_score: float   # final score: -1.0 to +1.0
    bullish_count: int
    bearish_count: int
    neutral_count: int
    fear_level: float        # 0.0 to 1.0
    top_bullish: list
    top_bearish: list

    @property
    def label(self) -> str:
        if self.composite_score > 0.15:
            return "BULLISH"
        elif self.composite_score < -0.15:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def confidence(self) -> float:
        return min(abs(self.composite_score) / 0.5, 1.0)


class NewsSentimentAnalyzer:
    def __init__(self, max_age_hours: int = 48):
        self.max_age_hours = max_age_hours

    def _fetch_headlines(self, ticker: str = "SPY") -> list[dict]:
        """Fetch headlines from multiple RSS feeds."""
        headlines = []
        ticker_lower = ticker.lower()

        # Select relevant feeds
        feeds_to_use = ["google_market", "google_fed"]
        if ticker_lower in ("spy", "spx"):
            feeds_to_use.append("google_spy")
        elif ticker_lower in ("qqq", "nasdaq"):
            feeds_to_use.append("google_qqq")
        else:
            feeds_to_use.append("google_spy")

        for feed_key in feeds_to_use:
            url = FEED_URLS.get(feed_key)
            if not url:
                continue
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    pub_date = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    headlines.append({
                        "title": entry.get("title", ""),
                        "source": entry.get("source", {}).get("title", feed_key),
                        "published": pub_date,
                        "link": entry.get("link", ""),
                    })
            except Exception:
                continue

        return headlines

    def _keyword_score(self, text: str) -> float:
        """Score text based on bullish/bearish/fear keyword presence."""
        text_lower = text.lower()
        bull_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bear_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        fear_hits = sum(1 for kw in FEAR_KEYWORDS if kw in text_lower)

        total = bull_hits + bear_hits + fear_hits
        if total == 0:
            return 0.0

        score = (bull_hits - bear_hits - fear_hits * 0.5) / total
        return max(-1.0, min(1.0, score))

    def _age_hours(self, pub_date) -> float:
        if pub_date is None:
            return 24.0  # assume 1 day old if unknown
        delta = datetime.utcnow() - pub_date
        return delta.total_seconds() / 3600

    def _time_weight(self, age_hours: float) -> float:
        """Recent news weighted higher. Exponential decay."""
        return np.exp(-age_hours / 24)  # half-life of ~24 hours

    def analyze(self, ticker: str = "SPY") -> MarketSentiment:
        """Analyze sentiment from latest news headlines for a ticker."""
        headlines = self._fetch_headlines(ticker)

        sentiments = []
        for h in headlines:
            age = self._age_hours(h["published"])
            if age > self.max_age_hours:
                continue

            title = h["title"]
            # TextBlob NLP sentiment
            blob = TextBlob(title)
            polarity = blob.sentiment.polarity  # -1.0 to 1.0

            # Keyword-based sentiment
            kw_score = self._keyword_score(title)

            # Combined: 40% NLP + 60% keywords (keywords more reliable for finance)
            combined = 0.4 * polarity + 0.6 * kw_score

            sentiments.append(HeadlineSentiment(
                title=title,
                source=h["source"],
                polarity=polarity,
                keyword_score=kw_score,
                combined=combined,
                age_hours=age,
            ))

        if not sentiments:
            return MarketSentiment(
                headlines_analyzed=0, avg_polarity=0, avg_keyword_score=0,
                composite_score=0, bullish_count=0, bearish_count=0,
                neutral_count=0, fear_level=0, top_bullish=[], top_bearish=[],
            )

        # Time-weighted composite score
        weights = np.array([self._time_weight(s.age_hours) for s in sentiments])
        scores = np.array([s.combined for s in sentiments])
        composite = float(np.average(scores, weights=weights))

        bullish = [s for s in sentiments if s.combined > 0.1]
        bearish = [s for s in sentiments if s.combined < -0.1]
        neutral = [s for s in sentiments if -0.1 <= s.combined <= 0.1]

        # Fear level from fear keywords
        fear_scores = [abs(s.keyword_score) for s in sentiments
                       if any(kw in s.title.lower() for kw in FEAR_KEYWORDS)]
        fear_level = float(np.mean(fear_scores)) if fear_scores else 0.0

        # Top headlines
        sorted_bull = sorted(bullish, key=lambda s: s.combined, reverse=True)
        sorted_bear = sorted(bearish, key=lambda s: s.combined)

        return MarketSentiment(
            headlines_analyzed=len(sentiments),
            avg_polarity=float(np.mean([s.polarity for s in sentiments])),
            avg_keyword_score=float(np.mean([s.keyword_score for s in sentiments])),
            composite_score=round(composite, 4),
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            fear_level=round(fear_level, 3),
            top_bullish=[s.title for s in sorted_bull[:3]],
            top_bearish=[s.title for s in sorted_bear[:3]],
        )

    def print_report(self, ticker: str = "SPY"):
        result = self.analyze(ticker)
        width = 60
        sep = "─" * width
        print(f"\n┌{sep}┐")
        print(f"│{'  NEWS SENTIMENT: ' + ticker:^{width}}│")
        print(f"├{sep}┤")
        print(f"│  Headlines Analyzed : {result.headlines_analyzed:<{width - 23}}│")
        print(f"│  Sentiment          : {result.label:<{width - 23}}│")
        print(f"│  Composite Score    : {result.composite_score:+.3f}{'':>{width - 30}}│")
        print(f"│  Confidence         : {result.confidence:.0%}{'':>{width - 28}}│")
        print(f"│  Fear Level         : {result.fear_level:.1%}{'':>{width - 28}}│")
        print(f"├{sep}┤")
        print(f"│  Bullish: {result.bullish_count}  Bearish: {result.bearish_count}  Neutral: {result.neutral_count}{'':>{width - 42}}│")
        print(f"├{sep}┤")
        if result.top_bullish:
            print(f"│  Top Bullish:{'':>{width - 15}}│")
            for h in result.top_bullish:
                print(f"│    + {h[:width - 7]:<{width - 7}}│")
        if result.top_bearish:
            print(f"│  Top Bearish:{'':>{width - 15}}│")
            for h in result.top_bearish:
                print(f"│    - {h[:width - 7]:<{width - 7}}│")
        print(f"└{sep}┘")
        return result


if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    for t in ["SPY", "QQQ"]:
        analyzer.print_report(t)
