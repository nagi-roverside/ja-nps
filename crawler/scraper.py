"""
Trustpilot Review Scraper for JustAnswer.com

Uses crawl4ai to scrape reviews from Trustpilot and store them in a SQLite
database.  Reviews are parsed from the Next.js __NEXT_DATA__ payload which
contains richer data than the JSON-LD markup (reply text, consumer country,
verification status, etc.).

NOTE: Trustpilot limits unauthenticated access to 10 pages (≈200 reviews).
Run the scraper periodically to accumulate reviews over time — duplicates
are automatically skipped via the UNIQUE constraint on review_id.
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.trustpilot.com/review/www.justanswer.com"
DB_PATH = Path(__file__).parent / "reviews.db"
MAX_PAGES = 10  # Trustpilot hard limit for unauthenticated users


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id           TEXT UNIQUE,
            author_name         TEXT,
            rating              INTEGER,
            title               TEXT,
            review_body         TEXT,
            date_published      TEXT,
            date_experienced    TEXT,
            language            TEXT,
            consumer_country    TEXT,
            is_verified         INTEGER,
            verification_source TEXT,
            reply_text          TEXT,
            reply_date          TEXT,
            page_number         INTEGER,
            scraped_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def insert_reviews(conn: sqlite3.Connection, reviews: list[dict]) -> int:
    """Insert reviews, skipping duplicates. Returns count of newly inserted."""
    inserted = 0
    for r in reviews:
        try:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO reviews
                    (review_id, author_name, rating, title, review_body,
                     date_published, date_experienced, language, consumer_country,
                     is_verified, verification_source, reply_text, reply_date,
                     page_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["review_id"], r["author_name"], r["rating"],
                    r["title"], r["review_body"], r["date_published"],
                    r["date_experienced"], r["language"], r["consumer_country"],
                    r["is_verified"], r["verification_source"],
                    r["reply_text"], r["reply_date"], r["page_number"],
                ),
            )
            if cursor.rowcount > 0:
                inserted += 1
        except sqlite3.Error as e:
            logger.warning("DB insert error for review %s: %s", r.get("review_id"), e)
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_reviews_from_html(html: str, page_number: int) -> list[dict]:
    """Extract review data from __NEXT_DATA__ embedded in the HTML."""
    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
    )
    if not match:
        return []

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    raw_reviews = data.get("props", {}).get("pageProps", {}).get("reviews", [])

    reviews: list[dict] = []
    for r in raw_reviews:
        dates = r.get("dates", {})
        consumer = r.get("consumer", {})
        verification = r.get("labels", {}).get("verification", {})
        reply = r.get("reply") or {}

        reviews.append({
            "review_id": r.get("id", ""),
            "author_name": consumer.get("displayName", ""),
            "rating": r.get("rating", 0),
            "title": r.get("title", ""),
            "review_body": r.get("text", ""),
            "date_published": dates.get("publishedDate", ""),
            "date_experienced": dates.get("experiencedDate", ""),
            "language": r.get("language", ""),
            "consumer_country": consumer.get("countryCode", ""),
            "is_verified": 1 if verification.get("isVerified") else 0,
            "verification_source": verification.get("reviewSourceName", ""),
            "reply_text": reply.get("message", ""),
            "reply_date": reply.get("publishedDate", ""),
            "page_number": page_number,
        })

    return reviews


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def oldest_review_date(reviews: list[dict]) -> datetime | None:
    """Return the oldest date_published among the reviews."""
    oldest = None
    for r in reviews:
        raw = r.get("date_published", "")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if oldest is None or dt < oldest:
                oldest = dt
        except ValueError:
            continue
    return oldest


async def scrape_page(
    crawler: AsyncWebCrawler,
    page_number: int,
    config: CrawlerRunConfig,
    max_retries: int = 5,
) -> list[dict]:
    """Scrape one Trustpilot page with retries."""
    url = f"{BASE_URL}?page={page_number}"

    for attempt in range(1, max_retries + 1):
        try:
            result = await crawler.arun(url=url, config=config)
            if not result.success:
                logger.warning("Page %d attempt %d failed: %s", page_number, attempt, result.error_message)
            else:
                reviews = parse_reviews_from_html(result.html, page_number)
                if reviews:
                    return reviews
                logger.warning("Page %d attempt %d: no reviews found", page_number, attempt)
        except Exception as e:
            logger.warning("Page %d attempt %d exception: %s", page_number, attempt, e)

        if attempt < max_retries:
            backoff = 5 * (2 ** (attempt - 1))
            logger.info("Backing off %ds before retry...", backoff)
            await asyncio.sleep(backoff)

    logger.error("Page %d: all %d attempts failed", page_number, max_retries)
    return []


async def run_scraper(cutoff_date: datetime | None) -> None:
    """Scrape pages 1 through MAX_PAGES, stopping early if cutoff date reached."""
    conn = init_db()
    total_found = 0
    total_inserted = 0

    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        enable_stealth=True,
        viewport_width=1280,
        viewport_height=800,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scan_full_page=False,
        page_timeout=60000,
    )

    logger.info(
        "Starting scrape (max %d pages, cutoff %s)",
        MAX_PAGES,
        cutoff_date.strftime("%Y-%m-%d") if cutoff_date else "none",
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for page in range(1, MAX_PAGES + 1):
            logger.info("Scraping page %d / %d ...", page, MAX_PAGES)
            reviews = await scrape_page(crawler, page, run_config)

            if not reviews:
                logger.warning("Page %d: no reviews returned, stopping", page)
                break

            total_found += len(reviews)
            inserted = insert_reviews(conn, reviews)
            total_inserted += inserted
            logger.info("Page %d: found %d, inserted %d new", page, len(reviews), inserted)

            # Stop early if oldest review on this page is before cutoff
            if cutoff_date:
                oldest = oldest_review_date(reviews)
                if oldest and oldest < cutoff_date:
                    logger.info(
                        "Oldest review on page %d (%s) is before cutoff, stopping",
                        page, oldest.strftime("%Y-%m-%d %H:%M"),
                    )
                    break

            # Random delay 300-3000ms between pages
            if page < MAX_PAGES:
                delay = random.randint(300, 3000) / 1000.0
                logger.info("Waiting %.1fs ...", delay)
                await asyncio.sleep(delay)

    conn.close()

    # Summary
    summary = conn_summary(DB_PATH)
    logger.info(
        "Done. This run: found %d, inserted %d new. DB total: %d reviews (%s to %s)",
        total_found, total_inserted, summary["count"], summary["oldest"], summary["newest"],
    )


def conn_summary(db_path: Path) -> dict:
    """Return quick stats about the database."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*), MIN(date_published), MAX(date_published) FROM reviews"
    ).fetchone()
    conn.close()
    return {
        "count": row[0],
        "oldest": (row[1] or "")[:10],
        "newest": (row[2] or "")[:10],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Trustpilot reviews for JustAnswer.com into SQLite"
    )
    parser.add_argument(
        "--cutoff", type=str, default="2026-01-01",
        help="Stop if reviews are older than this date, YYYY-MM-DD (default: 2026-01-01)",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Path to SQLite database file (default: reviews.db in script dir)",
    )
    args = parser.parse_args()

    if args.db:
        global DB_PATH
        DB_PATH = Path(args.db)

    cutoff_date = datetime.strptime(args.cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    asyncio.run(run_scraper(cutoff_date))


if __name__ == "__main__":
    main()
