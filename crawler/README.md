# Trustpilot Review Crawler

A web crawler for scraping Trustpilot reviews for JustAnswer.com and storing them in a SQLite database.

## Features

- Scrapes reviews from Trustpilot (www.trustpilot.com/review/www.justanswer.com)
- Extracts rich data from Next.js `__NEXT_DATA__` payload
- Stores data in SQLite database with duplicate prevention
- Handles pagination (up to 10 pages due to Trustpilot limits)
- Includes retry logic with exponential backoff
- Respects rate limits with random delays between requests

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Usage

### Basic Usage
```bash
python scraper.py
```

### With Custom Cutoff Date
```bash
python scraper.py --cutoff 2025-12-01
```

### With Custom Database Path
```bash
python scraper.py --db /path/to/reviews.db
```

### All Options
```bash
python scraper.py --cutoff 2025-12-01 --db /path/to/reviews.db
```

## Database Schema

The SQLite database (`reviews.db`) contains a `reviews` table with these columns:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| review_id | TEXT | Unique Trustpilot review ID |
| author_name | TEXT | Reviewer's display name |
| rating | INTEGER | Rating (1-5 stars) |
| title | TEXT | Review title |
| review_body | TEXT | Full review text |
| date_published | TEXT | ISO format publication date |
| date_experienced | TEXT | ISO format experience date |
| language | TEXT | Review language code |
| consumer_country | TEXT | Reviewer's country code |
| is_verified | INTEGER | 1 if verified, 0 otherwise |
| verification_source | TEXT | Source of verification |
| reply_text | TEXT | Company reply text |
| reply_date | TEXT | ISO format reply date |
| page_number | INTEGER | Page where review was found |
| scraped_at | TIMESTAMP | When review was scraped |

## Data Extraction

The crawler extracts data from Trustpilot's Next.js application by parsing the `__NEXT_DATA__` script tag, which contains richer data than the JSON-LD markup, including:

- Reply text from JustAnswer
- Consumer country information
- Verification status and source
- Experience dates (when available)

## Limitations

1. **Rate Limits**: Trustpilot limits unauthenticated access to ~10 pages (~200 reviews)
2. **Pagination**: The scraper stops when it reaches the cutoff date or maximum pages
3. **Dynamic Content**: Requires headless browser (Playwright) to render JavaScript

## Scheduling

For continuous data collection, schedule the crawler to run periodically:

```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/crawler && python scraper.py
```

## Integration with NPS Growth Engine

The crawler creates `reviews.db` which can be used by the NPS Growth Engine application. To export data for the Streamlit app:

```bash
# Export to CSV
sqlite3 reviews.db "SELECT date_published, rating, review_body FROM reviews;" > reviews.csv
```

## Troubleshooting

### Common Issues

1. **No reviews found**: Trustpilot may have changed their HTML structure
2. **Connection errors**: Check internet connectivity and firewall settings
3. **Playwright errors**: Ensure Chromium is installed: `playwright install chromium`

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Dependencies

See `requirements.txt` for complete list:
- `crawl4ai`: Web crawling framework
- `playwright`: Headless browser automation
- `python-dotenv`: Environment variable management
- `aiohttp`: Async HTTP client

## License

Part of the NPS Growth Engine project.