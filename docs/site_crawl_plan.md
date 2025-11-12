# Site Crawl & Change Monitoring Plan

## Goals
- Spider entire doc sections (e.g., `https://example.com/docs/*`), ingest each page.
- Watch those routes for changes; reingest pages when content updates.
- Support both HTML pages and linked files (PDFs, etc.).

## Functionality
1. **Crawl job definition**
   - `/ingest/crawl` accepts `{base_url, include_patterns, exclude_patterns, depth, schedule}`.
   - Downloader fetches pages via Firecrawl/HTTP, converts to text + metadata, and enqueues ingestion.

2. **Change detection**
   - Store fingerprint (hash) per URL. On each crawl, compare hash; if changed, reingest.
   - Track status (new, updated, deleted) per page.

3. **Scheduling / Monitoring**
   - Background worker runs crawl jobs on a schedule (cron-like or at intervals).
   - Job summary reports pages crawled, new/updated counts, failures.

4. **Linked files**
   - When encountering links to PDFs/Word docs, download them and treat as regular ingestion items.
   - Apply same change detection logic (hash of downloaded file).

5. **Configuration**
   - `CATALYST_CRAWLER__max_depth`, `__max_pages`, global throttle settings.
   - Option to limit by domain or path prefix.

## Implementation Steps
1. Build `CrawlerJob` schema with URL patterns and schedule.
2. Implement crawler worker using Firecrawl or requests+BeautifulSoup to follow links.
3. Store per-URL fingerprints to detect changes; enqueue ingestion only when new/updated.
4. Expose APIs to start crawl, list runs, and inspect page status.
5. Add metrics (pages crawled, change rate) and CLI support (`catalystctl crawl status`).
