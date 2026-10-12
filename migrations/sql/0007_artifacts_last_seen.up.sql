-- 0007: when were these bytes last fetched from this URL?
--
-- An artifact is unique per (source_url, sha256), so fetching identical bytes again reuses
-- the existing row. fetched_at then still says when they were FIRST seen, and a page that
-- changes A → B → A would look like it ended on B. last_seen_at is moved forward on every
-- fetch, and parsing picks each URL's artifact with the newest last_seen_at.

ALTER TABLE platform.artifacts ADD COLUMN last_seen_at TIMESTAMPTZ;
UPDATE platform.artifacts SET last_seen_at = fetched_at;
ALTER TABLE platform.artifacts ALTER COLUMN last_seen_at SET NOT NULL;
