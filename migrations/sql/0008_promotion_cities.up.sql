-- 0008: promotions.cities, for questions like "new stores in Kaohsiung".
--
-- Full city or county names (高雄市, 新竹縣), written with 台. Store notices fill it from
-- the store's address, by rules (no LLM). Empty = not limited to some cities, or unknown.
-- Existing rows get an empty list; re-running extract fills them.

ALTER TABLE rag.promotions ADD COLUMN cities VARCHAR[] NOT NULL DEFAULT '{}';
