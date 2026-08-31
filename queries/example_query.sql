-- =============================================================================
-- example_query.sql
--
-- Worked example of the two-stage retrieval procedure, rendered in full for a
-- single domain (`rail`). The same template is applied to every domain by
-- substituting that domain's rows from cpc_codes.csv and keywords.csv; `rail`
-- is shown here because it exercises both keyword matching modes (substring
-- and word-boundary).
--
-- Dialect: PostgreSQL. The `~*` operator is a case-insensitive POSIX regular
-- expression match and `\y` is the PostgreSQL word-boundary escape; both need
-- to be replaced when porting to another RDBMS.
--
-- Expected source schema
-- ----------------------
--   patents(pub_number, pub_date, appl_date, title, abstract)
--   patent_classifications(pub_number, scheme, section, class, subclass,
--                          main_group, subgroup)
--
-- In the study the two source tables were first materialised as subsets
-- restricted to appl_date >= 2018-01-01 to avoid repeated full scans. That is
-- a performance optimisation only and does not affect the result set.
--
-- Matching rules are documented in queries/README.md.
-- =============================================================================

WITH base AS (
    SELECT pub_number, pub_date, appl_date, title, abstract
    FROM patents
    WHERE appl_date >= DATE '2018-01-01'
),

-- ---------------------------------------------------------------------------
-- Stage 1: domain identification (CPC OR keyword)
-- ---------------------------------------------------------------------------
domain_cpc AS (
    SELECT DISTINCT pub_number
    FROM patent_classifications
    WHERE scheme = 'cpc'
      AND (
             ("section" = 'B' AND "class" = '61')                                                    -- B61
          OR ("section" = 'B' AND "class" = '61' AND "subclass" = 'L')                               -- B61L
          OR ("section" = 'B' AND "class" = '60' AND "subclass" = 'L')                               -- B60L
          OR ("section" = 'G' AND "class" = '05' AND "subclass" = 'D' AND "main_group" = '1')        -- G05D1
          OR ("section" = 'G' AND "class" = '06' AND "subclass" = 'Q'
              AND "main_group" = '50' AND "subgroup" = '40')                                         -- G06Q50/40
      )
),

domain_keyword AS (
    SELECT pub_number
    FROM base
    WHERE (
             abstract ILIKE '%railway%'
          OR abstract ILIKE '%railway train%'
          OR abstract ILIKE '%railway track%'
          OR abstract ILIKE '%train track%'
          OR abstract ILIKE '%railway signaling%'
          OR abstract ILIKE '%train signaling%'
          OR abstract ILIKE '%locomotive%'
          OR abstract ILIKE '%railroad%'
      )
      -- word-boundary term: plain substring matching would also hit
      -- "trail", "trailer", "frail"
      OR (abstract ~* '\yrail\y')
),

-- ---------------------------------------------------------------------------
-- Stage 2: AI identification (CPC OR keyword)
-- ---------------------------------------------------------------------------
ai_cpc AS (
    SELECT DISTINCT pub_number
    FROM patent_classifications
    WHERE scheme = 'cpc'
      AND ("section" = 'G' AND "class" = '06' AND "subclass" = 'N')                                  -- G06N
),

ai_keyword AS (
    SELECT pub_number
    FROM base
    WHERE (
             abstract ILIKE '%machine learning%'
          OR abstract ILIKE '%deep learning%'
          OR abstract ILIKE '%neural network%'
          OR abstract ILIKE '%reinforcement learning%'
          OR abstract ILIKE '%artificial intelligence%'
          OR abstract ILIKE '%support vector machine%'
          OR abstract ILIKE '%random forest%'
          OR abstract ILIKE '%decision tree%'
          OR abstract ILIKE '%naive bayes%'
          OR abstract ILIKE '%hidden markov%'
          OR abstract ILIKE '%convolutional neural%'
          OR abstract ILIKE '%recurrent neural%'
          OR abstract ILIKE '%generative adversarial%'
          OR abstract ILIKE '%large language model%'
      )
      -- word-boundary terms: plain substring matching on "AI" would also hit
      -- "contain", "detail", "maintain", "said"
      OR (
             abstract ~* '\yAI\y'
          OR abstract ~* '\yAI-based\y'
          OR abstract ~* '\yLLM\y'
      )
)

-- ---------------------------------------------------------------------------
-- Final set: Stage 1 AND Stage 2
-- ---------------------------------------------------------------------------
SELECT DISTINCT b.pub_number, b.pub_date, b.appl_date, b.title, b.abstract
FROM base b
WHERE b.pub_number IN (
        SELECT pub_number FROM domain_cpc
        UNION
        SELECT pub_number FROM domain_keyword
      )
  AND b.pub_number IN (
        SELECT pub_number FROM ai_cpc
        UNION
        SELECT pub_number FROM ai_keyword
      )
ORDER BY b.appl_date DESC;
