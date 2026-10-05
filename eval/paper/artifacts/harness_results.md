# BFCL Multiple — harness results

Shared catalog C = 443 tools, 200 queries, k = 10, sentence-transformers/all-MiniLM-L6-v2.
Protocol: `shared_catalog`. BFCL-derived; **not** an official Gorilla leaderboard score.

| | |
|---|---|
| Queries scored | 200 per model |
| Models | 5 |
| Catalog C | 443 tools |
| Context compression at k=10 | 97.7% |
| Largest name-acc Δ vs baseline | +86.0 pp (llama-3.1-8b-instruct, ToolScope@10) |
| Instances skipped | 0 |

---

Skipped instances: **0**. `api_fail` on at least one condition: **23** queries across the matrix.

## Tool name accuracy (headline)

Share of queries where the model called a ground-truth tool name. Retrieval metrics are identical across models for a given retriever.

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.5% | 85.0% | 85.5% | 82.5% | 84.5% | 84.5% | 83.5% |
| llama-3.1-8b-instruct | 6.0% | 89.0% | 91.5% | 89.5% | 90.5% | 92.0% | 92.5% |
| qwen2.5-7b-instruct | 40.0% | 83.5% | 86.0% | 88.5% | 84.5% | 87.0% | 87.5% |
| qwen3-32b | 72.5% | 85.0% | 89.0% | 89.5% | 86.5% | 88.0% | 89.0% |
| llama-3.3-70b-instruct | 79.0% | 88.0% | 90.5% | 91.0% | 87.5% | 91.0% | 91.5% |

Models ordered by baseline name accuracy (weakest catalog handler first).

## Δ name acc vs full catalog

Selection gain shrinks as baseline name accuracy rises. McNemar is exact two-sided on paired name-acc flips (ToolScope@10 vs baseline).

| Model | Baseline name acc | BM25 Δ | ToolScope Δ | ToolScope@10 flips (win/lose) | McNemar p |
|---|---:|---:|---:|---|---:|
| llama-3.2-3b-instruct | 2.5% | +83.0 pp | +82.0 pp | +165 / −1 | < 0.001 |
| llama-3.1-8b-instruct | 6.0% | +85.5 pp | +86.0 pp | +173 / −1 | < 0.001 |
| qwen2.5-7b-instruct | 40.0% | +46.0 pp | +47.0 pp | +104 / −10 | < 0.001 |
| qwen3-32b | 72.5% | +16.5 pp | +15.5 pp | +41 / −10 | < 0.001 |
| llama-3.3-70b-instruct | 79.0% | +11.5 pp | +12.0 pp | +28 / −4 | < 0.001 |

## Per-condition matrix

| Model | Condition | Name acc | AST acc | Δ name | Recall@10 | NDCG@10 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | Baseline | 2.5% | 2.0% | — | — | — | 28.0 s |
| llama-3.2-3b-instruct | BM25@5 | 85.0% | 47.5% | +82.5 pp | 95.0% | 0.874 | 1.8 s |
| llama-3.2-3b-instruct | BM25@10 | 85.5% | 47.0% | +83.0 pp | 97.0% | 0.881 | 2.7 s |
| llama-3.2-3b-instruct | BM25@20 | 82.5% | 44.5% | +80.0 pp | 99.0% | 0.886 | 3.6 s |
| llama-3.2-3b-instruct | ToolScope@5 | 84.5% | 47.5% | +82.0 pp | 96.0% | 0.877 | 1.7 s |
| llama-3.2-3b-instruct | ToolScope@10 | 84.5% | 46.5% | +82.0 pp | 98.5% | 0.885 | 2.9 s |
| llama-3.2-3b-instruct | ToolScope@20 | 83.5% | 44.0% | +81.0 pp | 99.5% | 0.888 | 3.4 s |
| llama-3.1-8b-instruct | Baseline | 6.0% | 3.5% | — | — | — | 36.3 s |
| llama-3.1-8b-instruct | BM25@5 | 89.0% | 49.5% | +83.0 pp | 95.0% | 0.874 | 1.5 s |
| llama-3.1-8b-instruct | BM25@10 | 91.5% | 50.5% | +85.5 pp | 97.0% | 0.881 | 1.6 s |
| llama-3.1-8b-instruct | BM25@20 | 89.5% | 49.0% | +83.5 pp | 99.0% | 0.886 | 2.5 s |
| llama-3.1-8b-instruct | ToolScope@5 | 90.5% | 49.0% | +84.5 pp | 96.0% | 0.877 | 1.4 s |
| llama-3.1-8b-instruct | ToolScope@10 | 92.0% | 52.0% | +86.0 pp | 98.5% | 0.885 | 1.6 s |
| llama-3.1-8b-instruct | ToolScope@20 | 92.5% | 52.0% | +86.5 pp | 99.5% | 0.888 | 2.4 s |
| qwen2.5-7b-instruct | Baseline | 40.0% | 23.5% | — | — | — | 56.1 s |
| qwen2.5-7b-instruct | BM25@5 | 83.5% | 52.0% | +43.5 pp | 95.0% | 0.874 | 1.8 s |
| qwen2.5-7b-instruct | BM25@10 | 86.0% | 53.5% | +46.0 pp | 97.0% | 0.881 | 1.9 s |
| qwen2.5-7b-instruct | BM25@20 | 88.5% | 55.0% | +48.5 pp | 99.0% | 0.886 | 2.4 s |
| qwen2.5-7b-instruct | ToolScope@5 | 84.5% | 51.5% | +44.5 pp | 96.0% | 0.877 | 1.7 s |
| qwen2.5-7b-instruct | ToolScope@10 | 87.0% | 53.5% | +47.0 pp | 98.5% | 0.885 | 1.8 s |
| qwen2.5-7b-instruct | ToolScope@20 | 87.5% | 56.0% | +47.5 pp | 99.5% | 0.888 | 2.2 s |
| qwen3-32b | Baseline | 72.5% | 45.0% | — | — | — | 75.2 s |
| qwen3-32b | BM25@5 | 85.0% | 52.5% | +12.5 pp | 95.0% | 0.874 | 60.6 s |
| qwen3-32b | BM25@10 | 89.0% | 55.5% | +16.5 pp | 97.0% | 0.881 | 61.4 s |
| qwen3-32b | BM25@20 | 89.5% | 57.5% | +17.0 pp | 99.0% | 0.886 | 106.5 s |
| qwen3-32b | ToolScope@5 | 86.5% | 54.5% | +14.0 pp | 96.0% | 0.877 | 56.2 s |
| qwen3-32b | ToolScope@10 | 88.0% | 55.0% | +15.5 pp | 98.5% | 0.885 | 65.8 s |
| qwen3-32b | ToolScope@20 | 89.0% | 56.0% | +16.5 pp | 99.5% | 0.888 | 99.0 s |
| llama-3.3-70b-instruct | Baseline | 79.0% | 46.5% | — | — | — | 16.5 s |
| llama-3.3-70b-instruct | BM25@5 | 88.0% | 60.0% | +9.0 pp | 95.0% | 0.874 | 15.1 s |
| llama-3.3-70b-instruct | BM25@10 | 90.5% | 60.0% | +11.5 pp | 97.0% | 0.881 | 14.6 s |
| llama-3.3-70b-instruct | BM25@20 | 91.0% | 60.0% | +12.0 pp | 99.0% | 0.886 | 18.4 s |
| llama-3.3-70b-instruct | ToolScope@5 | 87.5% | 60.0% | +8.5 pp | 96.0% | 0.877 | 14.0 s |
| llama-3.3-70b-instruct | ToolScope@10 | 91.0% | 61.0% | +12.0 pp | 98.5% | 0.885 | 14.5 s |
| llama-3.3-70b-instruct | ToolScope@20 | 91.5% | 60.5% | +12.5 pp | 99.5% | 0.888 | 18.2 s |

Heuristic tool-schema tokens (JSON chars ÷ 4): baseline ~60,051 vs BM25@5 ~699, BM25@10 ~1,401, BM25@20 ~2,789, ToolScope@5 ~683, ToolScope@10 ~1,362, ToolScope@20 ~2,700 (~97.7% compression). Latency is one-turn `bind_tools` only; tools are never executed.

## Model-reported prompt lengths (usage)

Two length measures are recorded. **Heuristic tool tokens** (`tokens` / `mean_baseline_tokens`) equal `⌊len(json.dumps(openai_tools)) / 4⌋` and are identical across models for a given bound set — used for compression. **Usage prompt tokens** (`prompt_tokens`) come from the OpenAI-compatible `usage.prompt_tokens` / LangChain `usage_metadata.input_tokens` field after chat+tools templating and are model- and serving-specific.

Full injection always *attempts* to bind all catalogue tools. The harness does not truncate the tool list client-side. HTTP / invoke exceptions are fail-closed as `api_fail` for that condition only; successful responses with no parseable tool call are `parse_fail`. Effective exposure still depends on the configured llama.cpp slot context (`n_ctx`): when the serialized catalogue exceeds the slot, actual usage counts fall well below the heuristic catalogue size.

| Model | Configured n_ctx | Heuristic baseline | Usage mean Baseline | Usage mean BM25@10 | Usage mean ToolScope@10 |
|---|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | 32,768 | 60,051 | 16,424 (n=199; 16411–16454) | 2,018 (n=189) | 1,951 (n=188) |
| llama-3.1-8b-instruct | 32,768 | 60,051 | 16,424 (n=200; 16411–16454) | 2,022 (n=200) | 1,963 (n=200) |
| qwen2.5-7b-instruct | 32,768 | 60,051 | 28,363 (n=200; 28349–28399) | 1,560 (n=200) | 1,512 (n=200) |
| qwen3-32b | 65,536 | 60,051 | 61,115 (n=199; 61101–61151) | 1,554 (n=27) | 1,534 (n=27) |
| llama-3.3-70b-instruct | 131,072 | 60,051 | 81,960 (n=200; 81947–81990) | 2,009 (n=118) | 1,972 (n=118) |

Usage counts can be missing on `api_fail` rows (no successful response metadata). Compression ratios continue to use the heuristic catalogue measure so they stay comparable across models.

## AST accuracy

Name selection does not close the AST gap. Leftover error after a correct name is almost entirely `bad_args`.

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 2.0% | 47.5% | 47.0% | 44.5% | 47.5% | 46.5% | 44.0% |
| llama-3.1-8b-instruct | 3.5% | 49.5% | 50.5% | 49.0% | 49.0% | 52.0% | 52.0% |
| qwen2.5-7b-instruct | 23.5% | 52.0% | 53.5% | 55.0% | 51.5% | 53.5% | 56.0% |
| qwen3-32b | 45.0% | 52.5% | 55.5% | 57.5% | 54.5% | 55.0% | 56.0% |
| llama-3.3-70b-instruct | 46.5% | 60.0% | 60.0% | 60.0% | 60.0% | 61.0% | 60.5% |

## AST given correct name

| Model | Baseline | BM25@5 | BM25@10 | BM25@20 | ToolScope@5 | ToolScope@10 | ToolScope@20 |
|---|---|---|---|---|---|---|---|
| llama-3.2-3b-instruct | 80.0% | 55.9% | 55.0% | 53.9% | 56.2% | 55.0% | 52.7% |
| llama-3.1-8b-instruct | 58.3% | 55.6% | 55.2% | 54.7% | 54.1% | 56.5% | 56.2% |
| qwen2.5-7b-instruct | 58.8% | 62.3% | 62.2% | 62.1% | 60.9% | 61.5% | 64.0% |
| qwen3-32b | 62.1% | 61.8% | 62.4% | 64.2% | 63.0% | 62.5% | 62.9% |
| llama-3.3-70b-instruct | 58.9% | 68.2% | 66.3% | 65.9% | 68.6% | 67.0% | 66.1% |

Once the name is right, ~20–47% of calls still fail AST (`bad_args`). Retrieval does not fix argument quality.

## Where the remaining errors are

Counts. Fully correct (name + AST) is listed first; the rest are the error taxonomy.

| Model | Condition | Fully correct | bad_args | wrong_tool | parse_fail | no_call | retrieval_miss | api_fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b-instruct | Baseline | 4 | 1 | 11 | 183 | 0 | 0 | 1 |
| llama-3.2-3b-instruct | BM25@5 | 95 | 75 | 14 | 0 | 0 | 10 | 6 |
| llama-3.2-3b-instruct | BM25@10 | 94 | 77 | 12 | 0 | 0 | 6 | 11 |
| llama-3.2-3b-instruct | BM25@20 | 89 | 76 | 21 | 0 | 0 | 2 | 12 |
| llama-3.2-3b-instruct | ToolScope@5 | 95 | 74 | 17 | 0 | 0 | 8 | 6 |
| llama-3.2-3b-instruct | ToolScope@10 | 93 | 76 | 16 | 0 | 0 | 3 | 12 |
| llama-3.2-3b-instruct | ToolScope@20 | 88 | 79 | 20 | 0 | 0 | 1 | 12 |
| llama-3.1-8b-instruct | Baseline | 7 | 5 | 27 | 161 | 0 | 0 | 0 |
| llama-3.1-8b-instruct | BM25@5 | 99 | 79 | 12 | 0 | 0 | 10 | 0 |
| llama-3.1-8b-instruct | BM25@10 | 101 | 82 | 11 | 0 | 0 | 6 | 0 |
| llama-3.1-8b-instruct | BM25@20 | 98 | 81 | 19 | 0 | 0 | 2 | 0 |
| llama-3.1-8b-instruct | ToolScope@5 | 98 | 83 | 11 | 0 | 0 | 8 | 0 |
| llama-3.1-8b-instruct | ToolScope@10 | 104 | 80 | 13 | 0 | 0 | 3 | 0 |
| llama-3.1-8b-instruct | ToolScope@20 | 104 | 81 | 14 | 0 | 0 | 1 | 0 |
| qwen2.5-7b-instruct | Baseline | 47 | 33 | 110 | 10 | 0 | 0 | 0 |
| qwen2.5-7b-instruct | BM25@5 | 104 | 63 | 22 | 3 | 0 | 8 | 0 |
| qwen2.5-7b-instruct | BM25@10 | 107 | 65 | 22 | 2 | 0 | 4 | 0 |
| qwen2.5-7b-instruct | BM25@20 | 110 | 67 | 21 | 2 | 0 | 0 | 0 |
| qwen2.5-7b-instruct | ToolScope@5 | 103 | 66 | 22 | 2 | 0 | 7 | 0 |
| qwen2.5-7b-instruct | ToolScope@10 | 107 | 67 | 23 | 0 | 0 | 3 | 0 |
| qwen2.5-7b-instruct | ToolScope@20 | 112 | 63 | 24 | 1 | 0 | 0 | 0 |
| qwen3-32b | Baseline | 90 | 55 | 52 | 2 | 0 | 0 | 1 |
| qwen3-32b | BM25@5 | 105 | 65 | 18 | 5 | 0 | 7 | 0 |
| qwen3-32b | BM25@10 | 111 | 67 | 15 | 3 | 0 | 4 | 0 |
| qwen3-32b | BM25@20 | 115 | 64 | 17 | 4 | 0 | 0 | 0 |
| qwen3-32b | ToolScope@5 | 109 | 64 | 17 | 4 | 0 | 6 | 0 |
| qwen3-32b | ToolScope@10 | 110 | 66 | 20 | 2 | 0 | 2 | 0 |
| qwen3-32b | ToolScope@20 | 112 | 66 | 19 | 3 | 0 | 0 | 0 |
| llama-3.3-70b-instruct | Baseline | 93 | 65 | 40 | 2 | 0 | 0 | 0 |
| llama-3.3-70b-instruct | BM25@5 | 120 | 56 | 14 | 1 | 0 | 9 | 0 |
| llama-3.3-70b-instruct | BM25@10 | 120 | 61 | 13 | 1 | 0 | 5 | 0 |
| llama-3.3-70b-instruct | BM25@20 | 120 | 62 | 16 | 0 | 0 | 2 | 0 |
| llama-3.3-70b-instruct | ToolScope@5 | 120 | 55 | 16 | 2 | 0 | 7 | 0 |
| llama-3.3-70b-instruct | ToolScope@10 | 122 | 60 | 15 | 0 | 0 | 3 | 0 |
| llama-3.3-70b-instruct | ToolScope@20 | 121 | 62 | 16 | 0 | 0 | 1 | 0 |

qwen2.5-7b-instruct's ToolScope@10 name-acc gain is almost entirely fewer `wrong_tool` (110 → 23), not better arguments.

## ToolScope@10 vs baseline name-acc flips

### llama-3.2-3b-instruct

Name acc 2.5% → 84.5% (+82.0 pp). Flips +165 / −1, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_187` GT `whole_foods.check_price`: baseline `—` → ToolScope@10 `whole_foods.check_price` (recall=1). Check the price of tomatoes and lettuce at the Whole Foods in Los Angeles.
- `multiple_101` GT `math.gcd`: baseline `—` → ToolScope@10 `math.gcd` (recall=1). Find the greatest common divisor (GCD) of 12 and 18
- `multiple_193` GT `maps.get_distance_duration`: baseline `—` → ToolScope@10 `maps.get_distance_duration` (recall=1). Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
- `multiple_111` GT `calculate_genotype_frequency`: baseline `—` → ToolScope@10 `calculate_genotype_frequency` (recall=1). What is the genotype frequency of AA genotype in a population, given that allele freque...
- `multiple_13` GT `corporate_finance.revenue_forecast`: baseline `—` → ToolScope@10 `corporate_finance.revenue_forecast` (recall=1). How much revenue would company XYZ generate if we increase the sales units of product A...

Losses (baseline right, retriever wrong):
- `multiple_39` GT `ride_hailing.get_rides`: baseline `ride_hailing.get_rides` → ToolScope@10 `—` (recall=1). Find a ride from New York to Philadelphia with maximum cost of $50

1 of 1 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

### llama-3.1-8b-instruct

Name acc 6.0% → 92.0% (+86.0 pp). Flips +173 / −1, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_66` GT `traffic_estimate`: baseline `—` → ToolScope@10 `traffic_estimate` (recall=1). How much traffic should I expect from Las Vegas to Los Angeles this weekend?
- `multiple_187` GT `whole_foods.check_price`: baseline `—` → ToolScope@10 `whole_foods.check_price` (recall=1). Check the price of tomatoes and lettuce at the Whole Foods in Los Angeles.
- `multiple_101` GT `math.gcd`: baseline `—` → ToolScope@10 `math.gcd` (recall=1). Find the greatest common divisor (GCD) of 12 and 18
- `multiple_193` GT `maps.get_distance_duration`: baseline `geo_distance.calculate` → ToolScope@10 `maps.get_distance_duration` (recall=1). Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
- `multiple_111` GT `calculate_genotype_frequency`: baseline `—` → ToolScope@10 `calculate_genotype_frequency` (recall=1). What is the genotype frequency of AA genotype in a population, given that allele freque...

Losses (baseline right, retriever wrong):
- `multiple_24` GT `route_planner.calculate_route`: baseline `route_planner.calculate_route` → ToolScope@10 `maps.shortest_path` (recall=0). What is the fastest route from London to Edinburgh for playing a chess championship? Al...

### qwen2.5-7b-instruct

Name acc 40.0% → 87.0% (+47.0 pp). Flips +104 / −10, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_187` GT `whole_foods.check_price`: baseline `wholefoods.vegan_products` → ToolScope@10 `whole_foods.check_price` (recall=1). Check the price of tomatoes and lettuce at the Whole Foods in Los Angeles.
- `multiple_193` GT `maps.get_distance_duration`: baseline `route_planner.calculate_route` → ToolScope@10 `maps.get_distance_duration` (recall=1). Get me the travel distance and duration from the Eiffel Tower to the Louvre Museum
- `multiple_111` GT `calculate_genotype_frequency`: baseline `—` → ToolScope@10 `calculate_genotype_frequency` (recall=1). What is the genotype frequency of AA genotype in a population, given that allele freque...
- `multiple_13` GT `corporate_finance.revenue_forecast`: baseline `calculate_return_on_investment` → ToolScope@10 `corporate_finance.revenue_forecast` (recall=1). How much revenue would company XYZ generate if we increase the sales units of product A...
- `multiple_2` GT `country_info.capital`: baseline `get_highest_scoring_player` → ToolScope@10 `country_info.capital` (recall=1). What is the capital of Brazil?

Losses (baseline right, retriever wrong):
- `multiple_99` GT `calculus.derivative`: baseline `calculus.derivative` → ToolScope@10 `calculate_derivative` (recall=1). Calculate the derivative of the function 2x^2 at x = 1.
- `multiple_21` GT `generate_sound_wave`: baseline `generate_sound_wave` → ToolScope@10 `audio.generate` (recall=1). I want to generate a sound of 440Hz frequency for 5 seconds. What is the function and h...
- `multiple_119` GT `database.query`: baseline `database.query` → ToolScope@10 `db_fetch_records` (recall=1). Find records in database in user table where age is greater than 25 and job is 'engineer'.
- `multiple_96` GT `solve_quadratic_equation`: baseline `solve_quadratic_equation` → ToolScope@10 `solve_quadratic` (recall=1). Solve a quadratic equation where a=2, b=6, and c=5
- `multiple_11` GT `math_roots.quadratic`: baseline `math_roots.quadratic` → ToolScope@10 `solve_quadratic` (recall=1). Calculate the roots of a quadratic equation with coefficients 5, 20, and -25

9 of 10 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

### qwen3-32b

Name acc 72.5% → 88.0% (+15.5 pp). Flips +41 / −10, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_101` GT `math.gcd`: baseline `calculate_gcd` → ToolScope@10 `math.gcd` (recall=1). Find the greatest common divisor (GCD) of 12 and 18
- `multiple_13` GT `corporate_finance.revenue_forecast`: baseline `corporate_finance.product_price` → ToolScope@10 `corporate_finance.revenue_forecast` (recall=1). How much revenue would company XYZ generate if we increase the sales units of product A...
- `multiple_170` GT `soccer_stat.get_player_stats`: baseline `player_statistic` → ToolScope@10 `soccer_stat.get_player_stats` (recall=1). Get the player stats of Cristiano Ronaldo in the 2019-2020 season
- `multiple_128` GT `calculate_return_on_equity`: baseline `financial_ratios.calculate_ROE` → ToolScope@10 `calculate_return_on_equity` (recall=1). Calculate the company's return on equity given its net income of $2,000,000, shareholde...
- `multiple_158` GT `religious_history.get_papal_biography`: baseline `religion.history_info` → ToolScope@10 `religious_history.get_papal_biography` (recall=1). Get the biography and main contributions of Pope Innocent III.

Losses (baseline right, retriever wrong):
- `multiple_78` GT `museum_info`: baseline `museum_info` → ToolScope@10 `museum_working_hours.get` (recall=1). Get me information about Natural History Museum in London including timings, exhibition...
- `multiple_99` GT `calculus.derivative`: baseline `calculus.derivative` → ToolScope@10 `calculate_derivative` (recall=1). Calculate the derivative of the function 2x^2 at x = 1.
- `multiple_124` GT `probabilities.calculate_single`: baseline `probabilities.calculate_single` → ToolScope@10 `card_game_probability.calculate` (recall=0). What's the probability of drawing a king from a well shuffled standard deck of 52 cards?
- `multiple_138` GT `legal_case.fetch`: baseline `legal_case.fetch` → ToolScope@10 `law_case_search.find_historical` (recall=1). How to obtain the detailed case information of the R vs Adams legal case?
- `multiple_119` GT `database.query`: baseline `database.query` → ToolScope@10 `db_fetch_records` (recall=1). Find records in database in user table where age is greater than 25 and job is 'engineer'.

9 of 10 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

### llama-3.3-70b-instruct

Name acc 79.0% → 91.0% (+12.0 pp). Flips +28 / −4, McNemar p = < 0.001.

Wins (baseline wrong, retriever right):
- `multiple_126` GT `t_test`: baseline `—` → ToolScope@10 `t_test` (recall=1). Find the statistical significance between two set of variables, dataset_A with the valu...
- `multiple_36` GT `kinematics.calculate_speed_from_rest`: baseline `kinematics.calculate_final_speed` → ToolScope@10 `kinematics.calculate_speed_from_rest` (recall=1). Find out how fast an object was going if it started from rest and traveled a distance o...
- `multiple_153` GT `get_event_date`: baseline `—` → ToolScope@10 `get_event_date` (recall=1). When was the signing of the Treaty of Lisbon?
- `multiple_52` GT `currency_conversion`: baseline `currency_conversion.convert` → ToolScope@10 `currency_conversion` (recall=1). I have 100 euro. How much is it in USD?
- `multiple_73` GT `religion.get_origin`: baseline `religion_origin_get` → ToolScope@10 `religion.get_origin` (recall=1). Who was the founder of Buddhism and where was it originated?

Losses (baseline right, retriever wrong):
- `multiple_192` GT `currency_conversion.convert`: baseline `currency_conversion.convert` → ToolScope@10 `currency_conversion` (recall=1). Convert 150 Euros to Canadian dollars.
- `multiple_68` GT `library.search_books`: baseline `library.search_books` → ToolScope@10 `library.search_book` (recall=1). Can I find a historical fiction book at the New York public library?
- `multiple_97` GT `geometry.area_circle`: baseline `geometry.area_circle` → ToolScope@10 `math.circle_area` (recall=1). What's the area of a circle with a radius of 10?
- `multiple_190` GT `book_hotel`: baseline `book_hotel` → ToolScope@10 `hotel_booking` (recall=1). Book a single room for two nights at the Hilton Hotel in Chicago, starting from 10th De...

4 of 4 losses still have recall = 1: the ground-truth tool was bound and the model preferred a sibling still inside the shortlist.

## Retrieval quality (model-independent)

| Retriever | Recall@10 | NDCG@10 | Missed queries | Mean tokens |
|---|---:|---:|---:|---:|
| BM25@5 | 95.0% | 0.874 | 10 / 200 | 699 |
| BM25@10 | 97.0% | 0.881 | 6 / 200 | 1,401 |
| BM25@20 | 99.0% | 0.886 | 2 / 200 | 2,789 |
| ToolScope@5 | 96.0% | 0.877 | 8 / 200 | 683 |
| ToolScope@10 | 98.5% | 0.885 | 3 / 200 | 1,362 |
| ToolScope@20 | 99.5% | 0.888 | 1 / 200 | 2,700 |

When ToolScope@10 recall is 1, name acc is 85.8% on the first model's traces. When recall is 0, name acc is 0% — the agent cannot call a tool that is not bound.
Missed ground-truth names: `linear_regression`, `probabilities.calculate_single`, `route_planner.calculate_route`.

## Catalog hazards

| Hazard | Count | Effect on scores |
|---|---:|---|
| Same name, different schema (first-seen kept) | 42 records / 33 names | llama-3.2-3b-instruct ToolScope@10 name acc 80.0% on 25 colliding-GT queries vs 85.1% on 175 others; llama-3.1-8b-instruct ToolScope@10 name acc 84.0% on 25 colliding-GT queries vs 93.1% on 175 others; qwen2.5-7b-instruct ToolScope@10 name acc 68.0% on 25 colliding-GT queries vs 89.7% on 175 others; qwen3-32b ToolScope@10 name acc 76.0% on 25 colliding-GT queries vs 89.7% on 175 others; llama-3.3-70b-instruct ToolScope@10 name acc 76.0% on 25 colliding-GT queries vs 93.1% on 175 others |
| Dotted vs underscore aliases after sanitizing | 2 groups | `car.rental` / `car_rental` → `car_rental`; `solve.quadratic_equation` / `solve_quadratic_equation` → `solve_quadratic_equation`. Dedupe keeps first-seen; original_name stays in metadata. |
| Confusable siblings inside top-k | Most remaining `wrong_tool` | Ground truth is retrieved (recall = 1) but the model prefers a near-duplicate still in the shortlist. |

## Analysis

### Selection over injection is necessary, but its value depends on model capability

These results evaluate semantic tool selection — retrieving a small candidate set before model inference — against monolithic injection of a 443-tool catalog. The central pattern is clear: **restricting exposure to k ≈ 10 tools dramatically improves tool-name accuracy while cutting serialized tool-schema context by ~97.7%** (from ~60,051 to ~1,362 tokens). What varies is *how much* improvement matters, because that depends on whether the model could use the full catalog at all.

| Model | Baseline name acc | ToolScope@10 | Δ (pp) | McNemar p |
|---|---:|---:|---:|---|
| llama-3.2-3b-instruct | 2.5% | 84.5% | +82.0 | < 0.001 |
| llama-3.1-8b-instruct | 6.0% | 92.0% | +86.0 | < 0.001 |
| qwen2.5-7b-instruct | 40.0% | 87.0% | +47.0 | < 0.001 |
| qwen3-32b | 72.5% | 88.0% | +15.5 | < 0.001 |
| llama-3.3-70b-instruct | 79.0% | 91.0% | +12.0 | < 0.001 |

The benefit is **monotonic in baseline strength**: the weaker the model under full injection, the larger the absolute gain. But the gain does not vanish at the top of the matrix. Even Llama 3.3 70B — the strongest full-catalog handler at 79% — reaches 91% with ToolScope@10, a statistically significant +12 pp (p < 0.001). Selection is not merely a crutch for models that cannot read large prompts; it remains useful when the catalog is already tractable.

The deployment implication is concrete. An 8B model with retrieval (92.0% name accuracy) **outperforms a 70B model on the full catalog** (79.0%) on the same 200-query benchmark, with two orders of magnitude less tool JSON in context. That comparison holds only under this protocol — one turn, no tool execution, shared BFCL catalog — but it illustrates why tool filtering is an architectural decision, not an optional optimization.

---

### The context-availability paradox manifests in three distinct failure modes

Monolithic injection does not fail uniformly. The error taxonomy reveals three tiers of breakdown:

**Parse collapse (3B–8B Llama).** Llama 3.2 3B and Llama 3.1 8B achieve 2.5% and 6.0% baseline name accuracy. The dominant error is `parse_fail` — 183 and 161 of 200 instances respectively. The harness offers the full ~60,051-heuristic-token catalogue, but model-reported usage under `n_ctx=32,768` is only ~16.4k prompt tokens — so these failures reflect context pressure / incomplete effective exposure, not unconstrained reading of all 443 schemas. Retrieval restores operability: name accuracy jumps to ~85–92%, and `parse_fail` drops to zero.

**Wrong-tool saturation (7B).** Qwen2.5 7B baseline is partially capable (40% name accuracy) but commits 110 `wrong_tool` errors — more than half of all failures. The model calls *something*, but rarely the right function among 443 candidates. ToolScope@10 cuts wrong-tool errors to 23; the +47 pp gain is almost entirely better **selection**, not better arguments.

**Suboptimal but usable baselines (32B–70B).** Qwen3 32B and Llama 3.3 70B handle the full catalog well enough to be deployed without filtering (72.5% and 79.0%), yet both improve meaningfully with retrieval (+15.5 and +12.0 pp). The failure mode shifts from catastrophic breakdown to **confusable-sibling selection** within the shortlist.

Latency reinforces the operational picture. Baseline inference on SLMs takes 28–75 seconds per query; retrieval conditions complete in 1.4–2.9 seconds. Even for 32B and 70B, retrieval reduces per-turn latency. The paradox is not only about accuracy — binding the entire registry imposes a persistent cost that scales with catalog size regardless of whether the model eventually picks correctly.

---

### Retrieval, selection, and calling are separable stages with separable bottlenecks

A single end-to-end score would obscure where the pipeline breaks. Decomposing into stages clarifies what tool RAG solves and what it leaves open.

**Retrieval** is strong but incomplete. At k = 10, ToolScope finds the ground-truth tool in 98.5% of queries (BM25: 97.0%). Three queries miss entirely (`linear_regression`, `probabilities.calculate_single`, `route_planner.calculate_route`); on those, name accuracy is 0% by construction — the model cannot call a tool it never sees. Retrieval quality is model-independent: the same ranked lists are used for every LLM. ToolScope edges BM25 on recall and NDCG, but the gap is modest.

**Selection** is the primary beneficiary of filtering. When the ground-truth tool is in the bound set (recall = 1), aggregate name accuracy is ~85.8%. The remaining ~14% are not retrieval failures — they are **sibling-confusion** errors where the model picks a near-duplicate still in the top-k (`calculus.derivative` vs `calculate_derivative`, `database.query` vs `db_fetch_records`, `currency_conversion.convert` vs `currency_conversion`). On Qwen3 and Llama 70B, 9 of 10 paired losses (baseline correct, retrieval wrong) still have recall = 1. Widening k to 20 improves recall marginally but can **decrease** name accuracy by introducing more siblings into the shortlist. The post-retrieval bottleneck is disambiguation, not recall.

**Calling** (argument correctness) is largely independent of retrieval. AST accuracy at ToolScope@10 ranges from 46.5% (3B) to 61.0% (70B) — far below the corresponding name-accuracy figures. Even conditional on picking the correct tool name, 33–45% of calls still fail the AST check (`bad_args`). Retrieval raises AST scores indirectly by fixing names, but it does not teach the model to fill parameters correctly. For Qwen2.5 7B, the path from 40% to 87% name accuracy adds only ~30 pp of AST improvement (23.5% → 53.5%), confirming that most retrieval gain is routing, not invocation quality.

---

### Lexical and dense retrieval are similarly effective as gates

BM25 and ToolScope produce nearly identical downstream name accuracy at k = 10 (within 0–3 pp for every model). ToolScope has a small advantage in recall (98.5% vs 97.0%) and uses slightly fewer tokens (1,362 vs 1,401 mean), but neither retriever systematically dominates selection quality on this catalog.

This is an important finding for practitioners: **the act of filtering to k ≈ 10 matters far more than the choice between sparse and dense retrieval**. A minimal BM25 gate in front of the model recovers the bulk of the selection benefit. Dense semantic retrieval refines edge cases — the three queries BM25 misses but ToolScope finds — but does not transform the overall picture. Tool RAG should be understood as a class of architectures (retrieve-then-bind) rather than as a single embedding-model choice.

The k-ablation reinforces that k = 10 is a reasonable default. At k = 5, recall drops to 95–96% and name accuracy suffers on hard queries. At k = 20, recall approaches saturation (99%+) but name accuracy does not consistently improve — and can decline for smaller models as more confusable siblings enter the shortlist. The trade-off is between recall coverage and shortlist purity, not between retrieval method and model size.

---

### AgentOps: what becomes observable when injection is replaced by selection

Dynamic tool selection changes what operators can inspect. Under full injection, the model sees 443 tools and the failure mode is opaque — a wrong answer could stem from any of hundreds of definitions. Under retrieval, each turn exposes an explicit candidate set of 10 tools that can be logged, audited, and constrained by policy filters before inference.

The error taxonomy makes this concrete. Failures are attributed to `retrieval_miss` (GT not in shortlist), `wrong_tool` (sibling or unrelated choice), `bad_args` (right name, wrong parameters), `parse_fail` (no valid call), or `api_fail` (inference failure). Across the full matrix, only 23 query-condition pairs remain as `api_fail`; the dominant residual errors are `wrong_tool` and `bad_args`. An operator debugging a failed agent interaction can determine whether to improve the index, add reranking, upgrade the calling model, or add argument validation — rather than treating all failures as undifferentiated "agent errors."

Context compression (~97.7%) and latency reduction are directly measurable efficiency gains. Governance outcomes (compliance, safety enforcement) are not measured here, but the architecture creates the **affordance** for tag-based allow/deny filters and candidate-set auditing that monolithic injection does not provide.

---

### Limitations

These findings apply to a specific experimental setting and should not be over-generalized.

**Benchmark, not production.** The catalog is derived from BFCL V4 Multiple — 443 function definitions with 33 colliding names (same name, different schema) affecting 25 ground-truth queries. Enterprise MCP registries may have different naming conventions, schema quality, and domain structure. The collision sensitivity slice (name accuracy ~4–8 pp lower on affected queries) should be reported alongside headline numbers.

**Single turn, no execution.** Each instance is one `bind_tools` → predict cycle. Tools are never invoked, so multi-step trajectories, state changes, tool-side failures, and end-to-end task completion are outside scope. Selection and argument quality are necessary but not sufficient for reliable agent behaviour.

**Local GGUF models, mixed quantization.** Models are served via llama.cpp on a single DGX Spark node, sequentially, with Q8_0 (SLMs) and Q4_K_M (8B–70B) quantizations. Model size and quantization are not fully disentangled. Results may not transfer to API-hosted models with different context handling, tool-calling formats, or serving infrastructure.

**Single embedder, no reranking.** ToolScope uses `all-MiniLM-L6-v2` without cross-encoder reranking, policy filters, or sticky-session reuse — all of which are supported by the library but not evaluated here. Stronger embedders or rerankers may reduce sibling confusion, the dominant post-retrieval error mode.

**Not an official leaderboard score.** Protocol, agent wrapper, grading pipeline, and model serving differ from the official BFCL generate/eval pipeline. Scores are BFCL-derived and internally consistent, but not comparable to Gorilla leaderboard entries.

---

### Summary

Semantic tool selection over a 443-tool shared catalog produces large, statistically significant improvements in tool-name accuracy across five locally served models (3B–70B), with ~97.7% context compression at k = 10. The magnitude of benefit scales inversely with baseline catalog-handling ability: full injection breaks small models entirely, while larger models still gain meaningfully. Retrieval quality is high (98.5% recall) but not saturated; remaining selection errors are dominated by sibling confusion within the shortlist, not by retrieval misses. Argument correctness remains a separate bottleneck that retrieval does not address. Lexical (BM25) and dense (ToolScope) retrieval perform similarly as pre-inference gates, suggesting that the filtering architecture matters more than the specific ranker. These results support tool RAG as a practical, framework-agnostic strategy for scaling agent tool exposure — but they also delineate its boundary: it solves **which tools the model sees**, not **how the model calls them**.
