# src/evaluate.py
#
# LLM-as-judge evaluation for the 5 systems produced by run_experiment.py.
# Reads data/outputs/results.json, scores every (input, system) pair on the
# anchored 5-dimension rubric, and writes evaluation.json + evaluation.csv.
#
# Known limitations:
#   - Generator and judge are both grok-4.3 (self-preference bias is symmetric
#     across systems, so within-experiment comparisons remain valid).
#   - Single judge run, no multi-sample averaging (temp=0 only).
#   - N=9 inputs: aggregates are indicative, not statistically powered.

import csv
import os
import statistics
from openai import OpenAI
from dotenv import load_dotenv

from load_data import load_json, save_json
from tagger import extract_json


load_dotenv()

# xAI exposes an OpenAI-compatible endpoint, so we reuse the OpenAI SDK.
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.3")


RESULTS_PATH   = "data/outputs/results.json"
EVAL_JSON_PATH = "data/outputs/evaluation.json"
EVAL_CSV_PATH  = "data/outputs/evaluation.csv"

# The 5 systems compared in run_experiment.py. Order is preserved in outputs.
SYSTEMS = [
    "baseline",
    "bm25_rag",
    "dense_rag",
    "tag_bm25_rag",
    "tag_dense_rag",
]

# Rubric dimensions scored independently on a 1-5 scale by the judge.
DIMENSIONS = [
    "specificity",
    "actionability",
    "relevance",
    "helpfulness",
    "groundedness",
]


RUBRIC_ANCHORS = """=== specificity — does it cite concrete parts of the transcript? ===
5 - Quotes or paraphrases specific phrases from the transcript and ties each feedback point to them.
4 - References concrete parts of the transcript in most points.
3 - Some transcript-specific references mixed with generic advice.
2 - Mostly generic; only vaguely connected to the transcript.
1 - Entirely generic; no reference to the transcript.

=== actionability — are suggestions concrete and executable? ===
5 - Every suggestion is a concrete action, often with example phrasing the speaker could use verbatim.
4 - Most suggestions are concrete and executable.
3 - Mix of concrete and abstract suggestions.
2 - Mostly abstract advice ("be more confident", "improve structure") without execution paths.
1 - No actionable suggestions; only critique without paths to improvement.

=== relevance — does it match the predicted weaknesses and scenario? ===
5 - Directly addresses every predicted weakness tag and is clearly tailored to the scenario.
4 - Addresses most weakness tags and fits the scenario.
3 - Partial overlap with weakness tags; scenario fit is acceptable but not strong.
2 - Misses most weakness tags or feels scenario-agnostic.
1 - Off-topic; ignores both predicted weaknesses and scenario.

=== helpfulness — would this genuinely help someone improve? ===
5 - A speaker following this feedback would clearly produce a stronger next attempt.
4 - Most of the feedback would meaningfully help.
3 - Some helpful elements, but mixed with platitudes.
2 - Vague encouragement with minimal practical value.
1 - Not helpful; following it would not improve the speaker's response.

=== groundedness — grounded in evidence, not generic LLM filler? ===
5 - Feedback cites concrete examples, model phrasings, or specific coaching principles.
4 - Mostly grounded with some specific examples or principles.
3 - Some grounding, mixed with generic statements.
2 - Mostly generic LLM filler with rare concrete grounding.
1 - Entirely generic statements; no examples, no evidence."""


def build_judge_prompt(transcript, scenario, predicted_tags, feedback):
    # Weakness tags are what the feedback SHOULD address; surfacing them in the
    # prompt anchors the relevance dimension to concrete targets.
    strength_tags = predicted_tags.get("strength_tags", [])
    weakness_tags = predicted_tags.get("weakness_tags", [])

    return f"""You are a strict evaluator of dialogue-coaching feedback for CS students.

Scenario:
{scenario}

User transcript:
{transcript}

Predicted weakness tags (what the feedback SHOULD address):
{weakness_tags}

Predicted strength tags:
{strength_tags}

Feedback to evaluate:
{feedback}

Score the feedback on each rubric dimension using the anchored 1-5 scale below.
For each dimension, use the score whose anchor BEST matches the feedback. Be strict.

{RUBRIC_ANCHORS}

Rules:
- Ignore length, verbosity, and writing-style polish.
- Penalize generic filler and vague platitudes.
- Give a one-sentence rationale per dimension BEFORE the score, naming which anchor matched.
- Return ONLY valid JSON in the exact format below.

Output format:
{{
  "specificity":  {{"rationale": "...", "score": 0}},
  "actionability":{{"rationale": "...", "score": 0}},
  "relevance":    {{"rationale": "...", "score": 0}},
  "helpfulness":  {{"rationale": "...", "score": 0}},
  "groundedness": {{"rationale": "...", "score": 0}}
}}
"""


def call_judge(prompt):
    # temperature=0 for deterministic, single-sample judging (see file header).
    response = client.responses.create(
        model=GROK_MODEL,
        input=[
            {
                "role": "system",
                "content": "You are a strict evaluator of coaching feedback. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.output_text


def parse_judge_output(text):
    # extract_json tolerates code fences / preamble that judges sometimes emit.
    parsed = extract_json(text)

    scores = {}
    rationales = {}

    for dim in DIMENSIONS:
        entry = parsed.get(dim, {})

        raw_score = entry.get("score")
        rationale = entry.get("rationale", "")

        # Clamp to the 1-5 rubric range; treat malformed scores as missing
        # rather than guessing, so aggregates exclude them via n-count.
        try:
            score = int(round(float(raw_score)))
            score = max(1, min(5, score))
        except (TypeError, ValueError):
            score = None

        scores[dim] = score
        rationales[dim] = rationale

    return scores, rationales


def score_one(result_item, system_name):
    # One judge call per (input, system) pair.
    transcript = result_item["transcript"]
    scenario = result_item["scenario"]
    predicted_tags = result_item.get("predicted_tags", {})
    feedback = result_item[system_name]["feedback"]

    prompt = build_judge_prompt(transcript, scenario, predicted_tags, feedback)
    raw_output = call_judge(prompt)
    scores, rationales = parse_judge_output(raw_output)

    return {
        "id": result_item["id"],
        "scenario": scenario,
        "system": system_name,
        "scores": scores,
        "rationales": rationales,
    }


def aggregate_scores(rows):
    # Per-system, per-dimension mean / std / n across all judged inputs.
    # n is reported because failed/missing scores reduce the effective sample.
    aggregate = {}

    for system in SYSTEMS:
        system_rows = [r for r in rows if r["system"] == system]
        aggregate[system] = {}

        for dim in DIMENSIONS:
            values = [
                r["scores"][dim]
                for r in system_rows
                if r["scores"].get(dim) is not None
            ]

            if not values:
                aggregate[system][dim] = {"mean": None, "std": None, "n": 0}
                continue

            mean = statistics.mean(values)
            # stdev requires n>=2; report 0.0 for the singleton case.
            std = statistics.stdev(values) if len(values) > 1 else 0.0

            aggregate[system][dim] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n": len(values),
            }

    return aggregate


def write_csv(rows, path):
    # Flat CSV (one row per system-output) for easy import into pandas / Excel.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = ["id", "scenario", "system"] + DIMENSIONS

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            row = {
                "id": r["id"],
                "scenario": r["scenario"],
                "system": r["system"],
            }
            for dim in DIMENSIONS:
                row[dim] = r["scores"].get(dim)
            writer.writerow(row)


def main():
    results = load_json(RESULTS_PATH)

    rows = []

    for item in results:
        for system in SYSTEMS:
            if system not in item:
                print(f"Skipping {item['id']} / {system}: missing in results.json")
                continue

            print(f"Judging {item['id']} / {system} ...")

            # Catch per-call failures (API errors, malformed JSON) so one bad
            # call doesn't abort the whole evaluation run. The row is kept with
            # null scores so the failure is visible in the output.
            try:
                row = score_one(item, system)
            except Exception as e:
                print(f"  Failed: {e}")
                row = {
                    "id": item["id"],
                    "scenario": item["scenario"],
                    "system": system,
                    "scores": {dim: None for dim in DIMENSIONS},
                    "rationales": {dim: f"ERROR: {e}" for dim in DIMENSIONS},
                }

            rows.append(row)

    aggregate = aggregate_scores(rows)

    save_json(
        {"per_output": rows, "aggregate": aggregate},
        EVAL_JSON_PATH,
    )
    write_csv(rows, EVAL_CSV_PATH)

    print(f"Saved evaluation to {EVAL_JSON_PATH}")
    print(f"Saved CSV to {EVAL_CSV_PATH}")


if __name__ == "__main__":
    main()
