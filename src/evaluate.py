# src/evaluate.py
#
# LLM-as-judge evaluation for the 5 systems produced by run_experiment.py.
# Reads data/outputs/results.json, scores every (input, system) pair on the
# anchored 5-dimension rubric, and writes evaluation.json + evaluation.csv.
#
# Known limitations:
#   - Generator and judge are both llama-3.3-70b-versatile (self-preference bias
#     is symmetric across systems, so within-experiment comparisons remain valid).
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

# Groq exposes an OpenAI-compatible endpoint, so we reuse the OpenAI SDK.
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=5,
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


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
5 - Quotes verbatim phrases from the transcript AND ties multiple distinct feedback points to those specific quoted lines.
4 - References concrete parts of the transcript in most points (paraphrased rather than quoted).
3 - Some transcript-specific references, but meaningful portions rely on generic phrasing that could apply to any transcript.
2 - Mostly generic; only one or two vague references to the transcript.
1 - No reference to the transcript at all.

=== actionability — are suggestions concrete and executable? ===
5 - Every suggestion is paired with concrete example phrasing the speaker could use verbatim (e.g., "Instead of X, say 'Y'"). No abstract advice remains.
4 - Most suggestions are concrete; one or two remain abstract.
3 - Roughly half concrete, half abstract; suggestions can be followed but require interpretation.
2 - Mostly abstract advice ("be more specific", "improve structure") with rare execution detail.
1 - No actionable path — only critique without instructions on what to do differently.

=== relevance — does it match the predicted weaknesses and scenario? ===
5 - Substantively addresses EVERY predicted weakness tag with dedicated, distinct feedback, AND visibly adapts content to the scenario.
4 - Addresses most weakness tags; scenario fit is clear but not heavily emphasized.
3 - Addresses some weakness tags but misses others, OR feels generic to the scenario.
2 - Misses most weakness tags or feels scenario-agnostic.
1 - Off-topic; neither predicted weaknesses nor scenario are reflected.

=== helpfulness — would this genuinely help someone improve? ===
5 - A speaker following this feedback would produce a substantially stronger next attempt — a clear before/after improvement.
4 - Most of the feedback would meaningfully help; small parts are filler.
3 - Mixed: some helpful elements alongside platitudes; net improvement modest.
2 - Vague encouragement with minimal practical lift.
1 - Following it would not measurably improve the response.

=== groundedness — grounded in evidence, not generic LLM filler? ===
5 - Multiple concrete examples, named principles, or specific coaching frames (e.g., STAR, "show-don't-tell") applied to the actual transcript content.
4 - Some grounding with specific examples or principles, though not throughout.
3 - Mixed: occasional grounding amid generic statements.
2 - Mostly generic LLM filler; rare concrete grounding.
1 - Entirely generic statements; no examples or evidence."""


def format_retrieved_for_judge(retrieved_docs):
    # Compact view of what the RAG system had access to. Only the fields the
    # judge needs to recognize grounding — id/type/tags/text — not retrieval
    # scores or scenario duplicates.
    lines = []
    for i, doc in enumerate(retrieved_docs, start=1):
        lines.append(
            f"[Doc {i}] id={doc.get('id')} type={doc.get('type')} "
            f"weakness_tags={doc.get('weakness_tags', [])}\n"
            f"  {doc.get('text', '')}"
        )
    return "\n".join(lines)


def build_judge_prompt(transcript, scenario, predicted_tags, feedback, retrieved_docs=None):
    # Weakness tags are what the feedback SHOULD address; surfacing them in the
    # prompt anchors the relevance dimension to concrete targets.
    strength_tags = predicted_tags.get("strength_tags", [])
    weakness_tags = predicted_tags.get("weakness_tags", [])

    # For RAG systems, surface what was retrieved so the judge can recognize
    # when feedback is genuinely grounded in retrieved coaching evidence vs.
    # generic LLM advice. Omitted entirely for baseline (no retrieval).
    if retrieved_docs:
        retrieved_section = f"""

Retrieved coaching context (what this system had access to when generating feedback):
{format_retrieved_for_judge(retrieved_docs)}

For 'groundedness', credit feedback that meaningfully draws on this retrieved coaching evidence — specific examples, strategies, or patterns visible in the retrieved docs above — rather than only generic LLM advice. Do not penalize the system for not citing every retrieved doc, but reward visible use of the retrieved material."""
    else:
        retrieved_section = ""

    return f"""You are a strict evaluator of dialogue-coaching feedback for CS students.

Scenario:
{scenario}

User transcript:
{transcript}

Predicted weakness tags (what the feedback SHOULD address):
{weakness_tags}

Predicted strength tags:
{strength_tags}
{retrieved_section}

Feedback to evaluate:
{feedback}

Score the feedback on each rubric dimension using the anchored 1-5 scale below.

Calibration (read carefully — this changes how you should score):
- Default to 3 unless there is clear evidence for a higher or lower score.
- Reserve 5 for feedback that EXCEEDS the anchor description — exemplary, not merely competent.
- Most real feedback falls in the 2-4 range. A row of all 5s almost always means the rubric was not applied strictly.
- Actively search for what is MISSING or GENERIC, not just what is present.
- If you cannot cite a specific reason for a 5, the score is 4 or lower.

{RUBRIC_ANCHORS}

Rules:
- Ignore length, verbosity, and writing-style polish.
- Penalize generic filler, vague platitudes, and feedback that could apply to any transcript.
- In each rationale, cite a SPECIFIC strength or weakness of the feedback (not a summary of what the feedback says).
- Give the one-sentence rationale per dimension BEFORE the score.
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
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
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

    return response.choices[0].message.content


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
    # None for baseline (no retrieval); list of docs for RAG variants.
    retrieved_docs = result_item[system_name].get("retrieved_docs")

    prompt = build_judge_prompt(transcript, scenario, predicted_tags, feedback, retrieved_docs)
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