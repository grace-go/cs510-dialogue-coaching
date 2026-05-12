# src/run_experiment.py

from load_data import load_json, save_json, load_all_corpora
from tagger import tag_user_input
from retriever import (
    BM25Retriever,
    DenseRetriever,
    TagAwareBM25Retriever,
    TagAwareDenseRetriever
)
from generator import generate_feedback


FEEDBACK_PATH = "data/corpus/feedback_corpus.json"
GUIDELINE_PATH = "data/corpus/guideline_corpus.json"
REVISION_PATH = "data/corpus/revision_corpus.json"
TAG_PATH = "data/corpus/tag_definitions.json"

INPUT_PATH = "data/inputs/test_inputs.json"
OUTPUT_PATH = "data/outputs/results.json"


def summarize_retrieved_docs(retrieved_docs):
    return [
        {
            "id": doc.get("id"),
            "type": doc.get("type"),
            "scenario": doc.get("scenario"),
            "score": round(float(score), 4),
            "strength_tags": doc.get("strength_tags", []),
            "weakness_tags": doc.get("weakness_tags", []),
            "text": doc.get("text", "")[:300]
        }
        for doc, score in retrieved_docs
    ]


def run():
    corpus = load_all_corpora(
        feedback_path=FEEDBACK_PATH,
        guideline_path=GUIDELINE_PATH,
        revision_path=REVISION_PATH
    )

    tag_definitions = load_json(TAG_PATH)
    inputs = load_json(INPUT_PATH)[:5]  # Limit to first 5 inputs for testing

    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    tag_bm25 = TagAwareBM25Retriever(corpus)
    tag_dense = TagAwareDenseRetriever(corpus)

    results = []

    for item in inputs:
        input_id = item["id"]
        scenario = item["scenario"]
        transcript = item["transcript"]

        print(f"Running input: {input_id}")

        predicted_tags = tag_user_input(
            transcript=transcript,
            scenario=scenario,
            tag_definitions=tag_definitions
        )

        baseline_feedback = generate_feedback(
            transcript=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            retrieved_docs=None
        )

        bm25_docs = bm25.retrieve(
            query=transcript,
            scenario=scenario,
            top_k=3
        )

        bm25_feedback = generate_feedback(
            transcript=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            retrieved_docs=bm25_docs
        )

        dense_docs = dense.retrieve(
            query=transcript,
            scenario=scenario,
            top_k=3
        )

        dense_feedback = generate_feedback(
            transcript=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            retrieved_docs=dense_docs
        )

        tag_bm25_docs = tag_bm25.retrieve(
            query=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            top_k=3
        )

        tag_bm25_feedback = generate_feedback(
            transcript=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            retrieved_docs=tag_bm25_docs
        )

        tag_dense_docs = tag_dense.retrieve(
            query=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            top_k=3
        )

        tag_dense_feedback = generate_feedback(
            transcript=transcript,
            scenario=scenario,
            query_tags=predicted_tags,
            retrieved_docs=tag_dense_docs
        )

        results.append({
            "id": input_id,
            "scenario": scenario,
            "transcript": transcript,
            "predicted_tags": predicted_tags,

            "baseline": {
                "feedback": baseline_feedback
            },

            "bm25_rag": {
                "retrieved_docs": summarize_retrieved_docs(bm25_docs),
                "feedback": bm25_feedback
            },

            "dense_rag": {
                "retrieved_docs": summarize_retrieved_docs(dense_docs),
                "feedback": dense_feedback
            },

            "tag_bm25_rag": {
                "retrieved_docs": summarize_retrieved_docs(tag_bm25_docs),
                "feedback": tag_bm25_feedback
            },

            "tag_dense_rag": {
                "retrieved_docs": summarize_retrieved_docs(tag_dense_docs),
                "feedback": tag_dense_feedback
            }
        })

    save_json(results, OUTPUT_PATH)
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()