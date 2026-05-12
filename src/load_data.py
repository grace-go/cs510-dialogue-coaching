import json
from pathlib import Path


def load_json(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_all_corpora(feedback_path, guideline_path, revision_path):
    feedback = load_json(feedback_path)
    guidelines = load_json(guideline_path)
    revisions = load_json(revision_path)

    corpus = []

    for item in feedback:
        item["source_file"] = "feedback_corpus"
        corpus.append(item)

    for item in guidelines:
        item["source_file"] = "guideline_corpus"
        corpus.append(item)

    for item in revisions:
        item["source_file"] = "revision_corpus"
        corpus.append(item)

    return corpus