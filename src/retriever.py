import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def get_doc_text(doc):
    parts = []

    if doc.get("user_input"):
        parts.append(f"User input: {doc['user_input']}")

    if doc.get("text"):
        parts.append(f"Content: {doc['text']}")

    parts.append(f"Scenario: {doc.get('scenario', '')}")
    parts.append(f"Type: {doc.get('type', '')}")

    strength_tags = " ".join(doc.get("strength_tags", []))
    weakness_tags = " ".join(doc.get("weakness_tags", []))

    parts.append(f"Strength tags: {strength_tags}")
    parts.append(f"Weakness tags: {weakness_tags}")

    return "\n".join(parts)


def scenario_match(doc, scenario):
    return doc.get("scenario") == scenario or doc.get("scenario") == "general"


def tag_overlap_score(query_tags, doc):
    query_strength = set(query_tags.get("strength_tags", []))
    query_weakness = set(query_tags.get("weakness_tags", []))

    doc_strength = set(doc.get("strength_tags", []))
    doc_weakness = set(doc.get("weakness_tags", []))

    strength_overlap = len(query_strength.intersection(doc_strength))
    weakness_overlap = len(query_weakness.intersection(doc_weakness))

    return strength_overlap + weakness_overlap


def filter_by_scenario(corpus, scenario):
    return [doc for doc in corpus if scenario_match(doc, scenario)]


def filter_by_tags(corpus, query_tags, min_overlap=1):
    filtered = []

    for doc in corpus:
        score = tag_overlap_score(query_tags, doc)

        if score >= min_overlap:
            filtered.append((doc, score))

    return filtered


class BM25Retriever:
    def __init__(self, corpus):
        self.corpus = corpus

    def retrieve(self, query, scenario, top_k=5):
        candidate_docs = filter_by_scenario(self.corpus, scenario)

        if not candidate_docs:
            candidate_docs = self.corpus

        texts = [get_doc_text(doc) for doc in candidate_docs]
        tokenized_texts = [text.lower().split() for text in texts]

        bm25 = BM25Okapi(tokenized_texts)
        scores = bm25.get_scores(query.lower().split())

        ranked = sorted(
            zip(candidate_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]


class TagAwareBM25Retriever:
    def __init__(self, corpus, tag_weight=2.0):
        self.corpus = corpus
        self.tag_weight = tag_weight

    def retrieve(self, query, scenario, query_tags, top_k=5):
        scenario_docs = filter_by_scenario(self.corpus, scenario)

        if not scenario_docs:
            scenario_docs = self.corpus

        tagged_docs = filter_by_tags(scenario_docs, query_tags, min_overlap=1)

        if tagged_docs:
            candidate_docs = [doc for doc, _ in tagged_docs]
            tag_scores = [score for _, score in tagged_docs]
        else:
            candidate_docs = scenario_docs
            tag_scores = [0 for _ in candidate_docs]

        texts = [get_doc_text(doc) for doc in candidate_docs]
        tokenized_texts = [text.lower().split() for text in texts]

        bm25 = BM25Okapi(tokenized_texts)
        bm25_scores = bm25.get_scores(query.lower().split())

        combined = []

        for doc, bm25_score, tag_score in zip(candidate_docs, bm25_scores, tag_scores):
            final_score = float(bm25_score) + self.tag_weight * tag_score
            combined.append((doc, final_score))

        combined = sorted(combined, key=lambda x: x[1], reverse=True)

        return combined[:top_k]


class DenseRetriever:
    def __init__(self, corpus, model_name="all-MiniLM-L6-v2"):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query, scenario, top_k=5):
        candidate_docs = filter_by_scenario(self.corpus, scenario)

        if not candidate_docs:
            candidate_docs = self.corpus

        texts = [get_doc_text(doc) for doc in candidate_docs]

        doc_embeddings = self.model.encode(texts, normalize_embeddings=True)
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        scores = np.dot(doc_embeddings, query_embedding)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [(candidate_docs[i], float(scores[i])) for i in ranked_indices]


class TagAwareDenseRetriever:
    def __init__(self, corpus, model_name="all-MiniLM-L6-v2", tag_weight=0.3):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self.tag_weight = tag_weight

    def retrieve(self, query, scenario, query_tags, top_k=5):
        scenario_docs = filter_by_scenario(self.corpus, scenario)

        if not scenario_docs:
            scenario_docs = self.corpus

        tagged_docs = filter_by_tags(scenario_docs, query_tags, min_overlap=1)

        if tagged_docs:
            candidate_docs = [doc for doc, _ in tagged_docs]
            tag_scores = [score for _, score in tagged_docs]
        else:
            candidate_docs = scenario_docs
            tag_scores = [0 for _ in candidate_docs]

        texts = [get_doc_text(doc) for doc in candidate_docs]

        doc_embeddings = self.model.encode(texts, normalize_embeddings=True)
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        dense_scores = np.dot(doc_embeddings, query_embedding)

        combined = []

        for doc, dense_score, tag_score in zip(candidate_docs, dense_scores, tag_scores):
            final_score = float(dense_score) + self.tag_weight * tag_score
            combined.append((doc, final_score))

        combined = sorted(combined, key=lambda x: x[1], reverse=True)

        return combined[:top_k]