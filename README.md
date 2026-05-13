# cs510-dialogue-coaching

A retrieval-augmented dialogue coaching system for generating feedback on interview responses, elevator pitches, and presentations using Large Language Models (LLMs).

This project explores whether Retrieval-Augmented Generation (RAG) can improve the quality, specificity, relevance, and actionability of AI-generated coaching feedback compared to a standalone LLM baseline.

The system compares multiple retrieval strategies:
- Baseline LLM (no retrieval)
- BM25 sparse retrieval
- Dense retrieval using sentence embeddings
- Tag-aware BM25 retrieval
- Tag-aware dense retrieval

The project was developed for CS 510 (Advanced Information Retrieval).

---

# Project Overview

Given a user transcript, the system:

1. Predicts communication coaching tags
2. Retrieves relevant coaching examples and guidelines
3. Generates structured feedback using an LLM
4. Evaluates generated feedback across different retrieval methods

The corpus contains:
- Feedback examples
- Coaching guidelines
- Revision-pair examples
- Coaching tag definitions

---

# Setup


## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Create `.env` File

Create a `.env` file in the project root directory:

```text
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

---

# Running the Pipeline

## Step 1 — Tag User Inputs

Predict communication coaching tags for each transcript.

```bash
python src/tag_inputs.py
```

Input:
- `data/inputs/test_inputs.json`

Output:
- `data/inputs/tagged_test_inputs.json`

---

## Step 2 — Run Retrieval and Feedback Generation

Generate coaching feedback using all retrieval methods.

```bash
python src/run_experiment.py
```

Output:
- `data/outputs/results.json`

---

## Step 3 — Evaluate Generated Feedback

Evaluate generated feedback using rubric-based scoring.

```bash
python src/evaluate.py
```

Outputs:
- `data/outputs/evaluation.json`
- `data/outputs/evaluation.csv`

---

## Step 4 — Generate Summary Tables

Generate aggregate result tables (mean ± std).

```bash
python src/table_generator.py
```

Output:
- `data/outputs/summary_table.csv`
