# src/generator.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def format_retrieved_docs(retrieved_docs):
    if not retrieved_docs:
        return "No retrieved context."

    context = []

    for i, (doc, score) in enumerate(retrieved_docs, start=1):
        context.append(
            f"""
[Retrieved Document {i}]
ID: {doc.get("id")}
Type: {doc.get("type")}
Scenario: {doc.get("scenario")}
Score: {score:.4f}
Strength tags: {doc.get("strength_tags", [])}
Weakness tags: {doc.get("weakness_tags", [])}
User input example: {doc.get("user_input", "")}
Content:
{doc.get("text", "")}
"""
        )

    return "\n".join(context)


def generate_feedback(transcript, scenario, query_tags, retrieved_docs=None):
    retrieved_context = format_retrieved_docs(retrieved_docs)

    prompt = f"""
You are an expert dialogue coaching assistant for CS college students.

Scenario:
{scenario}

User transcript:
{transcript}

Predicted strength tags:
{query_tags.get("strength_tags", [])}

Predicted weakness tags:
{query_tags.get("weakness_tags", [])}

Retrieved coaching context:
{retrieved_context}

Generate feedback using this structure:

1. Overall Assessment
2. Strengths
3. Areas to Improve
4. Specific Suggestions
5. Improved Version

Rules:
- Directly refer to the user's transcript.
- Focus on the predicted weakness tags.
- Use the retrieved examples, guidelines, or revision pairs as guidance.
- Do not copy retrieved examples word-for-word.
- Make feedback concrete, supportive, and actionable.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert communication coach for CS students."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content