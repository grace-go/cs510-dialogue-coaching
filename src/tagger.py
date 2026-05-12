# src/tagger.py

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=5,
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def extract_json(text):
    """
    Safely extract JSON from the model output.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    return json.loads(text[start:end + 1])


def tag_user_input(transcript, scenario, tag_definitions):
    strength_tags = tag_definitions["strength_tags"]
    weakness_tags = tag_definitions["weakness_tags"]

    prompt = f"""
You are a strict dialogue-coaching tag classifier.

Scenario:
{scenario}

User transcript:
{transcript}

Allowed strength tags:
{strength_tags}

Allowed weakness tags:
{weakness_tags}

Task:
Choose the most appropriate tags for the transcript.

Rules:
- Use ONLY tags from the allowed lists.
- Select 0 to 3 strength tags.
- Select 1 to 3 weakness tags.
- Do not invent new tags.
- If the transcript has no clear strength, return an empty strength_tags list.
- Return ONLY valid JSON.

Output format:
{{
  "strength_tags": [],
  "weakness_tags": []
}}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify spoken coaching responses into controlled tags."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    output_text = response.choices[0].message.content
    parsed = extract_json(output_text)

    parsed["strength_tags"] = [
        tag for tag in parsed.get("strength_tags", [])
        if tag in strength_tags
    ]

    parsed["weakness_tags"] = [
        tag for tag in parsed.get("weakness_tags", [])
        if tag in weakness_tags
    ]

    if not parsed["weakness_tags"]:
        parsed["weakness_tags"] = ["missing_details"]

    return parsed