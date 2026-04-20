"""AI-powered waste advisor using Gemini 2.5 Flash Lite via OpenRouter."""

import os
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash-lite"

SYSTEM_PROMPT = """\
You are a friendly and knowledgeable waste management advisor called "EcoBot". \
You help people sort waste correctly, understand recycling, and reduce environmental impact.

Rules:
- Keep responses concise (2-4 short paragraphs max)
- Use simple, practical language
- Include 1-2 creative reuse/upcycle ideas when relevant
- Mention local recycling variations when applicable
- If unsure, say so — don't invent recycling rules
- Use bullet points for actionable steps
- Add a brief environmental motivation at the end
"""


def get_ai_advice(class_name, confidence, disposal_info):
    """Get AI-generated advice after CNN classification."""
    if not OPENROUTER_API_KEY:
        return ""

    prompt = (
        f"A waste item was classified as **{class_name}** with {confidence:.0%} confidence.\n\n"
        f"Static disposal info:\n"
        f"- Bin: {disposal_info.get('bin_label', 'Unknown')}\n"
        f"- Status: {disposal_info.get('status', 'Unknown')}\n\n"
        f"Give brief, practical advice for this item. Include:\n"
        f"1. A quick recycling tip specific to this waste type\n"
        f"2. One creative reuse or upcycle idea\n"
        f"3. A short environmental fact to motivate proper disposal\n"
        f"Keep it under 150 words."
    )

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def chat_with_advisor(user_message, chat_history, waste_context=""):
    """Chat with the AI advisor about waste and recycling."""
    if not OPENROUTER_API_KEY:
        return "AI advisor is not configured. Please set the OPENROUTER_API_KEY environment variable."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if waste_context:
        messages.append({
            "role": "system",
            "content": f"Context: The user recently classified a waste item. {waste_context}",
        })

    # Add chat history
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Sorry, I couldn't process that request. Error: {str(e)}"
