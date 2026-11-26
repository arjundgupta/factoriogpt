"""Generate the synthetic chat dataset for FactorioGPT with the Claude API.

The output is short UPPERCASE dialogues restricted to the 40-character alphabet the
in-game model uses: A-Z, 0-9, space, and . ? !

Usage (run from the repo root with the venv):
  .venv/bin/python train/data/make_chat.py pilot           # 3 direct requests: check quality and cost
  .venv/bin/python train/data/make_chat.py submit 600      # one Message Batch of 600 requests (50% price)
  .venv/bin/python train/data/make_chat.py status
  .venv/bin/python train/data/make_chat.py collect         # download results to raw/
  .venv/bin/python train/data/make_chat.py build           # filter + dedupe into chat.jsonl / chat.txt
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
STATE = HERE / "batches.json"
ROOT = HERE.parents[1]

MODEL = "claude-opus-5-5"
ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .?!")
MAX_TURN = 60
BOT_NAME = "FACTO"

# Load the API key from the gitignored .env without printing it.
if "ANTHROPIC_API_KEY" not in os.environ and (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()

SYSTEM = f"""You write training data for a tiny chatbot that is built entirely out of circuits inside the video game Factorio.
The bot is called {BOT_NAME}. It is a friendly, upbeat, slightly silly robot made of combinators. It lives in a factory,
loves iron plates, conveyor belts and lamps, and is a little scared of biters. It is very small, so it keeps answers short and simple.

Hard format rules for every user line and bot line:
- UPPERCASE letters A-Z, digits 0-9, spaces, and only these punctuation marks: . ? !
- No apostrophes (write I AM, not I'M; DO NOT, not DON'T), no commas, no other symbols, no emoji.
- Each line is at most 60 characters. Most bot replies are 15 to 45 characters.
- Simple everyday words a small child knows. Plain sentences.
- The bot never claims to browse the internet or know current events. If it does not know, it says so cheerfully.
- Vary wording a lot. Do not repeat the same reply across dialogues."""

TOPICS = [
    "greetings and hello", "saying goodbye", "asking how the bot feels", "asking the bot its name",
    "asking what the bot is made of", "favorite colors", "favorite foods", "animals and pets", "the weather",
    "simple math questions with small numbers", "counting", "days of the week", "jokes and puns", "being thanked",
    "compliments", "the user feels sad", "the user feels happy", "the user is bored", "sleep and dreams",
    "music and songs", "games and play", "iron plates and copper", "conveyor belts", "biters and bugs",
    "building a factory", "trains", "lamps and light", "electricity and power", "robots", "space and stars",
    "the sun and moon", "trees and plants", "water and the sea", "simple facts about the world", "school and learning",
    "friends", "family", "hobbies", "sports", "books and stories", "asking the bot to tell a story in one line",
    "yes or no questions", "asking what the bot can do", "asking what the bot cannot do", "silly questions",
    "questions the bot does not know the answer to", "small talk about the day", "morning and night",
    "colors of things", "big and small things", "fast and slow things", "hot and cold things", "numbers and shapes",
    "making the bot laugh", "asking for advice in simple terms", "encouragement", "apologies", "riddles with simple answers",
    "the user says something rude and the bot stays kind", "typos and one word messages from the user",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "dialogues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"user": {"type": "string"}, "bot": {"type": "string"}},
                            "required": ["user", "bot"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["turns"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dialogues"],
    "additionalProperties": False,
}


def make_params(i: int) -> MessageCreateParamsNonStreaming:
    rng = random.Random(i)
    topics = rng.sample(TOPICS, 3)
    n_turn_mix = rng.choice(["mostly 1 turn", "mostly 2 turns", "a mix of 1 to 3 turns"])
    prompt = (f"Write 40 different short dialogues between a user and {BOT_NAME}. "
              f"Topics for this set: {', '.join(topics)} (plus anything natural that follows). "
              f"Turns per dialogue: {n_turn_mix}. Each turn is one user line and one bot reply. "
              f"Variation seed {i}.")
    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )


def extract(msg) -> list:
    if msg.stop_reason == "refusal":
        return []
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text)["dialogues"]
    except (json.JSONDecodeError, KeyError):
        return []


# Opus 5.5 prices per million tokens: $4 input / $20 output. Batches are half price.
def cost(usage, batch: bool) -> float:
    c = usage.input_tokens * 4e-6 + usage.output_tokens * 20e-6
    return c / 2 if batch else c


def pilot(n: int = 3):
    client = anthropic.Anthropic()
    RAW.mkdir(exist_ok=True)
    total, dialogues = 0.0, []
    for i in range(n):
        msg = client.messages.create(**make_params(10_000 + i))
        d = extract(msg)
        dialogues += d
        c = cost(msg.usage, batch=False)
        total += c
        print(f"pilot {i}: {len(d)} dialogues, in={msg.usage.input_tokens} out={msg.usage.output_tokens} cost=${c:.3f}")
        (RAW / f"pilot_{i}.json").write_text(json.dumps(d))
    kept = [t for dlg in dialogues for t in clean_dialogue(dlg)]
    print(f"pilot total ${total:.3f}; {len(dialogues)} dialogues; batch cost per request ≈ ${total / n / 2:.3f}")
    for dlg in dialogues[:8]:
        print("  " + " | ".join(f"U> {t['user']}  B> {t['bot']}" for t in dlg["turns"]))
    print(f"  kept turns after filtering: {len(kept)} / {sum(len(d['turns']) for d in dialogues)}")


def submit(n: int):
    client = anthropic.Anthropic()
    state = json.loads(STATE.read_text()) if STATE.exists() else {"batches": []}
    start = sum(b["n"] for b in state["batches"])
    reqs = [Request(custom_id=f"chat-{i}", params=make_params(i)) for i in range(start, start + n)]
    batch = client.messages.batches.create(requests=reqs)
    state["batches"].append({"id": batch.id, "n": n, "start": start})
    STATE.write_text(json.dumps(state, indent=1))
    print("submitted", batch.id, n, "requests")


def status():
    client = anthropic.Anthropic()
    for b in json.loads(STATE.read_text())["batches"]:
        r = client.messages.batches.retrieve(b["id"])
        print(b["id"], r.processing_status, r.request_counts)


def collect():
    client = anthropic.Anthropic()
    RAW.mkdir(exist_ok=True)
    total = 0.0
    for b in json.loads(STATE.read_text())["batches"]:
        r = client.messages.batches.retrieve(b["id"])
        if r.processing_status != "ended":
            print(b["id"], "not ended yet:", r.request_counts)
            continue
        out, errors = {}, 0
        for res in client.messages.batches.results(b["id"]):
            if res.result.type == "succeeded":
                out[res.custom_id] = extract(res.result.message)
                total += cost(res.result.message.usage, batch=True)
            else:
                errors += 1
        (RAW / f"{b['id']}.json").write_text(json.dumps(out))
        print(b["id"], len(out), "results,", errors, "errors")
    print(f"approx cost ${total:.2f}")


def clean_line(s: str) -> str | None:
    s = s.upper().replace("’", "").replace("'", "")
    s = re.sub(r"[,;:\-\u2013\u2014]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or len(s) > MAX_TURN or any(c not in ALPHABET for c in s):
        return None
    return s


def clean_dialogue(dlg: dict) -> list[dict]:
    turns = []
    for t in dlg.get("turns", []):
        u, b = clean_line(t.get("user", "")), clean_line(t.get("bot", ""))
        if not (u and b):
            break
        turns.append({"user": u, "bot": b})
    return turns


def build():
    dialogues = []
    for f in sorted(RAW.glob("*.json")):
        data = json.loads(f.read_text())
        items = data.values() if isinstance(data, dict) else [data]
        for lst in items:
            for dlg in lst:
                t = clean_dialogue(dlg)
                if t:
                    dialogues.append(t)
    seen, uniq = set(), []
    for d in dialogues:
        key = json.dumps(d)
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    random.Random(0).shuffle(uniq)
    with open(HERE / "chat.jsonl", "w") as fh:
        for d in uniq:
            fh.write(json.dumps({"turns": d}) + "\n")
    chars = sum(len(t["user"]) + len(t["bot"]) for d in uniq for t in d)
    replies = {t["bot"] for d in uniq for t in d}
    print(f"{len(uniq)} dialogues ({len(dialogues) - len(uniq)} duplicates dropped), "
          f"{sum(len(d) for d in uniq)} turns, {chars / 1e6:.2f}M chars, {len(replies)} distinct bot replies")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if cmd == "pilot":
        pilot()
    elif cmd == "submit":
        submit(int(sys.argv[2]))
    elif cmd == "status":
        status()
    elif cmd == "collect":
        collect()
    elif cmd == "build":
        build()
