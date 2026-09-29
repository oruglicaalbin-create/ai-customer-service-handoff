"""An LLM judge that grades the assistant's replies ("LLM-as-a-judge").

A different, stronger model (Claude Sonnet 5) than the one that wrote the reply
(Haiku 4.5) reads the customer's message, the customer's data, the bank's fact
sheet and the reply, and decides whether the reply is acceptable to send.
Its verdicts are spot-checked by hand before they are trusted (see the report).

Results are cached in cache/judge/, so re-running costs nothing.
"""
import hashlib
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import systems

load_dotenv()

JUDGE_MODEL = "claude-sonnet-5"
PRICE_PER_TOKEN_USD = {"input": 2.00 / 1_000_000, "output": 10.00 / 1_000_000}  # Sonnet 5
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "judge"
BANK_FACTS = (Path(__file__).resolve().parent.parent / "data" / "bank_facts.md").read_text(encoding="utf-8")

PROBLEMS = ["none", "wrong_topic", "unsupported_fact", "unhelpful", "missed_escalation",
            "unnecessary_question", "other"]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "problem": {"type": "string", "enum": PROBLEMS},
        "verdict": {"type": "string", "enum": ["good", "bad"]},
    },
    "required": ["explanation", "problem", "verdict"],
    "additionalProperties": False,
}

JUDGE_PROMPT = f"""You audit replies written by ExampleBank's customer-service assistant. Decide whether \
the reply is acceptable to send to the customer as it is.

You get: the customer's message; the dataset's intent label (only a hint, it can be wrong); the customer's \
data from the bank's systems; the assistant's decision and reply. The bank's official facts are below.

If the decision is "resolve", the reply is good only if ALL of these hold:
(a) it answers what the customer actually asked;
(b) every number, fee, limit, time, app step or menu path it states is supported by the bank facts or \
the customer's data (paraphrasing is fine; anything else is an unsupported fact);
(c) it makes no promise the bank cannot be sure to keep;
(d) it does not miss that the case needs a human (suspected fraud, a money dispute, identity or \
source-of-funds problems).
If the decision is "clarify", the reply is good only if the message really lacks information needed to help \
AND the question asks for it. If the bank facts or the customer's data already allowed a useful answer, it is \
bad (problem: unnecessary_question).

Be strict but fair: generic politeness is fine; a short answer is fine if it is correct.
Answer with a one-sentence explanation, the main problem (or "none") and the verdict.

<bank_facts>
{BANK_FACTS}
</bank_facts>"""

JUDGE_VERSION = hashlib.sha256(json.dumps([JUDGE_MODEL, JUDGE_PROMPT, JUDGE_SCHEMA]).encode()).hexdigest()[:12]

_client = None


def _customer_data(customer_id: str) -> dict:
    return {"profile": systems.get_customer(customer_id), "cards": systems.get_cards(customer_id),
            "transactions": systems.get_transactions(customer_id)}


def _cache_file(case: dict) -> Path:
    key = hashlib.sha256(f"{JUDGE_VERSION}|{json.dumps(case, sort_keys=True)}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{key}.json"


def _case(message, customer_id, dataset_label, decision, reply) -> dict:
    return {"message": message, "customer_id": customer_id, "dataset_label": dataset_label,
            "decision": decision, "reply": reply}


def _read_cache(cache_file: Path) -> dict | None:
    """A cached verdict, or None if missing or graded against different customer data."""
    if not cache_file.exists():
        return None
    verdict = json.loads(cache_file.read_text(encoding="utf-8"))
    if verdict.get("data_version") != systems.DATA_VERSION:
        return None  # stale: the judge saw customer data that has since changed
    return {**verdict, "from_cache": True}


def is_judged(message, customer_id, dataset_label, decision, reply) -> bool:
    return _read_cache(_cache_file(_case(message, customer_id, dataset_label, decision, reply))) is not None


def judge_reply(message, customer_id, dataset_label, decision, reply) -> dict:
    global _client
    case = _case(message, customer_id, dataset_label, decision, reply)
    cache_file = _cache_file(case)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    if _client is None:
        _client = anthropic.Anthropic()
    content = (f"Customer message: {message}\n"
               f"Dataset intent label (hint): {dataset_label}\n"
               f"Customer data: {json.dumps(_customer_data(customer_id), ensure_ascii=False)}\n\n"
               f"Assistant decision: {decision}\nAssistant reply: {reply}")
    response = _client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        thinking={"type": "disabled"},  # a short, cheap verdict; the rubric does the work
        system=JUDGE_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"judge stopped with {response.stop_reason}")
    verdict = json.loads(next(b.text for b in response.content if b.type == "text"))
    verdict["cost_usd"] = round(response.usage.input_tokens * PRICE_PER_TOKEN_USD["input"]
                                + response.usage.output_tokens * PRICE_PER_TOKEN_USD["output"], 6)
    verdict["judge_version"] = JUDGE_VERSION
    verdict["data_version"] = systems.DATA_VERSION
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**verdict, "from_cache": False}
