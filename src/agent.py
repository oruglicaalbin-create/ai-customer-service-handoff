"""The LLM agent: understand -> gather (tools) -> decide.

One customer message goes in; out comes a structured decision:
  intent, confidence, decision (resolve / clarify / handoff), a reply to the
  customer and, for handoffs, a summary for the human agent.

Flow of one message (a "tool-use loop"):
  1. Send the message + tool descriptions to Claude.
  2. If Claude asks to call a tool, run the matching Python function from
     systems.py and send the result back.
  3. Repeat until Claude gives its final answer (as JSON, enforced by a schema).
  4. Apply the business rules in code (policy.py). The LLM suggests; code decides.

Every result is cached to disk (cache/agent/), so re-running costs nothing.
Run a demo: python src/agent.py
"""
import hashlib
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import systems
from load_data import load_banking77
from policy import ALWAYS_HANDOFF, must_handoff

load_dotenv()  # puts ANTHROPIC_API_KEY from .env into the environment

MODEL = "claude-haiku-4-5-20251001"
PRICE_PER_TOKEN_USD = {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000}  # Haiku 4.5
MAX_ROUNDS = 6  # safety limit on tool-call rounds per message
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "agent"

INTENTS = sorted(load_banking77("train")["intent"].unique())

# --- Tools: what Claude is told it can call ---------------------------------
# Note: no tool takes a customer_id. The code fills in the logged-in customer,
# so the model cannot look up anyone else's data, even if a message asks it to.
TOOLS = [
    {
        "name": "get_customer_profile",
        "description": "Get the logged-in customer's profile: name, account status "
                       "(active/restricted), whether identity is verified, balance.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_cards",
        "description": "Get the logged-in customer's cards with status (active, not_activated, "
                       "in_delivery, blocked_pin, frozen, expired), expiry and delivery estimate.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_transactions",
        "description": "Get the logged-in customer's 10 most recent transactions (card payments, "
                       "transfers, top-ups, cash withdrawals, direct debits), newest first. "
                       "Filter by status, or use 'any' for all.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["any", "completed", "pending", "declined", "reverted", "failed"],
                },
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    },
]


def run_tool(name: str, tool_input: dict, customer_id: str) -> dict:
    """Call the Python function behind a tool, always for the logged-in customer."""
    if name == "get_customer_profile":
        # Data minimisation (GDPR): the model gets only the fields it needs.
        # Email isn't needed, and the profile language made the model reply in
        # Finnish to English messages.
        profile = systems.get_customer(customer_id)
        return {k: v for k, v in profile.items() if k not in ("email", "language")}
    if name == "get_cards":
        return systems.get_cards(customer_id)
    if name == "get_transactions":
        status = tool_input.get("status", "any")
        return systems.get_transactions(customer_id, status=None if status == "any" else status)
    return {"error": f"Unknown tool {name}"}


# --- Final answer format (structured output) ---------------------------------
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "intent": {"type": "string", "enum": INTENTS},
        "confidence": {"type": "number"},
        "decision": {"type": "string", "enum": ["resolve", "clarify", "handoff"]},
        "customer_reply": {"type": "string"},
        "handoff_summary": {"type": "string"},
    },
    "required": ["reason", "intent", "confidence", "decision", "customer_reply", "handoff_summary"],
    "additionalProperties": False,
}

# The bank's official service facts. Business staff can edit this file without touching code.
BANK_FACTS = (Path(__file__).resolve().parent.parent / "data" / "bank_facts.md").read_text(encoding="utf-8")

SYSTEM_PROMPT = f"""You are a customer-service assistant at ExampleBank. The customer is logged in \
to the bank's app, so you can look up their data with the tools.

For each message:
1. Understand: pick the single best intent from this list: {", ".join(INTENTS)}
   Judge the intent from the message text alone, as a person reading only the message would. \
Do not change the intent based on what the customer's data shows.
2. Gather: call tools when the customer's data could help. Don't call tools you don't need.
3. Decide one of:
   - resolve: you can answer from the bank facts below and the customer's data.
   - clarify: the message itself lacks information you need to answer; ask ONE specific question.
   - handoff: a human must take over. Always hand off these intents: {", ".join(ALWAYS_HANDOFF)}. \
Also hand off if the data shows a problem you cannot fix.
   The systems may not yet show very recent events. If the customer describes something the data doesn't \
show, answer from the bank facts instead of questioning the customer about it.

Only state numbers, fees, limits, times or app steps that appear in the bank facts or the tool results. \
If the answer needs anything else, say a colleague will confirm it.

<bank_facts>
{BANK_FACTS}
</bank_facts>

Final answer fields:
- reason: one sentence on how you chose the intent and decision.
- intent: from the list.
- confidence: your probability (0 to 1) that the intent is correct. Be honest; use low values when unsure.
- decision: resolve, clarify or handoff.
- customer_reply: the message to the customer (the answer, the question, or that a colleague will take over). \
Always write it in the same language as the customer's message. Be brief and friendly.
- handoff_summary: for handoff, 1-3 sentences for the human agent with the key facts; otherwise empty string."""

# Changing the prompt, tools, schema or model automatically invalidates the cache.
PROMPT_VERSION = hashlib.sha256(
    json.dumps([MODEL, SYSTEM_PROMPT, TOOLS, OUTPUT_SCHEMA], sort_keys=True).encode()
).hexdigest()[:12]

# The simulated systems' data version (see systems.py). Each cached result records it;
# a result made with different data is treated as stale and recomputed.
DATA_VERSION = systems.DATA_VERSION

# --- Hybrid mode: the classifier supplies the intent, the LLM gathers data and replies.
# The rules are the frozen prompt's rules word for word; only the intent part differs.
HYBRID_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "decision": {"type": "string", "enum": ["resolve", "clarify", "handoff"]},
        "customer_reply": {"type": "string"},
        "handoff_summary": {"type": "string"},
    },
    "required": ["reason", "decision", "customer_reply", "handoff_summary"],
    "additionalProperties": False,
}

HYBRID_SYSTEM_PROMPT = f"""You are a customer-service assistant at ExampleBank. The customer is logged in \
to the bank's app, so you can look up their data with the tools.

The bank's intent classifier has already identified what the message is about; it is given before the \
message. Treat it as the topic of the message and do not re-classify it.

For each message:
1. Gather: call tools when the customer's data could help. Don't call tools you don't need.
2. Decide one of:
   - resolve: you can answer from the bank facts below and the customer's data.
   - clarify: the message itself lacks information you need to answer; ask ONE specific question.
   - handoff: a human must take over. Hand off if the data shows a problem you cannot fix.
   The systems may not yet show very recent events. If the customer describes something the data doesn't \
show, answer from the bank facts instead of questioning the customer about it.

Only state numbers, fees, limits, times or app steps that appear in the bank facts or the tool results. \
If the answer needs anything else, say a colleague will confirm it.

<bank_facts>
{BANK_FACTS}
</bank_facts>

Final answer fields:
- reason: one sentence on how you chose the decision.
- decision: resolve, clarify or handoff.
- customer_reply: the message to the customer (the answer, the question, or that a colleague will take over). \
Always write it in the same language as the customer's message. Be brief and friendly.
- handoff_summary: for handoff, 1-3 sentences for the human agent with the key facts; otherwise empty string."""

HYBRID_VERSION = hashlib.sha256(
    json.dumps([MODEL, HYBRID_SYSTEM_PROMPT, TOOLS, HYBRID_OUTPUT_SCHEMA], sort_keys=True).encode()
).hexdigest()[:12]
HYBRID_CACHE_DIR = CACHE_DIR.parent / "hybrid"

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _client


def _call_claude(message: str, customer_id: str, system: str = SYSTEM_PROMPT,
                 schema: dict = OUTPUT_SCHEMA) -> dict:
    """Run the tool-use loop for one message. Returns the raw LLM result + usage."""
    messages = [{"role": "user", "content": message}]
    tools_called = []
    tokens = {"input": 0, "output": 0}

    for _ in range(MAX_ROUNDS):
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=1024,
            # temperature 0 = as deterministic as possible, for reproducible evaluation.
            # SDK 1.x dropped the keyword; Haiku 4.5 still accepts it via extra_body.
            extra_body={"temperature": 0},
            system=system,
            tools=TOOLS,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=messages,
        )
        tokens["input"] += response.usage.input_tokens
        tokens["output"] += response.usage.output_tokens

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_called.append(block.name)
                    output = run_tool(block.name, block.input, customer_id)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, ensure_ascii=False),
                    })
            messages.append({"role": "user", "content": results})  # all results in one message
            continue

        if response.stop_reason == "end_turn":
            text = next(b.text for b in response.content if b.type == "text")
            return {"llm": json.loads(text), "tools_called": tools_called, "tokens": tokens}

        # max_tokens, refusal or anything unexpected
        return {"llm": None, "error": f"stop_reason={response.stop_reason}",
                "tools_called": tools_called, "tokens": tokens}

    return {"llm": None, "error": "too many tool rounds", "tools_called": tools_called, "tokens": tokens}


def _cache_file(message: str, customer_id: str) -> Path:
    key = hashlib.sha256(f"{PROMPT_VERSION}|{customer_id}|{message}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{key}.json"


def _read_cache(cache_file: Path) -> dict | None:
    """A cached result, or None if missing or made with different customer data."""
    if not cache_file.exists():
        return None
    result = json.loads(cache_file.read_text(encoding="utf-8"))
    if result.get("data_version") != DATA_VERSION:
        return None  # stale: the simulated systems' data changed since this was computed
    result["from_cache"] = True  # cost_usd is what it cost originally; nothing paid now
    return result


def is_cached(message: str, customer_id: str) -> bool:
    return _read_cache(_cache_file(message, customer_id)) is not None


def _usage_fields(raw: dict) -> dict:
    tokens = raw["tokens"]
    return {
        "tools_called": raw["tools_called"],
        "input_tokens": tokens["input"],
        "output_tokens": tokens["output"],
        "cost_usd": round(tokens["input"] * PRICE_PER_TOKEN_USD["input"]
                          + tokens["output"] * PRICE_PER_TOKEN_USD["output"], 6),
        "model": MODEL,
        "data_version": DATA_VERSION,
        "from_cache": False,
    }


def _hybrid_cache_file(message: str, customer_id: str, intent: str) -> Path:
    key = hashlib.sha256(f"{HYBRID_VERSION}|{customer_id}|{intent}|{message}".encode()).hexdigest()[:20]
    return HYBRID_CACHE_DIR / f"{key}.json"


def is_cached_hybrid(message: str, customer_id: str, intent: str) -> bool:
    return _read_cache(_hybrid_cache_file(message, customer_id, intent)) is not None


def handle_message_with_intent(message: str, customer_id: str, intent: str) -> dict:
    """Hybrid mode: the classifier's intent is given; the LLM gathers data and writes the reply."""
    cache_file = _hybrid_cache_file(message, customer_id, intent)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    raw = _call_claude(f"Classifier intent: {intent}\n\nCustomer message:\n{message}", customer_id,
                       system=HYBRID_SYSTEM_PROMPT, schema=HYBRID_OUTPUT_SCHEMA)
    llm = raw["llm"]
    if llm is None:  # safe default: a human takes over
        result = {"decision": "handoff", "decision_source": "error: " + raw["error"],
                  "customer_reply": "", "handoff_summary": "", "reason": ""}
    else:
        result = {**llm, "decision_source": "llm"}
    result.update({"message": message, "customer_id": customer_id, "given_intent": intent,
                   "prompt_version": HYBRID_VERSION, **_usage_fields(raw)})
    if llm is not None:  # don't cache failures, so they are retried next time
        HYBRID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def handle_message(message: str, customer_id: str) -> dict:
    """Full pipeline for one message, with disk cache and business rules applied."""
    cache_file = _cache_file(message, customer_id)
    cached = _read_cache(cache_file)
    if cached is not None:
        return cached

    raw = _call_claude(message, customer_id)
    llm = raw["llm"]

    if llm is None:
        # Safe default: if anything goes wrong, a human takes over.
        result = {"intent": None, "confidence": 0.0, "llm_decision": None, "decision": "handoff",
                  "decision_source": "error: " + raw["error"], "customer_reply": "", "handoff_summary": "",
                  "reason": ""}
    else:
        decision, source = llm["decision"], "llm"
        if must_handoff(llm["intent"]) and decision != "handoff":
            decision, source = "handoff", f"policy: {ALWAYS_HANDOFF[llm['intent']]}"
        result = {
            "intent": llm["intent"],
            "confidence": min(max(float(llm["confidence"]), 0.0), 1.0),
            "llm_decision": llm["decision"],
            "decision": decision,
            "decision_source": source,
            "customer_reply": llm["customer_reply"],
            "handoff_summary": llm["handoff_summary"],
            "reason": llm["reason"],
        }

    result.update({"message": message, "customer_id": customer_id,
                   "prompt_version": PROMPT_VERSION, **_usage_fields(raw)})

    if llm is not None:  # don't cache failures, so they are retried next time
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    # Hand-picked demo: messages paired with a customer whose data fits the situation.
    demo = [
        ("I am still waiting on my card?", "C008"),                               # card in delivery
        ("Why was I charged twice for the same thing?", "C002"),                   # duplicate payment
        ("help my money isnt showing up", "C001"),                                 # vague
        ("My card got blocked after typing the wrong PIN, how do I fix it?", "C010"),
        ("Miksi tililleni ei ole vielä tullut rahaa?", "C003"),                    # Finnish
    ]
    paid_now = 0.0
    for message, customer_id in demo:
        r = handle_message(message, customer_id)
        if not r["from_cache"]:
            paid_now += r["cost_usd"]
        print(f"\n[{customer_id}] {message}")
        print(f"  intent:     {r['intent']}  (confidence {r['confidence']})")
        print(f"  decision:   {r['decision']}  (source: {r['decision_source']})")
        print(f"  tools:      {r['tools_called']}")
        print(f"  reply:      {r['customer_reply']}")
        if r["handoff_summary"]:
            print(f"  for human:  {r['handoff_summary']}")
    print(f"\nPaid for API calls in this run: ${paid_now:.4f} (results already in cache/ cost nothing)")
