"""Business rules: which intents the assistant must ALWAYS hand off to a human.

These are decided by risk, not by model confidence. All other intents may be
automated, but still pass the confidence threshold (optimised later).
"""

ALWAYS_HANDOFF = {
    # 1. Suspected fraud or security: needs blocking, investigation, identity checks.
    "compromised_card": "fraud/security",
    "lost_or_stolen_card": "fraud/security",
    "lost_or_stolen_phone": "fraud/security",
    "card_payment_not_recognised": "fraud/security",
    "cash_withdrawal_not_recognised": "fraud/security",
    "direct_debit_payment_not_recognised": "fraud/security",
    # 2. Money disputes: the customer claims the bank owes them money.
    "transaction_charged_twice": "money dispute",
    "wrong_amount_of_cash_received": "money dispute",
    # 3. Irreversible or regulatory actions.
    "terminate_account": "irreversible/regulatory",
    "verify_source_of_funds": "irreversible/regulatory",  # anti-money-laundering
    "unable_to_verify_identity": "irreversible/regulatory",
}


def must_handoff(intent: str) -> bool:
    return intent in ALWAYS_HANDOFF


if __name__ == "__main__":
    from load_data import load_banking77

    data = load_banking77("test")
    unknown = set(ALWAYS_HANDOFF) - set(data["intent"])
    assert not unknown, f"Intent names not in BANKING77: {unknown}"

    share = data["intent"].map(must_handoff).mean()
    print(f"{len(ALWAYS_HANDOFF)} of {data['intent'].nunique()} intents are always handed off")
    print(f"= {share:.1%} of test messages go to a human by rule")
