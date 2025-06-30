from typing import Set
import pandas as pd
import json
from openai import OpenAI

client = OpenAI()  # picks up OPENAI_API_KEY from env

def build_edge_sentence_fn(users:   Set[int],
                           systems: Set[int],
                           resources:Set[int]):
    """
    Factory: returns a to_sentence(row) function bound to the three ID sets.
    Usage
    -----
    to_sentence = build_edge_sentence_fn(users, systems, resources)
    bullets = [to_sentence(r) for _, r in topk_edges.iterrows()]
    """

    def to_sentence(row: pd.Series) -> str:
        u, v = int(row.src), int(row.dst)
        w    = row.importance
        kind = row.kind                 # 'login' | 'lateral' | 'sys→res'

        if kind == 'login':
            # decide direction for nicer wording
            if u in users and v in systems:
                return f"User {u} logs into system {v} (w={w:.2f})"
            else:
                return f"User {v} logs into system {u} (w={w:.2f})"

        if kind == 'lateral':           # user ↔ user
            return f"User {u} shares creds with user {v} (w={w:.2f})"

        if kind == 'sys→res':           # system → resource
            if u in systems and v in resources:
                return f"System {u} accesses resource {v} (w={w:.2f})"
            else:
                return f"System {v} accesses resource {u} (w={w:.2f})"

        # fallback for any unforeseen kind
        return f"{u} – {v} (w={w:.2f}, kind={kind})"

    return to_sentence

def explain_edges_with_llm(edge_bullets: str,
                           model: str = "gpt-4o-mini",
                           temperature: float = 0.3) -> dict:
    """
    Parameters
    ----------
    edge_bullets : str   Lines like "• User 3 shares creds … (w=0.96)\n• …"
    model        : str   Any chat-completions model name you have access to.
    temperature  : float Lower = more deterministic

    Returns  ->  dict { "red_team": [...], "blue_team": [...] }
    """

    system_prompt = """
    You are an offensive-and-defensive cyber analyst.

    INPUT  – a bullet list whose lines look like
    • User 3 shares creds with user 2  (w=0.96)
    • User 3 logs into system 14       (w=0.94)
    …

    TASK
    ----
    Return VALID JSON only, with the keys *red_team* and *blue_team*.
    Each array element MUST reference **specific node / system / user IDs** that
    appear in the bullets.  Do not speak in generalities.

    Schema:

    {
    "red_team": [
        {
        "issue": "<short attacker opportunity incl. node numbers>",
        "why":   "<why attractive – reference the same IDs>",
        "steps": "<3‒4 concise offensive steps – each mentions at
                    least one explicit ID>"
        },
        …
    ],
    "blue_team": [
        {
        "focus":    "<defensive focus – explicit IDs>",
        "why":      "<why it helps>",
        "controls": "<3‒4 concrete mitigations – mention the IDs or
                    the asset class directly>"
        },
        …
    ]
    }

    Respond with JSON **only** – no markdown, no extra text.
    """.strip()

    user_content = f"Edge bullets:\n{edge_bullets}\n\nReturn JSON only."

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ]
    )

    # ✱ `resp.choices[0].message.content` is the assistant reply
    raw = resp.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON:\n{raw}") from e


# -----------------------------------------------------------------
# demo  (only runs when you execute this file directly)
# -----------------------------------------------------------------
if __name__ == "__main__":
    bullets = """• User 3 shares creds with user 2  (w=0.96)
• User 6 shares creds with user 2  (w=0.94)
• User 3 logs into system 14       (w=0.94)
• User 3 shares creds with user 9  (w=0.94)
• User 8 logs into system 14       (w=0.94)"""

    report = explain_edges_with_llm(bullets)
    print(json.dumps(report, indent=2))
