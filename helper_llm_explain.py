# helper_openai_explain_v1.py
import os, json
from openai import OpenAI          # <-- new import

client = OpenAI()                  # picks up OPENAI_API_KEY from env

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
    You are a cybersecurity analyst. Transform raw edge-importance bullets
    into actionable insights.  

    **Output VALID JSON ONLY** with exactly two arrays:

    red_team  → offensive guidance (how an attacker could exploit the finding)  
                Each object needs keys:
                issue   = short description of the ATTACK opportunity
                why     = why it is attractive FROM THE ATTACKER’S VIEW
                steps   = concrete offensive playbook steps an attacker would take

    blue_team → defensive guidance (how defenders can mitigate)  
                Each object needs keys:
                focus    = defensive focus area
                why      = why it helps FROM THE DEFENDER’S VIEW
                controls = concrete defensive controls / monitoring

    Return only the JSON, no commentary.
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
