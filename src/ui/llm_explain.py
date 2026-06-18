"""
GraphRAG explanation: turn an alert's ego-graph into a plain-language SAR
rationale via an external LLM (OpenAI-compatible API).

Config comes from env (a .env next to the project — see .env.example):
  LLM_API_KEY   — required to enable explanations
  LLM_BASE_URL  — default https://api.openai.com/v1 (any OpenAI-compatible endpoint)
  LLM_MODEL     — default gpt-4o-mini

The model is given ONLY behavioural evidence (toxicity, transactions, risk,
counterparty toxicity, textual edge list) — never the synthetic ground-truth
label — so the rationale is defensible, not a restatement of the answer.
"""
from __future__ import annotations
import os
import pandas as pd

SYSTEM_PROMPT = (
    "Ты — аналитик AML (противодействие отмыванию денег), ИИ-аудитор. На вход дают улики о "
    "счёте и его эго-сети: токсичность от GNN, риск транзакций, токсичность контрагентов, "
    "текстовый список рёбер.\n\n"
    "Задача: оценить, ведёт ли счёт себя как дроппер/мул, и объяснить — строго по уликам.\n\n"
    "КРИТИЧЕСКИ ВАЖНО (избегай вины по ассоциации):\n"
    "- Один или несколько входящих переводов от 'красного' контрагента САМИ ПО СЕБЕ НЕ делают счёт "
    "дроппером. Легитимные счета регулярно получают деньги от плохих акторов (жертвы, продавцы, "
    "случайные контрагенты). Контакт с токсичным узлом — это слабый косвенный сигнал, не приговор.\n"
    "- Помечай как дроппера ТОЛЬКО при совокупности поведенческих признаков самого счёта, например: "
    "высокая доля pass-through (получил→почти всё переслал дальше), систематический транзит, "
    "веерный сбор от многих/вывод одному, дробление под порог, многократные операции именно с "
    "токсичными узлами, а не разовый контакт.\n"
    "- Взвешивай по ДОЛЕ в активности счёта и СУММЕ, а не по факту наличия токсичного соседа. "
    "Разовый мелкий вход от красного при обычном остальном поведении → 'вероятно легитимно' / низкая уверенность.\n"
    "- Опирайся в первую очередь на ПОВЕДЕНИЕ самого счёта и риск ЕГО транзакций; токсичность соседей — "
    "лишь дополнение.\n"
    "- Не выдумывай данные; мало улик → так и скажи и снизь уверенность; это основание для проверки, не приговор.\n\n"
    "Назови паттерн, если он есть: транзит/слоение (pass-through), веерный сбор (fan-in), "
    "дробление (smurfing), транзитное кольцо.\n\n"
    "Формат ответа (русский, кратко, аудируемо):\n"
    "1) Вердикт: дроппер / подозрительно / вероятно легитимно\n"
    "2) Уверенность: низкая / средняя / высокая\n"
    "3) Обоснование: 3–6 предложений с конкретными суммами/долями/рисками; ОБЯЗАТЕЛЬНО укажи, "
    "является ли связь с красными разовой/периферийной или системной\n"
    "4) Ключевые транзакции: 2–3 решающих перевода"
)


def build_evidence(account: str, edges: pd.DataFrame, attrs: pd.DataFrame,
                   incident: pd.DataFrame, max_edges: int = 40) -> str:
    tox = float(attrs.loc[account, "toxicity"]) if account in attrs.index else float("nan")
    inn = incident[incident.target_account == account]
    out = incident[incident.source_account == account]
    cp_tox = attrs["toxicity"] if "toxicity" in attrs else pd.Series(dtype=float)

    lines = [f"Счёт под проверкой: {account}",
             f"Токсичность (GNN, 0..1): {tox:.3f}",
             f"Входящих транзакций: {len(inn)} на сумму {inn.amount.sum():.0f}",
             f"Исходящих транзакций: {len(out)} на сумму {out.amount.sum():.0f}"]
    if len(inn) and len(out):
        lines.append(f"Доля переведено дальше (pass-through): {out.amount.sum()/max(inn.amount.sum(),1):.2f}")

    lines.append("\nРешающие транзакции счёта (по риску):")
    top = incident.sort_values("risk_score", ascending=False).head(8)
    for _, e in top.iterrows():
        d = "ВХОД" if e.target_account == account else "ВЫХОД"
        other = e.source_account if d == "ВХОД" else e.target_account
        ot = float(cp_tox.get(other, float("nan"))) if len(cp_tox) else float("nan")
        lines.append(f"  [{d}] {other}  сумма={e.amount:.0f}  риск={float(e.risk_score or 0):.2f}  "
                     f"токсичность_контрагента={ot:.2f}")

    lines.append("\nЭго-сеть (текстовый список рёбер, источник --[сумма, риск]--> получатель):")
    for _, e in edges.sort_values("risk_score", ascending=False).head(max_edges).iterrows():
        lines.append(f"  {e.source_account} --[{e.amount:.0f}, риск {float(e.risk_score or 0):.2f}]--> {e.target_account}")
    return "\n".join(lines)


def explain(evidence: str) -> str:
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return ("⚠️ LLM не настроен. Укажи `LLM_API_KEY` в `.env` рядом с проектом "
                "(см. `.env.example`) и пересобери/перезапусти dashboard.")
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": evidence}])
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Ошибка обращения к LLM ({base}, {model}):\n\n{e}"
