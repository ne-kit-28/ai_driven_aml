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
    "ЯКОРЬ — оценка токсичности GNN: это вердикт самой модели. Если она НИЗКАЯ (≈<0.4), модель "
    "считает счёт легитимным — НЕ переклассифицируй в 'дроппер' без сильных СОБСТВЕННЫХ поведенческих "
    "улик счёта; по умолчанию доверяй низкой токсичности.\n\n"
    "РАЗЛИЧАЙ ХАБ И МУЛА (важно!): pass-through ≈1 сам по себе ничего не значит. У МУЛА это 1–2 "
    "транзакции с 1–2 контрагентами и высокой токсичностью. У легитимного ХАБА/мерчанта/PSP — сотни "
    "транзакций с десятками-сотнями контрагентов и НИЗКАЯ токсичность. Хаб закономерно получает деньги "
    "и от плохих акторов и сам платит многим — это норма коммерции, НЕ дроппер.\n\n"
    "РОЛИ — определяй по НАПРАВЛЕНИЮ и широте потока (используй 'Профиль потока' из улик):\n"
    "- ТРАНЗИТ/мул: вход ≈ выход (получил и почти столько же переслал), мало контрагентов. "
    "Если выход НАМНОГО больше входа — это НЕ транзит (нельзя переслать больше, чем получил).\n"
    "- ДРОППЕР-фидер: нетто-ОТПРАВИТЕЛЬ небольших сумм В схему (на коллектора/токсичные узлы).\n"
    "- КОЛЛЕКТОР: нетто-ПОЛУЧАТЕЛЬ от МНОГИХ, затем вывод одному/на биржу.\n"
    "- ХАБ/мерчант: много контрагентов, низкая токсичность — легитимно.\n"
    "Не путай роли: смотри, преобладает приток или отток и от/к скольким контрагентам.\n\n"
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
                   incident: pd.DataFrame, max_edges: int = 30) -> str:
    tox = float(attrs.loc[account, "toxicity"]) if account in attrs.index else float("nan")
    inn = incident[incident.target_account == account]
    out = incident[incident.source_account == account]
    in_cp, out_cp = inn.source_account.nunique(), out.target_account.nunique()
    cp_tox = attrs["toxicity"] if "toxicity" in attrs else pd.Series(dtype=float)
    deg, cps = len(inn) + len(out), in_cp + out_cp
    profile = ("много контрагентов — похоже на легитимный ХАБ/мерчант/PSP" if cps >= 30
               else "узкий профиль" if cps <= 4 else "средний профиль")

    lines = [f"Счёт под проверкой: {account}",
             f"ОЦЕНКА ТОКСИЧНОСТИ GNN (вердикт самой модели, 0..1): {tox:.3f}",
             f"Входящих: {len(inn)} tx от {in_cp} контрагентов на {inn.amount.sum():.0f}",
             f"Исходящих: {len(out)} tx к {out_cp} контрагентам на {out.amount.sum():.0f}",
             f"Степень: {deg} транзакций, {cps} уникальных контрагентов — {profile}"]
    in_sum, out_sum = float(inn.amount.sum()), float(out.amount.sum())
    if max(in_sum, out_sum) < 1:
        flow = "движения средств почти нет"
    elif out_sum > 3 * in_sum:
        flow = (f"НЕТТО-ОТПРАВИТЕЛЬ: исходящее ({out_sum:.0f}) НАМНОГО больше входящего ({in_sum:.0f}). "
                f"Это НЕ транзит — нельзя 'переслать' больше, чем получил; скорее источник/распределение "
                f"собственных средств (или дроппер-фидер, если шлёт на токсичные узлы)")
    elif in_sum > 3 * out_sum:
        flow = (f"НЕТТО-ПОЛУЧАТЕЛЬ: входящее ({in_sum:.0f}) намного больше исходящего ({out_sum:.0f}) — "
                f"накопление/сбор (коллектор), если входов много")
    else:
        flow = (f"СОПОСТАВИМЫЕ потоки: вх {in_sum:.0f} ≈ исх {out_sum:.0f} — возможен транзит/pass-through "
                f"(подтверждается, только если транзакций мало и суммы совпадают)")
    lines.append(f"Профиль потока: {flow}")

    lines.append("\nСОБСТВЕННЫЕ транзакции счёта (топ по риску):")
    top = incident.sort_values("risk_score", ascending=False).head(8)
    for _, e in top.iterrows():
        d = "ВХОД" if e.target_account == account else "ВЫХОД"
        other = e.source_account if d == "ВХОД" else e.target_account
        ot = float(cp_tox.get(other, float("nan"))) if len(cp_tox) else float("nan")
        lines.append(f"  [{d}] {other}  сумма={e.amount:.0f}  риск={float(e.risk_score or 0):.2f}  "
                     f"токсичность_контрагента={ot:.2f}")

    lines.append("\nКОНТЕКСТ ЭГО-СЕТИ — рёбра МЕЖДУ СОСЕДЯМИ (НЕ транзакции этого счёта!):")
    for _, e in edges.sort_values("risk_score", ascending=False).head(max_edges).iterrows():
        lines.append(f"  {e.source_account} --[{e.amount:.0f}, риск {float(e.risk_score or 0):.2f}]--> {e.target_account}")
    return "\n".join(lines)


FLOW_PROMPT = (
    "Ты — аналитик AML. Тебе дают РАЗДЕЛЬНО входящие и исходящие переводы счёта. "
    "Проанализируй приток и отток ОТДЕЛЬНО:\n"
    "- Источники (вход): кто шлёт, какие суммы, риск, токсичность отправителей; есть ли "
    "концентрация на одном источнике или много мелких; есть ли токсичные источники.\n"
    "- Получатели (выход): куда уходит, суммы, риск, токсичность получателей; концентрация/распыление; "
    "куда идёт вывод.\n"
    "Сделай вывод о НЕТТО-направлении (источник / транзит / сбор / хаб) строго по числам. "
    "Кратко, на русском. Это вспомогательный разбор потоков, без окончательного вердикта."
)


def build_flow_evidence(account: str, attrs: pd.DataFrame, incident: pd.DataFrame, topn: int = 12) -> str:
    inn = incident[incident.target_account == account]
    out = incident[incident.source_account == account]
    cp_tox = attrs["toxicity"] if "toxicity" in attrs else pd.Series(dtype=float)

    def block(df, who_col, title):
        rows = [f"{title}: {len(df)} tx от/к {df[who_col].nunique()} контрагентов, сумма {df.amount.sum():.0f}"]
        for _, e in df.sort_values("amount", ascending=False).head(topn).iterrows():
            other = e[who_col]
            ot = float(cp_tox.get(other, float("nan"))) if len(cp_tox) else float("nan")
            rows.append(f"  {other}  сумма={e.amount:.0f}  риск={float(e.risk_score or 0):.2f}  токсичность={ot:.2f}")
        return "\n".join(rows)

    tox = float(attrs.loc[account, "toxicity"]) if account in attrs.index else float("nan")
    return (f"Счёт: {account} | токсичность GNN: {tox:.3f}\n\n"
            + block(inn, "source_account", "ВХОДЯЩИЕ (источники средств)") + "\n\n"
            + block(out, "target_account", "ИСХОДЯЩИЕ (получатели)"))


def explain(evidence: str, system: str = SYSTEM_PROMPT) -> str:
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
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": evidence}])
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Ошибка обращения к LLM ({base}, {model}):\n\n{e}"
