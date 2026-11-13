"""
Чи LLM розуміє ієрархію понять, чи пам'ятає WordNet?

Задача (taxonomy discovery, як Task B у LLMs4OL): для поняття обрати
його безпосередній надклас (hypernym) з 5 варіантів.

Три умови для ТИХ САМИХ поняттів:
  real      — справжня назва + визначення
  renamed   — вигадане слово + те саме визначення (назва в тексті замінена)
  name_only — лише справжня назва, без визначення

real ≈ renamed  -> модель міркує зі змісту
real >> renamed -> модель спирається на запам'ятовану назву

Запуск:
  python experiment.py --mock            # перевірити конвеєр без API
  python experiment.py                   # справжня модель (див. README)
"""

import argparse
import base64
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import random
import re
import string
from pathlib import Path

from nltk.corpus import wordnet as wn

LETTERS = "ABCDE"
CONDITIONS = ["real", "renamed", "name_only"]
COND_LABEL = {
    "real": "Справжня назва + визначення",
    "renamed": "Вигадане слово + визначення",
    "name_only": "Лише назва",
}
OUT = Path("results")


# ---------------------------------------------------------------- dataset

def name_of(syn):
    return syn.lemmas()[0].name().replace("_", " ")


def make_pseudoword(rng):
    cons, vow = "bdfgklmnprstvz", "aeiou"
    return "".join(rng.choice(cons) + rng.choice(vow) for _ in range(3)) + rng.choice(["x", "n", "l", "r"])


def is_ancestor(candidate, syn):
    return any(candidate in path for path in syn.hypernym_paths())


def build_dataset(n, seed):
    rng = random.Random(seed)
    pool = []
    for s in wn.all_synsets("n"):
        lem = s.lemmas()[0]
        if not s.hypernyms() or "_" in lem.name() or not lem.name().isalpha():
            continue
        if lem.count() < 3 or not (5 <= s.min_depth() <= 11):
            continue
        pool.append(s)
    rng.shuffle(pool)

    parents_pool = list({s.hypernyms()[0] for s in pool[:3000]})
    items, used_names = [], set()
    for s in pool:
        if len(items) >= n:
            break
        name = name_of(s)
        parent = s.hypernyms()[0]
        pname = name_of(parent)
        if name.lower() in used_names or name.lower() == pname.lower():
            continue
        definition = s.definition()
        # без підказки: жодна назва надкласу не згадана у визначенні
        if any(re.search(rf"\b{re.escape(l.name().replace('_', ' '))}", definition, re.I) for l in parent.lemmas()):
            continue
        # правдоподібні дистрактори: «сусіди» надкласу в ієрархії (діти дідуся й прадіда),
        # потім — випадкові класи схожої глибини
        near = []
        for anc in parent.hypernyms() + [g for h in parent.hypernyms() for g in h.hypernyms()]:
            near += [c for c in anc.hyponyms() if c != parent]
        rng.shuffle(near)
        far = rng.sample(parents_pool, len(parents_pool))
        distractors, seen = [], {pname.lower(), name.lower()}
        for cand in near + far:
            cn = name_of(cand)
            if cand == parent or is_ancestor(cand, s) or is_ancestor(s, cand) or cn.lower() in seen:
                continue
            if cand not in near and abs(cand.min_depth() - parent.min_depth()) > 2:
                continue
            distractors.append(cand)
            seen.add(cn.lower())
            if len(distractors) == 4:
                break
        if len(distractors) < 4:
            continue
        options = [pname] + [name_of(d) for d in distractors]
        rng.shuffle(options)
        pseudo = make_pseudoword(rng)
        renamed_def = re.sub(rf"\b{re.escape(name)}\b", pseudo, definition, flags=re.I)
        items.append({
            "id": s.name(),
            "term": name,
            "pseudo": pseudo,
            "definition": definition,
            "renamed_definition": renamed_def,
            "options": options,
            "answer": LETTERS[options.index(pname)],
            "parent": pname,
            "n_near_distractors": sum(d in near for d in distractors),
        })
        used_names.add(name.lower())
    return items


# ---------------------------------------------------------------- prompts

def build_prompt(item, cond):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    if cond == "real":
        concept = f'Concept: "{item["term"]}"\nDefinition: {item["definition"]}'
    elif cond == "renamed":
        concept = f'Concept: "{item["pseudo"]}"\nDefinition: {item["renamed_definition"]}'
    else:
        concept = f'Concept: "{item["term"]}"'
    return (
        "You are building a taxonomy (is-a hierarchy) of English nouns.\n"
        f"{concept}\n\n"
        "Which option is the most specific correct parent class of this concept "
        "(i.e. the concept IS A kind of it)?\n"
        f"{opts}\n\n"
        "Answer with a single letter only."
    )


# ---------------------------------------------------------------- models

class MockModel:
    """Імітація для перевірки конвеєра. НЕ є результатом експерименту."""
    name = "mock"
    acc = {"real": 0.85, "renamed": 0.6, "name_only": 0.7}

    def ask(self, prompt, item, cond):
        r = random.Random(hashlib.md5((prompt + cond).encode()).hexdigest())
        return item["answer"] if r.random() < self.acc[cond] else r.choice(LETTERS)


class OpenAICompatModel:
    def __init__(self, model, max_tokens=1024):
        from openai import OpenAI
        self.client = OpenAI()  # OPENAI_API_KEY, OPENAI_BASE_URL з оточення
        self.name = model
        self.use_temperature = True
        self.max_tokens = max_tokens
        self.finish = {}
        self.cache_path = OUT / f"cache_{re.sub(r'[^A-Za-z0-9_.-]', '_', model)}.json"
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}

    def ask(self, prompt, item, cond):
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if self.cache.get(key):  # порожні відповіді не вважаємо закешованими
            return self.cache[key]
        kwargs = dict(model=self.name, messages=[{"role": "user", "content": prompt}], max_tokens=self.max_tokens)
        if self.use_temperature:
            kwargs["temperature"] = 0
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # деякі моделі не приймають temperature
            if "temperature" not in str(e) or "temperature" not in kwargs:
                raise
            self.use_temperature = False
            kwargs.pop("temperature")
            resp = self.client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        fr = resp.choices[0].finish_reason
        self.finish[fr] = self.finish.get(fr, 0) + 1
        if text:
            self.cache[key] = text
        return text

    def save(self):
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")


def parse_letter(text):
    text = (text or "").strip()
    m = (re.search(r"\b(?:answer|option)\s*(?:is)?\s*:?\s*\**\(?([A-E])\b", text, re.I)
         or re.match(r"^[\s*(\[]*([A-E])(?:$|[\s]*[.):\]*])", text)
         or re.search(r"(?<![A-Za-z])([B-E])(?![A-Za-z])", text))
    return m.group(1) if m else None


# ---------------------------------------------------------------- run

def run(items, model, workers):
    jobs = [(it, c) for it in items for c in CONDITIONS]

    def one(job):
        it, c = job
        raw = model.ask(build_prompt(it, c), it, c)
        pred = parse_letter(raw)
        return {"id": it["id"], "term": it["term"], "condition": c, "raw": raw,
                "pred": pred, "answer": it["answer"], "correct": pred == it["answer"]}

    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(one, jobs))
    finally:  # кеш зберігається навіть при падінні — повторний запуск не платить двічі
        if hasattr(model, "save"):
            model.save()
    return rows


def summarize(rows, items):
    by = {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}
    acc = {c: sum(r["correct"] for r in rs) / len(rs) for c, rs in by.items()}
    real = {r["id"]: r["correct"] for r in by["real"]}
    ren = {r["id"]: r["correct"] for r in by["renamed"]}
    flips = {
        "правильно лише зі справжньою назвою": sum(real[i] and not ren[i] for i in real),
        "правильно лише з вигаданим словом": sum(ren[i] and not real[i] for i in real),
        "правильно в обох": sum(real[i] and ren[i] for i in real),
        "помилка в обох": sum(not real[i] and not ren[i] for i in real),
    }
    no_answer = {c: sum(r["pred"] is None for r in rs) for c, rs in by.items()}
    acc_answered = {c: (sum(r["correct"] for r in rs) / max(1, len(rs) - no_answer[c])) for c, rs in by.items()}
    return {"n_items": len(items), "accuracy": acc, "no_answer": no_answer, "accuracy_answered": acc_answered, "chance": 0.2, "flips": flips}


# ---------------------------------------------------------------- report

def chart_png(summary, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    vals = [summary["accuracy"][c] * 100 for c in CONDITIONS]
    bars = ax.bar([COND_LABEL[c] for c in CONDITIONS], vals, color=["#2B5C8A", "#A9521A", "#8FB0CE"])
    ax.axhline(20, ls="--", color="#6B7280", lw=1)
    ax.text(2.45, 21.5, "випадкове вгадування, 20%", ha="right", fontsize=8, color="#6B7280")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Точність, %")
    ax.set_title(f"Вибір надкласу, {summary['n_items']} понять WordNet · {model_name}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    (OUT / "chart.png").write_bytes(buf.getvalue())
    return base64.b64encode(buf.getvalue()).decode()


def verdict(acc):
    gap = (acc["real"] - acc["renamed"]) * 100
    if gap >= 15:
        return f"Розрив {gap:.0f} п.п.: суттєва частина успіху тримається на знайомій назві, а не на змісті."
    if gap >= 5:
        return f"Розрив {gap:.0f} п.п.: назва допомагає помірно; основна частина успіху — зі змісту визначення."
    return f"Розрив {gap:.0f} п.п.: модель практично однаково справляється без знайомої назви — міркує зі змісту."


def write_report(summary, rows, items, model_name, mock):
    img = chart_png(summary, model_name)
    acc = summary["accuracy"]
    esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;")
    by_id = {it["id"]: it for it in items}
    real = {r["id"]: r for r in rows if r["condition"] == "real"}
    ren = {r["id"]: r for r in rows if r["condition"] == "renamed"}
    ex_ids = [i for i in real if real[i]["correct"] and not ren[i]["correct"]][:8]
    ex_rows = "".join(
        f"<tr><td>{esc(by_id[i]['term'])}</td><td>{esc(by_id[i]['pseudo'])}</td>"
        f"<td>{esc(by_id[i]['parent'])}</td>"
        f"<td>{esc(by_id[i]['options'][LETTERS.index(ren[i]['pred'])] if ren[i]['pred'] else '—')}</td>"
        f"<td class='def'>{esc(by_id[i]['definition'])}</td></tr>" for i in ex_ids)
    flip_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary["flips"].items())
    banner = ("<p class='mock'>MOCK-РЕЖИМ: числа згенеровані імітацією для перевірки конвеєра, "
              "це не результат експерименту.</p>") if mock else ""
    html = f"""<!DOCTYPE html><html lang="uk"><head><meta charset="utf-8">
<title>LLM: розуміння чи пам'ять — звіт</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#1B2335;line-height:1.5}}
h1{{font-size:26px}} h2{{font-size:19px;margin-top:32px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#13213A;color:#F3EFE6}} .def{{color:#4A5568;font-size:13px}}
.big{{display:flex;gap:16px}} .big div{{flex:1;background:#F6F4EE;border-radius:10px;padding:14px}}
.big b{{font-size:30px;display:block}} .mock{{background:#FBE3CF;padding:10px;border-radius:8px}}
.verdict{{background:#DCE6F0;padding:12px;border-radius:8px;font-weight:600}}
</style></head><body>
<h1>LLM будує таксономію: розуміння чи пам'ять?</h1>
{banner}
<p>Модель: <b>{esc(model_name)}</b> · понять: <b>{summary['n_items']}</b> · задача: обрати безпосередній надклас з 5 варіантів (випадкове вгадування — 20%).</p>
<div class="big">
<div>{COND_LABEL['real']}<b>{acc['real']*100:.0f}%</b></div>
<div>{COND_LABEL['renamed']}<b>{acc['renamed']*100:.0f}%</b></div>
<div>{COND_LABEL['name_only']}<b>{acc['name_only']*100:.0f}%</b></div>
</div>
<p class="verdict">{verdict(acc)}</p>
<img src="data:image/png;base64,{img}" style="width:100%" alt="Точність за умовами">
<h2>Хто кого «перемикає»</h2>
<table><tr><th>Результат на одному понятті (справжня назва vs вигадане слово)</th><th>Кількість</th></tr>{flip_rows}</table>
<p>Відбір: у жодному визначенні немає назви правильного надкласу; неправильні варіанти — переважно «сусіди» надкласу в ієрархії WordNet.</p>
<h2>Приклади: правильно зі справжньою назвою, помилка з вигаданою</h2>
<table><tr><th>Поняття</th><th>Вигадане слово</th><th>Правильний надклас</th><th>Відповідь моделі</th><th>Визначення</th></tr>{ex_rows or '<tr><td colspan=5>немає</td></tr>'}</table>
</body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="кількість понять")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true", help="без API, лише перевірка конвеєра")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    ds_path = OUT / f"dataset_n{args.n}_seed{args.seed}.json"
    if ds_path.exists():
        items = json.loads(ds_path.read_text(encoding="utf-8"))
    else:
        items = build_dataset(args.n, args.seed)
        ds_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Понять: {len(items)}; запитів: {len(items) * len(CONDITIONS)}")

    model = MockModel() if args.mock else OpenAICompatModel(args.model, args.max_tokens)
    rows = run(items, model, args.workers)

    with open(OUT / "raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, items)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(summary, rows, items, model.name, args.mock)

    for c in CONDITIONS:
        print(f"{COND_LABEL[c]:32s} {summary['accuracy'][c]*100:5.1f}%   без відповіді: {summary['no_answer'][c]}")
    if getattr(model, "finish", None):
        print("finish_reason:", model.finish)
    if sum(summary["no_answer"].values()):
        print("УВАГА: є порожні відповіді — запусти ще раз (вони перезапитаються) або збільш --max-tokens")
    print(verdict(summary["accuracy"]))
    print(f"Звіт: {OUT / 'report.html'}")


if __name__ == "__main__":
    main()