#!/usr/bin/env python3
"""Compact LongBench runner for our vLLM endpoint with enable_thinking=False.

Runs a configurable subset of LongBench tasks via the chat API.
Scores with F1 (English QA), ROUGE-L (summarization), exact-match (classification).
"""
import argparse, json, os, re, string, sys, time
from collections import Counter
import urllib.request
import urllib.error

# Task configuration: HF dataset subset name, prompt template, metric
TASKS = {
    "narrativeqa":       {"metric": "f1",    "prompt": "You are given a story and a question. Answer the question as concisely as possible, using a few words only.\n\nStory:\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "qasper":            {"metric": "f1",    "prompt": "You are given a scientific article and a question. Answer the question as concisely as possible, using a few words only. If the question is unanswerable, reply 'unanswerable'.\n\nArticle:\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "multifieldqa_en":   {"metric": "f1",    "prompt": "Read the following article and answer the question as concisely as possible. Use a few words.\n\nArticle:\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "hotpotqa":          {"metric": "f1",    "prompt": "Answer the question based on the passages. Only give me the answer, no other words.\n\nPassages:\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "2wikimqa":          {"metric": "f1",    "prompt": "Answer the question based on the passages. Only give me the answer, no other words.\n\nPassages:\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "musique":           {"metric": "f1",    "prompt": "Answer the question based on the passages. Only give me the answer, no other words.\n\nPassages:\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "gov_report":        {"metric": "rouge", "prompt": "Summarize the following government report.\n\n{context}\n\nSummary:"},
    "qmsum":             {"metric": "rouge", "prompt": "Summarize the meeting transcript answering the query.\n\nQuery: {input}\n\nTranscript:\n{context}\n\nSummary:"},
    "multi_news":        {"metric": "rouge", "prompt": "Summarize the following news articles.\n\n{context}\n\nSummary:"},
    "trec":              {"metric": "em",    "prompt": "Classify the question type. Reply with only the class label.\n\n{context}\n\nQuestion: {input}\n\nClass:"},
    "triviaqa":          {"metric": "f1",    "prompt": "Answer the question. Only give me the answer.\n\n{context}\n\nQuestion: {input}\n\nAnswer:"},
    "samsum":            {"metric": "rouge", "prompt": "Summarize the dialogue.\n\n{context}\n\nSummary:"},
    "passage_retrieval_en": {"metric": "em", "prompt": "{context}\n\nThe question is: {input}. Reply only with the paragraph number (e.g. 'Paragraph 3')."},
    "passage_count":     {"metric": "em",    "prompt": "{context}\n\n{input}\nReply only with the count as a number."},
    "lcc":               {"metric": "code",  "prompt": "Complete the code. Reply only with the next line.\n\n{context}"},
    "repobench-p":       {"metric": "code",  "prompt": "Complete the code. Reply only with the next line.\n\n{context}"},
}

API_KEY = "genesis-local"

def normalize_answer(s):
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t):
        return " ".join(t.split())
    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))
    def lower(t):
        return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def qa_f1_score(pred, ground):
    norm_pred = normalize_answer(pred)
    norm_gt = normalize_answer(ground)
    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if norm_pred == norm_gt else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_l_score(pred, ground):
    try:
        from rouge import Rouge
        rouge = Rouge()
        if not pred.strip() or not ground.strip():
            return 0.0
        scores = rouge.get_scores(pred, ground, avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0

def em_score(pred, ground):
    return 1.0 if normalize_answer(pred) == normalize_answer(ground) else 0.0

def score_example(pred, answers, metric):
    if not isinstance(answers, list):
        answers = [answers]
    if metric == "f1":
        return max(qa_f1_score(pred, a) for a in answers)
    if metric == "em":
        return max(em_score(pred, a) for a in answers)
    if metric == "rouge":
        return max(rouge_l_score(pred, a) for a in answers)
    if metric == "code":
        # simple prefix match — accept if prediction starts with ground
        return max(1.0 if normalize_answer(pred).startswith(normalize_answer(a)[:50]) else 0.0 for a in answers)
    return 0.0

def call_model(host, model, prompt, max_tokens=128, enable_thinking=False, timeout=600):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            js = json.loads(r.read())
        dt = time.perf_counter() - t0
        msg = js["choices"][0]["message"]
        content = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        usage = js.get("usage", {})
        return {"content": content, "reasoning": rc, "dt": dt,
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "err": None}
    except Exception as e:
        return {"content": "", "reasoning": "", "dt": time.perf_counter()-t0, "err": str(e)[:200]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--model", default="qwen3.6-35b-a3b")
    p.add_argument("--tasks", default="narrativeqa,qasper,hotpotqa,gov_report,trec,lcc")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--label", default="longbench_run")
    p.add_argument("--out", default="/tmp/longbench_results.jsonl")
    p.add_argument("--max-context-chars", type=int, default=240000, help="truncate context to avoid >128k ctx")
    args = p.parse_args()

    data_dir = "/tmp/longbench_data/extracted/data"

    outfh = open(args.out, "w")
    task_results = {}
    print(f"# {args.label}: tasks={args.tasks} limit={args.limit} max_tokens={args.max_tokens}")
    for task in args.tasks.split(","):
        if task not in TASKS:
            print(f"  [SKIP] unknown task: {task}")
            continue
        cfg = TASKS[task]
        path = os.path.join(data_dir, f"{task}.jsonl")
        if not os.path.exists(path):
            print(f"  [ERR] missing data file: {path}")
            continue
        ds = [json.loads(l) for l in open(path)]
        scores = []
        total_dt = 0
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break
            ctx = ex.get("context", "")[: args.max_context_chars]
            q = ex.get("input", "")
            prompt = cfg["prompt"].format(context=ctx, input=q)
            ans = ex.get("answers", [""])
            res = call_model(args.host, args.model, prompt, args.max_tokens)
            total_dt += res["dt"]
            pred = res["content"].strip()
            s = score_example(pred, ans, cfg["metric"])
            scores.append(s)
            outfh.write(json.dumps({"task": task, "i": i, "metric": cfg["metric"],
                                    "score": s, "pred": pred[:200], "ans": str(ans)[:200],
                                    "prompt_tokens": res.get("prompt_tokens"),
                                    "completion_tokens": res.get("completion_tokens"),
                                    "err": res.get("err"), "dt": res["dt"]}) + "\n")
            outfh.flush()
        avg = sum(scores) / max(1, len(scores))
        task_results[task] = {"metric": cfg["metric"], "n": len(scores), "score": avg, "dt_total": total_dt}
        print(f"  {task} [{cfg['metric']}]: {avg:.3f} (n={len(scores)}, {total_dt:.1f}s)")

    print()
    print(f"=== {args.label} SUMMARY ===")
    for t, r in task_results.items():
        print(f"  {t}: score={r['score']:.3f} [{r['metric']}] n={r['n']} time={r['dt_total']:.1f}s")
    agg = sum(r["score"] for r in task_results.values()) / max(1, len(task_results))
    print(f"  AVG: {agg:.3f} across {len(task_results)} tasks")
    outfh.close()

if __name__ == "__main__":
    main()
