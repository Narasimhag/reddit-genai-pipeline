# eval_adaptive.py
"""
Adaptive evaluator for reddit-genai-pipeline.

- Introspects Search and Generate classes to call available methods.
- Adds per-query timeout, small-context truncation, and logging.
- Outputs CSV with question, answer, precision, latency, and errors.
"""

import os
import time
import logging
import inspect
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Import your repo classes (adjust if your module path differs)
from search import Search
from generate import Generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def find_callable(obj, candidates):
    """Return first callable attribute on obj that matches one of candidates names."""
    for name in candidates:
        attr = getattr(obj, name, None)
        if callable(attr):
            logging.info(f"Using {obj.__class__.__name__}.{name}()")
            return attr, name
    return None, None


def safe_invoke(func, args=(), kwargs=None, timeout=15):
    """Run func with timeout using a ThreadPoolExecutor, return (result, error)."""
    kwargs = kwargs or {}
    with ThreadPoolExecutor(max_workers=1) as execu:
        fut = execu.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout), None
        except TimeoutError:
            fut.cancel()
            return None, f"timeout after {timeout}s"
        except Exception as e:
            return None, f"error: {e}"


def normalize_docs(raw_docs):
    """
    Try to convert docs to a list of text snippets.
    Accepts: list of dicts, list of objects, list of strings.
    """
    if raw_docs is None:
        return []

    # If it's a single dict (not in a list), wrap
    if isinstance(raw_docs, dict):
        raw_docs = [raw_docs]

    if not isinstance(raw_docs, (list, tuple)):
        # If search returned a dataframe/other, try conversion
        try:
            return [str(raw_docs)]
        except Exception:
            return []

    texts = []
    for d in raw_docs:
        if isinstance(d, str):
            texts.append(d)
        elif isinstance(d, dict):
            # common keys: 'text','content','selftext','body','doc'
            for k in ("text", "content", "selftext", "body", "doc"):
                if k in d and isinstance(d[k], str):
                    texts.append(d[k])
                    break
            else:
                # fallback: join all string values
                vals = [v for v in d.values() if isinstance(v, str)]
                texts.append(" ".join(vals[:3]) if vals else str(d))
        else:
            # fallback to string
            try:
                texts.append(str(d))
            except Exception:
                continue
    return texts


def build_prompt(question, docs, max_doc_chars=800, max_docs=3):
    """
    Create a compact context prompt from docs.
    Truncate documents and use only top-k docs.
    """
    docs = docs[:max_docs]
    truncated = []
    for d in docs:
        txt = d.replace("\n", " ").strip()
        if len(txt) > max_doc_chars:
            txt = txt[:max_doc_chars] + "..."
        truncated.append(txt)
    context = "\n\n".join(truncated)
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer (use ONLY the context to answer; if context is not relevant say 'No relevant info found'):"
    return prompt


def main(eval_csv="data/eval_set.csv", out_csv="data/eval/eval_results_adaptive.csv",
         max_queries=None, timeout_per_query=20, parallel=False):
    os.makedirs("data/eval", exist_ok=True)
    df = pd.read_csv(eval_csv)
    if max_queries:
        df = df.head(max_queries)

    # instantiate classes
    searcher = Search()
    generator = Generate()

    # discover probable method names
    search_candidates = ["search", "query", "retrieve", "get", "run", "search_query"]
    gen_candidates = ["summarize", "generate", "answer", "call", "run", "predict"]

    search_func, search_name = find_callable(searcher, search_candidates)
    gen_func, gen_name = find_callable(generator, gen_candidates)

    if not search_func:
        logging.error("No searchable method found on Search. Please inspect Search class.")
    if not gen_func:
        logging.error("No generate method found on Generate. Please inspect Generate class.")

    results = []

    def process_row(idx, row):
        question = row["question"]
        expected = str(row.get("expected_keywords", "")).lower()
        expected_keywords = [k.strip() for k in expected.split(";") if k.strip()]

        t0 = time.time()
        # 1) run search - try different call signatures
        raw_docs = None
        search_err = None

        if search_func:
            sig = inspect.signature(search_func)
            try:
                if "query" in sig.parameters:
                    raw_docs, search_err = safe_invoke(search_func, args=(question,), timeout=timeout_per_query)
                else:
                    # try top_k if present
                    if "top_k" in sig.parameters:
                        raw_docs, search_err = safe_invoke(search_func, args=(question,), kwargs={"top_k": 5}, timeout=timeout_per_query)
                    elif "k" in sig.parameters:
                        raw_docs, search_err = safe_invoke(search_func, args=(question, 5), timeout=timeout_per_query)
                    else:
                        # fallback: single arg
                        raw_docs, search_err = safe_invoke(search_func, args=(question,), timeout=timeout_per_query)
            except Exception as e:
                raw_docs, search_err = None, f"search-call-exception:{e}"
        else:
            raw_docs, search_err = None, "no-search-method"

        docs = normalize_docs(raw_docs) if raw_docs else []

        # 2) prepare prompt/context and call generator
        prompt = build_prompt(question, docs)
        gen_err = None
        answer = None

        if gen_func:
            sigg = inspect.signature(gen_func)
            # try calling patterns in order of likely usefulness
            attempts = []
            # pattern 1: gen_func(question, docs)
            attempts.append(("q_docs", (question, docs), {}))
            # pattern 2: gen_func(question, context_string)
            attempts.append(("q_ctx", (question, prompt), {}))
            # pattern 3: gen_func(prompt)   (single prompt)
            attempts.append(("prompt_only", (prompt,), {}))
            # pattern 4: gen_func(question=..., docs=...)
            attempts.append(("kw_q_docs", (), {"question": question, "docs": docs}))
            attempts.append(("kw_prompt", (), {"prompt": prompt, "question": question}))

            for name, a, kw in attempts:
                try:
                    ans, gen_err = safe_invoke(gen_func, args=a, kwargs=kw, timeout=timeout_per_query)
                    if ans is not None:
                        answer = ans
                        break
                except Exception as e:
                    gen_err = f"attempt-{name}-error:{e}"
                    continue
            if answer is None and gen_err is None:
                gen_err = "no-answer-returned"
        else:
            gen_err = "no-gen-method"

        elapsed = time.time() - t0

        # Normalize answer to string
        if isinstance(answer, dict) and "text" in answer:
            answer_text = answer["text"]
        else:
            answer_text = str(answer) if answer is not None else ""

        # simple keyword precision
        low = answer_text.lower()
        hits = sum(1 for kw in expected_keywords if kw and kw in low)
        precision = hits / len(expected_keywords) if expected_keywords else None

        return {
            "question": question,
            "expected_keywords": expected_keywords,
            "answer": answer_text,
            "precision": precision,
            "search_error": search_err,
            "generate_error": gen_err,
            "latency_sec": round(elapsed, 2)
        }

    # run rows (parallel optional)
    if parallel:
        max_workers = min(4, (os.cpu_count() or 2))
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(process_row, idx, row): idx for idx, row in df.iterrows()}
            for fut in futures:
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    logging.error(f"Parallel row error: {e}")
    else:
        for idx, row in df.iterrows():
            logging.info(f"Evaluating row {idx+1}/{len(df)}...")
            res = process_row(idx, row)
            results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    logging.info(f"Saved eval results to {out_csv}")


if __name__ == "__main__":
    # Example quick run for debugging
    main(eval_csv="data/eval/eval_set.csv", max_queries=10, timeout_per_query=20, parallel=False)