"""Extractive compression, exact raw-result retrieval, and schema shortlisting."""
import hashlib
import json
import re
from pathlib import Path

PROTECTED = re.compile(r"error|fail|exception|traceback|warning|exit.?code|summary|permission|denied|not found|passed", re.I)
OPTIONS = ["keep: preserve this chunk", "drop: omit this chunk", "defer: return the full original result"]


class RawStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def put(self, text):
        key = hashlib.sha256(text.encode()).hexdigest()
        path = self.directory / f"{key}.txt"
        if not path.exists():
            # Content-addressed local cache. These files can contain source/log data.
            with path.open("x", encoding="utf-8") as f:
                path.chmod(0o600)
                f.write(text)
        return key

    def get(self, key, start=0, end=None):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid raw reference")
        if start < 0 or (end is not None and end < start):
            raise ValueError("Invalid character range")
        return (self.directory / f"{key}.txt").read_text(encoding="utf-8")[start:end]


def chunks(text, max_chars=1200):
    """Contiguous original spans; never split a source line or discard its newline."""
    parts, start, current = [], 0, ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > max_chars:
            parts.append((start, start+len(current), current))
            start += len(current); current = ""
        current += line
    if current:
        parts.append((start, start+len(current), current))
    return parts


def compress(text, objective, predictor, store, max_chars=6000, experimental=False):
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    key = store.put(text)
    def original(reason):
        return {"text": text, "raw_ref": key, "compressed": False, "reason": reason,
                "original_chars": len(text), "returned_chars": len(text)}
    if not experimental:
        return original("shadow_mode")
    if len(text) <= max_chars:
        return original("within_budget")
    selected = []
    for start, end, chunk in chunks(text):
        if PROTECTED.search(chunk):
            selected.append((start, end, chunk)); continue
        try:
            decision = predictor.predict(objective, chunk, OPTIONS)
        except (ValueError, RuntimeError):
            return original("inference_failed_or_input_too_long")
        # These are deliberately conservative experimental heuristics, not calibrated gates.
        choice = decision["choice"]
        if choice["index"] == 2 or max(choice["probabilities"]) < .8:
            return original("defer_or_uncertain")
        if choice["index"] == 0 or decision["noul"] >= .1 or decision["score"]["value"] >= 1.5:
            selected.append((start, end, chunk))
    if not selected:
        return original("no_evidence_selected")
    output = f"Extracted tool-output spans; omitted text available via outputs_expand. raw_ref={key}\n"
    for start, end, chunk in selected:
        output += f"\n[characters {start}:{end}]\n{chunk}"
    # Essential evidence wins over requested budget. Avoid growing the result.
    if len(output) >= len(text):
        return original("no_savings")
    return {"text": output, "raw_ref": key, "compressed": True,
            "original_chars": len(text), "returned_chars": len(output),
            "budget_exceeded": len(output) > max_chars}


def shortlist(tools, query, predictor, top_k=3, experimental=False):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not experimental or len(tools) <= top_k:
        return tools
    options = [json.dumps(tool, ensure_ascii=False, sort_keys=True) for tool in tools]
    try:
        result = predictor.predict("Choose the next tool for the request.", query, options)
    except (ValueError, RuntimeError):
        return tools
    probabilities = result["choice"]["probabilities"]
    if max(probabilities) < .8:
        return tools
    selected = sorted(range(len(tools)), key=lambda i: probabilities[i], reverse=True)[:top_k]
    # Return complete original schemas, never shortened required fields or enums.
    return [tools[i] for i in selected]
