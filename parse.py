"""Turn screenplay .txt files into timestamped beats. Pure Python, no AI."""
import re
from pathlib import Path

BLOCK_RE = re.compile(r"^\*\*(?P<head>[^*]+?):\*\*\s*$")
TIME_RE = re.compile(r"Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)")
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "SOUND", "MUSIC", "TEXT", "SUPER", "TITLE", "VFX", "TRANSITION"}


def fmt(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60)
    return f"{int(m):02d}:{s:04.1f}"


def episode_number(path: Path, text: str) -> int:
    m = re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"episode-(\d+)", path.name)
    return int(m.group(1)) if m else 0


def parse_episode(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    ep = episode_number(path, text)
    logline = (re.search(r"EPISODE LOGLINE:\s*(.*)", text) or [None, ""])[1]
    lines = text.splitlines()
    beats, scene = [], ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Scene heading:"):
            scene = line.split(":", 1)[1].strip()
        m = BLOCK_RE.match(line)
        if m and i + 1 < len(lines):
            tm = TIME_RE.search(lines[i + 1])
            if tm:
                head = m.group("head").strip()
                start = int(tm.group(1)) * 60 + float(tm.group(2))
                end = int(tm.group(3)) * 60 + float(tm.group(4))
                body, j = [], i + 2
                while j < len(lines) and lines[j].strip() and not BLOCK_RE.match(lines[j].strip()):
                    body.append(lines[j].strip())
                    j += 1
                name = re.sub(r"\s*\(.*\)\s*$", "", head).strip()
                emotion = (re.search(r"\((.*)\)", head) or [None, ""])[1]
                btype = name if name in NON_DIALOGUE else "DIALOGUE"
                beats.append({
                    "id": f"E{ep:02d}-B{len(beats):03d}", "episode": ep, "scene": scene,
                    "start": round(start, 2), "end": round(end, 2), "type": btype,
                    "character": "" if btype != "DIALOGUE" else name, "emotion": emotion,
                    "text": " ".join(body).strip('"').strip(),
                })
                i = j
                continue
        i += 1
    duration = max((b["end"] for b in beats), default=0)
    return {"episode": ep, "file": path.name, "logline": logline, "duration": duration, "beats": beats}


def load_show(show_dir: str | Path) -> dict[int, dict]:
    eps = {}
    for p in sorted(Path(show_dir).glob("*.txt")):
        e = parse_episode(p)
        if e["beats"]:
            eps[e["episode"]] = e
    return eps


def beat_line(b: dict) -> str:
    """Compact text used in prompts (CAMERA beats excluded by callers)."""
    who = f"{b['character']} ({b['emotion']})" if b["type"] == "DIALOGUE" else b["type"]
    q = f'"{b["text"]}"' if b["type"] == "DIALOGUE" else b["text"]
    return f"[{b['id']}] {fmt(b['start'])}-{fmt(b['end'])} {who}: {q}"


if __name__ == "__main__":
    import sys
    eps = load_show(sys.argv[1])
    print(len(eps), "episodes;", sum(len(e["beats"]) for e in eps.values()), "beats")
