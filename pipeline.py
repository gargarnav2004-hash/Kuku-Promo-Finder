"""Promo-finding pipeline: parse -> AI tag -> build candidates -> AI judge -> validate -> write outputs."""
import csv
import re
import json
import time
from pathlib import Path

from llm import LLM, load_config
from parse import beat_line, fmt, load_show

ROOT = Path(__file__).parent

# ---------------------------------------------------------------- prompts
TAG_PROMPT = """You are a promo editor for vertical microdrama shows (like ReelShort / DramaBox ads).
Below is ONE episode screenplay as timestamped beats (dialogue and actions). Camera directions are omitted.
You only have the TEXT. Do not judge acting, visuals or sound.

Show: {show}
Episode {ep} logline: {logline}

BEATS:
{beats}

Find:
1. HOOKS: beats that would stop a scrolling stranger in the first 3 seconds of an ad
   (a slap or violence, a shocking reveal, a humiliating or outrageous line, a threat, a betrayal).
   The hook beat itself must grab attention instantly - not a slow build-up.
2. CLIFFHANGERS: beats where a clip could END on an unresolved moment so the viewer needs to know what happens next
   (a question left hanging, a threat about to land, a reveal cut before the reaction, a door opening...).
For each, flag needs_backstory=true if a stranger would NOT understand it without earlier episodes,
and spoiler=true if it gives away a major twist of the series (e.g. a secret identity being revealed).

Return 3-6 hooks and 3-6 cliffhangers, only real beat_ids from the list above. Strength 1-5 (5 = best).
Return ONLY JSON in this exact schema:
{{"hooks": [{{"beat_id": "E01-B012", "kind": "slap|reveal|shocking_line|threat|betrayal|other", "strength": 4,
   "needs_backstory": false, "spoiler": false, "reason": "short reason"}}],
 "cliffhangers": [{{"beat_id": "E01-B040", "strength": 5, "needs_backstory": false, "spoiler": false, "reason": "short reason"}}]}}
"""

JUDGE_PROMPT = """You are testing short promo clips for a vertical drama ad on TikTok/Instagram.
Pretend you have NEVER seen this show. You see ONLY each clip's script lines below (times are seconds from clip start;
"--- CUT ---" marks a jump to a different moment). Judge each clip completely independently of the others.
You only have the text: do not claim anything about acting, visuals or sound.

{clips}

Be a HARSH, calibrated critic: 3 = average, 5 = exceptional and rare. Most clips should NOT get 5s.
Penalise: unexplained names or relationships, starting mid-conversation, endings that resolve the tension,
cuts between moments that do not connect logically.

Score each clip 1-5 on:
- hook: does something in the first 3 seconds make a scroller stop?
- cliffhanger: does it END on an unresolved moment that makes you need the next part?
- standalone_clarity: does it make sense to a stranger with no backstory?
- timestamp_quality: does it start and end cleanly (not mid-thought, no dead air at start, no confusing cut)?
- promo_appeal: overall, would this pull new viewers in?
Also write: hook (what grabs attention, quote the line or action), hook_quote (exact words copied from the clip lines),
cliffhanger (what is left unresolved), cliffhanger_quote (exact words copied from the clip lines),
why_it_works (one line), risks (one line: what could make it fail, e.g. confusing names, needs context, spoiler, weak start).
And two strict yes/no checks:
- hook_works_cold: true only if the opening hook makes sense to someone with ZERO context about the show.
- ends_unresolved: true only if the clip's LAST moment is unresolved - an open question, a threat about to land,
  a reveal cut before the reaction, or an interrupted action. A vow, a resolution, a hug, a calm reaction or a
  feeling is NOT unresolved. Give unresolved_type: question|threat|reveal|interrupted_action|none.
Return ONLY JSON: {{"results": [{{"id": "...", "hook": 1, "cliffhanger": 1, "standalone_clarity": 1, "timestamp_quality": 1,
"promo_appeal": 1, "hook_text": "...", "hook_quote": "...", "cliffhanger_text": "...", "cliffhanger_quote": "...",
"why_it_works": "...", "risks": "...", "hook_works_cold": true, "ends_unresolved": true, "unresolved_type": "question"}}]}}
with one result per clip id: {ids}
"""

NEXT_PROMPT = """Each item below shows how a promo clip ENDS, followed by the very next lines in the screenplay that come
right AFTER the cut (the viewer does not see these). Decide for each: does the NEXT line immediately RESOLVE the
unresolved moment the clip ends on? (answers the question, carries out or defuses the threat, explains the reveal,
completes the interrupted action). If the next line resolves it, the cut is weak because the tension is gone.
If the next lines continue or escalate the tension, it is NOT resolved.

{items}

Return ONLY JSON: {{"results": [{{"id": "...", "next_line_resolves": false, "reason": "one short line"}}]}} with one result per id: {ids}
"""

CRITERIA = ["hook", "cliffhanger", "standalone_clarity", "timestamp_quality", "promo_appeal"]


# ---------------------------------------------------------------- helpers
def content_beats(ep: dict) -> list[dict]:
    return [b for b in ep["beats"] if b["type"] != "CAMERA"]


def snap_start(ep, t):
    """Never start in the middle of a line: move back to the start of any dialogue beat that spans t."""
    for b in ep["beats"]:
        if b["type"] == "DIALOGUE" and b["start"] < t < b["end"]:
            return b["start"]
    return t


def snap_end(ep, t):
    for b in ep["beats"]:
        if b["type"] == "DIALOGUE" and b["start"] < t < b["end"]:
            return b["end"]
    return t


def seg_beats(ep, s, e):
    return [b for b in content_beats(ep) if b["start"] < e - 0.05 and b["end"] > s + 0.05]


def norm(t: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in t).split())


def overlap_ratio(a: dict, b: dict) -> float:
    """Share of seconds two promos have in common, relative to the shorter promo."""
    common = 0.0
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["episode"] == sb["episode"]:
                common += max(0.0, min(sa["end"], sb["end"]) - max(sa["start"], sb["start"]))
    return common / max(1e-6, min(a["duration"], b["duration"]))


def make_promo(pid, kind, segs, hook, cliff, episodes, prescore):
    segments = []
    for (ep, s, e) in segs:
        lines = seg_beats(episodes[ep], s, e)
        segments.append({"episode": ep, "start": round(s, 2), "end": round(e, 2),
                         "start_ts": fmt(s), "end_ts": fmt(e), "beat_ids": [b["id"] for b in lines]})
    return {"id": pid, "kind": kind, "segments": segments,
            "duration": round(sum(x["end"] - x["start"] for x in segments), 1),
            "episodes": sorted({x["episode"] for x in segments}),
            "hook_beat": hook["beat_id"], "cliff_beat": cliff["beat_id"],
            "hook_tag": hook, "cliff_tag": cliff, "prescore": prescore}


# ---------------------------------------------------------------- step 2: AI tagging
def tag_episode(llm: LLM, show: str, ep: dict) -> dict:
    ids = {b["id"] for b in content_beats(ep)}
    prompt = TAG_PROMPT.format(show=show, ep=ep["episode"], logline=ep["logline"],
                               beats="\n".join(beat_line(b) for b in content_beats(ep)))

    def validate(o):
        if not isinstance(o, dict) or not isinstance(o.get("hooks"), list) or not isinstance(o.get("cliffhangers"), list):
            return "must have lists 'hooks' and 'cliffhangers'"
        for x in o["hooks"] + o["cliffhangers"]:
            if x.get("beat_id") not in ids:
                return f"unknown beat_id {x.get('beat_id')}"
            if not isinstance(x.get("strength"), int) or not 1 <= x["strength"] <= 5:
                return "strength must be an integer 1-5"
        if not o["hooks"] or not o["cliffhangers"]:
            return "need at least one hook and one cliffhanger"
        return None

    return llm.json_call(prompt, validate, tag=f"tag E{ep['episode']:02d}")


# ---------------------------------------------------------------- step 3: candidates (pure Python)
def build_candidates(episodes, tags, pc) -> list[dict]:
    lo, hi = pc["min_seconds"], pc["max_seconds"]
    target = pc.get("target_seconds", (lo + hi) / 2)
    beat_by_id = {b["id"]: b for e in episodes.values() for b in e["beats"]}
    out = []
    for epn, t in tags.items():
        ep = episodes[epn]
        cands = []
        for h in t["hooks"]:
            hb = beat_by_id[h["beat_id"]]
            s = snap_start(ep, hb["start"])
            for c in t["cliffhangers"]:
                cb = beat_by_id[c["beat_id"]]
                e = snap_end(ep, cb["end"])
                if cb["start"] <= hb["start"]:
                    continue
                base = 2 * h["strength"] + 2 * c["strength"] - 2 * (h.get("needs_backstory", False) + c.get("needs_backstory", False)) \
                    - (h.get("spoiler", False) + c.get("spoiler", False))
                if lo <= e - s <= hi:
                    score = base + (1 if abs((e - s) - target) <= 2 else 0)
                    cands.append(make_promo(f"E{epn:02d}@{fmt(s)}-{fmt(e)}", "single", [(epn, s, e)], h, c, episodes, score))
                elif pc.get("tighten_long_moments") and hi < e - s <= 90:
                    # keep the opening (hook) and the ending (cliffhanger) of the stretch, drop the middle
                    beats = content_beats(ep)
                    ends1 = [b["end"] for b in beats if s + 8 <= b["end"] <= s + 0.6 * target]
                    starts2 = [b["start"] for b in beats if e - 0.65 * target <= b["start"] <= e - 8]
                    if not ends1 or not starts2:
                        continue
                    e1 = snap_end(ep, min(ends1, key=lambda x: abs(x - (s + 0.45 * target))))
                    s2 = snap_start(ep, min(starts2, key=lambda x: abs(x - (e - 0.55 * target))))
                    if s2 <= e1 + 2 or not lo <= (e1 - s) + (e - s2) <= hi:
                        continue
                    cands.append(make_promo(f"T-E{epn:02d}@{fmt(s)}-{fmt(e1)}+{fmt(s2)}-{fmt(e)}", "tightened",
                                            [(epn, s, e1), (epn, s2, e)], h, c, episodes, base - 1))
        cands.sort(key=lambda x: -x["prescore"])
        kept = []
        for c in cands:
            if all(overlap_ratio(c, k) <= pc["near_duplicate_overlap"] for k in kept):
                kept.append(c)
            if len(kept) >= pc["candidates_per_episode"]:
                break
        out += kept

    # multi-segment: strong hook moment + strong cliffhanger moment from a LATER point (same or later episode)
    hooks = [(epn, h) for epn, t in tags.items() for h in t["hooks"] if h["strength"] >= 4 and not h.get("needs_backstory")]
    cliffs = [(epn, c) for epn, t in tags.items() for c in t["cliffhangers"] if c["strength"] >= 4 and not c.get("needs_backstory")]
    hooks.sort(key=lambda x: -x[1]["strength"])
    cliffs.sort(key=lambda x: -x[1]["strength"])
    pairs = []
    for he, h in hooks:
        hb = beat_by_id[h["beat_id"]]
        ep = episodes[he]
        s1 = snap_start(ep, hb["start"])
        ends = [b["end"] for b in content_beats(ep) if s1 + 8 <= b["end"] <= s1 + 20]
        if not ends:
            continue
        e1 = snap_end(ep, ends[0] if len(ends) == 1 else ends[len(ends) // 2])
        for ce, c in cliffs:
            cb = beat_by_id[c["beat_id"]]
            if (ce, cb["start"]) <= (he, e1):
                continue
            cep = episodes[ce]
            e2 = snap_end(cep, cb["end"])
            starts = [b["start"] for b in content_beats(cep) if e2 - 22 <= b["start"] <= e2 - 8]
            if not starts:
                continue
            s2 = snap_start(cep, starts[0])
            if ce == he and s2 < e1:  # same episode: second moment must come after the first
                continue
            total = (e1 - s1) + (e2 - s2)
            if not lo <= total <= hi:
                continue
            pairs.append((h["strength"] + c["strength"], (he, s1, e1), (ce, s2, e2), h, c))
    pairs.sort(key=lambda x: -x[0])
    used_h, used_c, multi = set(), set(), []
    for sc, seg1, seg2, h, c in pairs:
        if h["beat_id"] in used_h or c["beat_id"] in used_c:
            continue
        used_h.add(h["beat_id"]); used_c.add(c["beat_id"])
        pid = f"M-E{seg1[0]:02d}@{fmt(seg1[1])}+E{seg2[0]:02d}@{fmt(seg2[1])}"
        multi.append(make_promo(pid, "multi", [seg1, seg2], h, c, episodes, 2 * sc))
        if len(multi) >= pc["multi_segment_candidates"]:
            break
    return out + multi


# ---------------------------------------------------------------- step 4: AI judge (cold viewer)
def clip_text(p, episodes) -> str:
    beat_by_id = {b["id"]: b for e in episodes.values() for b in e["beats"]}
    rows, offset = [], 0.0
    for i, s in enumerate(p["segments"]):
        if i:
            rows.append("--- CUT ---")
        for bid in s["beat_ids"]:
            b = beat_by_id[bid]
            t = offset + max(0.0, b["start"] - s["start"])
            who = b["character"] if b["type"] == "DIALOGUE" else b["type"]
            txt = f'"{b["text"]}"' if b["type"] == "DIALOGUE" else b["text"]
            rows.append(f"{t:5.1f}s {who}: {txt}")
        offset += s["end"] - s["start"]
    return "\n".join(rows)


def judge(llm: LLM, batch, episodes) -> dict:
    clips = "\n\n".join(f"=== CLIP id={p['id']} (length {p['duration']:.0f}s) ===\n{clip_text(p, episodes)}" for p in batch)
    ids = [p["id"] for p in batch]
    prompt = JUDGE_PROMPT.format(clips=clips, ids=json.dumps(ids))

    def validate(o):
        res = o.get("results") if isinstance(o, dict) else None
        if not isinstance(res, list):
            return "missing 'results' list"
        got = {r.get("id") for r in res}
        if set(ids) - got:
            return f"missing results for ids {sorted(set(ids) - got)}"
        for r in res:
            for k in CRITERIA:
                if not isinstance(r.get(k), (int, float)) or not 1 <= r[k] <= 5:
                    return f"score '{k}' must be 1-5 for {r.get('id')}"
            for k in ("hook_works_cold", "ends_unresolved"):
                if not isinstance(r.get(k), bool):
                    return f"'{k}' must be true or false for {r.get('id')}"
        return None

    o = llm.json_call(prompt, validate, tag=f"judge {len(batch)} clips")
    return {r["id"]: r for r in o["results"]}


# ---------------------------------------------------------------- gate: hook test + cliffhanger test
def next_beats(p, episodes, n) -> list[dict]:
    """The script beats that come right after the promo's final cut (Python lookup, no AI)."""
    last = p["segments"][-1]
    ep = episodes[last["episode"]]
    nxt = [b for b in content_beats(ep) if b["start"] >= last["end"] - 0.05][:n]
    if len(nxt) < n and last["episode"] + 1 in episodes:  # promo ends at the end of an episode -> next episode's opening
        nxt += content_beats(episodes[last["episode"] + 1])[:n - len(nxt)]
    return nxt


def check_next_lines(llm: LLM, batch, episodes, n) -> dict:
    bb = {b["id"]: b for e in episodes.values() for b in e["beats"]}
    items, ids = [], []
    for p in batch:
        ending = [beat_line(bb[b]) for b in p["segments"][-1]["beat_ids"][-3:]]
        after = [beat_line(b) for b in p["next_beats"]]
        items.append(f"=== id={p['id']} ===\nCLIP ENDS WITH:\n" + "\n".join(ending) + "\nNEXT LINES AFTER THE CUT:\n" + "\n".join(after))
        ids.append(p["id"])

    def validate(o):
        res = o.get("results") if isinstance(o, dict) else None
        if not isinstance(res, list) or set(ids) - {r.get("id") for r in res}:
            return "need one result per id"
        if any(not isinstance(r.get("next_line_resolves"), bool) for r in res):
            return "next_line_resolves must be true or false"
        return None

    o = llm.json_call(NEXT_PROMPT.format(items="\n\n".join(items), ids=json.dumps(ids)), validate, tag=f"next-line check {len(batch)}")
    return {r["id"]: r for r in o["results"]}


def near_miss_gap(p, g):
    """How far a promo is from passing the gate (0 = passes). None = fails a hard timing check (never used to fill).
    1 point per missing score point, +1 for each failed yes/no check (hook needs context, ending resolved, next line resolves)."""
    gt, j = p.get("gate"), p.get("judge")
    if not gt or not j:
        return None
    hard = ("does not start within", "does not end within", "could not run", "not judged")
    if any(h in r for t in ("hook", "cliffhanger") for r in gt[t]["reasons"] for h in hard):
        return None
    nc = p.get("next_check") or {}
    return (max(0, g["hook_min_score"] - j["hook"]) + max(0, g["cliffhanger_min_score"] - j["cliffhanger"])
            + (0 if j.get("hook_works_cold") else 1) + (0 if j.get("ends_unresolved") else 1)
            + (1 if nc.get("next_line_resolves") else 0))


def gate_promo(p, episodes, pc, g) -> dict:
    """Return {'hook': {...}, 'cliffhanger': {...}} with pass flags and the reasons for any failure."""
    bb = {b["id"]: b for e in episodes.values() for b in e["beats"]}
    j, first, last = p.get("judge") or {}, p["segments"][0], p["segments"][-1]
    hook_fail, cliff_fail = [], []
    hb = bb.get(p["hook_beat"])
    if not hb or hb["episode"] != first["episode"] or not (first["start"] - 0.01 <= hb["start"] <= first["start"] + pc["hook_window_seconds"]):
        hook_fail.append(f"hook beat does not start within {pc['hook_window_seconds']}s of the promo start")
    if not j:
        hook_fail.append("not judged by AI"); cliff_fail.append("not judged by AI")
    else:
        if j["hook"] < g["hook_min_score"]:
            hook_fail.append(f"AI hook score {j['hook']} < {g['hook_min_score']}")
        if not j.get("hook_works_cold"):
            hook_fail.append("hook does not make sense with zero context (cold-viewer check)")
        if j["cliffhanger"] < g["cliffhanger_min_score"]:
            cliff_fail.append(f"AI cliffhanger score {j['cliffhanger']} < {g['cliffhanger_min_score']}")
        if not j.get("ends_unresolved"):
            cliff_fail.append(f"AI says the ending is not unresolved (type: {j.get('unresolved_type', 'none')})")
    cb = bb.get(p["cliff_beat"])
    if not cb or cb["episode"] != last["episode"] or last["end"] - cb["end"] > g["cliff_end_window_seconds"]:
        cliff_fail.append(f"promo does not end within {g['cliff_end_window_seconds']}s after the cliffhanger beat")
    nc = p.get("next_check")
    if nc is None:
        if not cliff_fail:
            cliff_fail.append("next-line check could not run")
    elif nc.get("next_line_resolves"):
        cliff_fail.append(f"next line in the script resolves it: {nc.get('reason', '')}")
    return {"hook": {"pass": not hook_fail, "reasons": hook_fail}, "cliffhanger": {"pass": not cliff_fail, "reasons": cliff_fail}}


# ---------------------------------------------------------------- step 5: validation (pure Python)
def validate_promo(p, episodes, pc) -> list[dict]:
    issues = []
    beat_by_id = {b["id"]: b for e in episodes.values() for b in e["beats"]}
    add = lambda lvl, code, msg: issues.append({"level": lvl, "code": code, "msg": msg})
    for s in p["segments"]:
        if s["episode"] not in episodes:
            add("error", "bad_episode", f"episode {s['episode']} does not exist"); continue
        ep = episodes[s["episode"]]
        if not (0 <= s["start"] < s["end"] <= ep["duration"] + 0.01):
            add("error", "bad_timestamps", f"segment E{s['episode']} {s['start_ts']}-{s['end_ts']} outside 0-{fmt(ep['duration'])} or start>=end")
        for b in ep["beats"]:
            if b["type"] == "DIALOGUE" and (b["start"] < s["start"] < b["end"] or b["start"] < s["end"] < b["end"]):
                add("error", "mid_line_cut", f"segment boundary cuts line {b['id']} mid-sentence")
        if not s["beat_ids"]:
            add("error", "no_evidence", "segment has no source lines")
    if not pc["min_seconds"] <= p["duration"] <= pc["max_seconds"]:
        add("error", "bad_duration", f"duration {p['duration']}s outside {pc['min_seconds']}-{pc['max_seconds']}s")
    hb = beat_by_id.get(p["hook_beat"])
    first = p["segments"][0]
    if not hb or hb["episode"] != first["episode"] or not (first["start"] - 0.01 <= hb["start"] <= first["start"] + pc["hook_window_seconds"]):
        add("error", "hook_late", f"hook beat {p['hook_beat']} does not start within {pc['hook_window_seconds']}s of the promo start")
    segs = p["segments"]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            if a["episode"] == b["episode"] and a["start"] < b["end"] and b["start"] < a["end"]:
                add("error", "segment_overlap", f"segments {i + 1} and {j + 1} overlap")
    j = p.get("judge")
    if not j:
        add("warning", "not_judged", "AI judge did not score this promo")
    else:
        src = norm(" ".join(beat_by_id[b]["text"] for s in segs for b in s["beat_ids"]))
        for k in ("hook_quote", "cliffhanger_quote"):
            raw = j.get(k, "")
            raw = re.sub(r"^[^\"]{0,40}:\s*", "", raw)  # drop a leading 'NAME:' speaker label
            q = norm(raw)
            words = q.split()
            found = bool(q) and (q in src or sum(wd in src.split() for wd in words) >= 0.8 * len(words))
            if not found:
                add("warning", "quote_not_in_source", f"AI {k} not found verbatim in source lines: '{j.get(k, '')[:60]}'")
        if j["standalone_clarity"] <= 2:
            add("warning", "needs_context", "AI judge says a stranger may not understand it")
    for tagname in ("hook_tag", "cliff_tag"):
        t = p[tagname]
        if t.get("needs_backstory"):
            add("warning", "needs_backstory", f"tagger flagged {t['beat_id']} as needing backstory")
        if t.get("spoiler"):
            add("warning", "spoiler", f"tagger flagged {t['beat_id']} as a possible major spoiler")
    return issues


# ---------------------------------------------------------------- orchestration
def run_pipeline(episode_numbers=None, out_dir: Path = ROOT / "outputs", show_dir=None, progress=None):
    cfg = load_config()
    pc, w = cfg["promo"], cfg["weights"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"
    log_path.write_text("")
    llm = LLM(cfg["llm"], log_path)
    say = lambda frac, msg: (llm.log(msg), progress and progress(frac, msg))
    t0 = time.time()
    show_dir = Path(show_dir or ROOT / cfg["show_dir"])
    show = show_dir.name
    episodes = load_show(show_dir)
    if episode_numbers:
        episodes = {k: v for k, v in episodes.items() if k in set(episode_numbers)}
    say(0.0, f"Parsed {len(episodes)} episodes, {sum(len(e['beats']) for e in episodes.values())} beats from '{show}'. Model: {cfg['llm']['model']}")

    tags, tag_failures = {}, []
    for i, (epn, ep) in enumerate(sorted(episodes.items())):
        say(0.05 + 0.6 * i / len(episodes), f"Tagging hooks & cliffhangers in episode {epn} ({i + 1}/{len(episodes)})")
        try:
            tags[epn] = tag_episode(llm, show, ep)
        except Exception as e:
            tag_failures.append({"episode": epn, "error": str(e)[:300]})
            llm.log(f"  TAGGING FAILED for episode {epn}: {e}")

    cands = build_candidates(episodes, tags, pc)
    cands.sort(key=lambda x: -x["prescore"])
    to_judge = cands[:pc["judge_top_n"]]
    say(0.65, f"Built {len(cands)} candidates; judging top {len(to_judge)}")
    bs = pc["judge_batch_size"]
    for i in range(0, len(to_judge), bs):
        batch = to_judge[i:i + bs]
        say(0.65 + 0.3 * i / max(1, len(to_judge)), f"AI judge: clips {i + 1}-{i + len(batch)} of {len(to_judge)}")
        try:
            res = judge(llm, batch, episodes)
            for p in batch:
                p["judge"] = res[p["id"]]
        except Exception as e:
            llm.log(f"  JUDGE FAILED for batch {i // bs}: {e}")

    for p in cands:
        j = p.get("judge")
        p["score"] = round(sum(w[k] * j[k] / 5 for k in CRITERIA), 1) if j else None
        p["validation"] = validate_promo(p, episodes, pc)
        p["status"] = "fail" if any(v["level"] == "error" for v in p["validation"]) else ("warn" if p["validation"] else "pass")
        bb = {b["id"]: b for e in episodes.values() for b in e["beats"]}
        p["source_lines"] = [beat_line(bb[b]) for s in p["segments"] for b in s["beat_ids"]]

    judged = sorted([p for p in cands if p["score"] is not None], key=lambda x: -x["score"])

    # ---- strict gate: hook test + cliffhanger test (thresholds in config.yaml -> gate)
    g = cfg["gate"]
    for p in cands:
        p["next_beats"] = next_beats(p, episodes, g["next_beats_to_check"])
        p["next_lines"] = [beat_line(b) for b in p["next_beats"]]
    # only check the next line where the rest of the cliffhanger test already passed (saves AI calls)
    need = [p for p in judged if p["judge"]["cliffhanger"] >= g["cliffhanger_min_score"] and p["judge"].get("ends_unresolved")
            and p["next_beats"]]
    for i in range(0, len(need), 10):
        batch = need[i:i + 10]
        say(0.95, f"Next-line check: clips {i + 1}-{i + len(batch)} of {len(need)}")
        try:
            res = check_next_lines(llm, batch, episodes, g["next_beats_to_check"])
            for p in batch:
                p["next_check"] = res[p["id"]]
        except Exception as e:
            llm.log(f"  NEXT-LINE CHECK FAILED: {e}")
    if g.get("fill_with_near_misses"):
        # second pass: check the rest too, so we know how close every near-miss really is
        rest = [p for p in judged if p not in need and p["next_beats"]]
        for i in range(0, len(rest), 10):
            batch = rest[i:i + 10]
            say(0.97, f"Next-line check (near-miss pool): clips {i + 1}-{i + len(batch)} of {len(rest)}")
            try:
                res = check_next_lines(llm, batch, episodes, g["next_beats_to_check"])
                for p in batch:
                    p["next_check"] = res[p["id"]]
            except Exception as e:
                llm.log(f"  NEXT-LINE CHECK FAILED: {e}")
    for p in judged:
        if not p["next_beats"]:
            p["next_check"] = {"next_line_resolves": False, "reason": "no later line in the provided episodes"}
    for p in cands:
        p.pop("next_beats", None)
        p["gate"] = gate_promo(p, episodes, pc, g)
        p["gate_pass"] = p["gate"]["hook"]["pass"] and p["gate"]["cliffhanger"]["pass"]
    rejected = [p for p in judged if not p["gate_pass"]]
    gate_passed = [p for p in judged if p["gate_pass"]]
    for p in cands:
        p["near_miss"], p["gap"] = False, near_miss_gap(p, g)

    ranked, dupes = [], []
    for p in gate_passed:
        d = next((k for k in ranked if overlap_ratio(p, k) > pc["near_duplicate_overlap"]), None)
        if d:
            dupes.append({"id": p["id"], "score": p["score"], "duplicate_of": d["id"]})
        else:
            ranked.append(p)
    # final list: best N that pass the rule checks, at most K per episode (spread across the show)
    per_ep, final, skipped = {}, [], []
    cap, n = pc.get("max_promos_per_episode", 99), pc.get("final_promo_count", len(ranked))
    for p in ranked:
        reason = None
        if p["status"] == "fail":
            reason = "fails rule checks"
        elif any(per_ep.get(e, 0) >= cap for e in p["episodes"]):
            reason = f"episode limit ({cap} per episode)"
        elif len(final) >= n:
            reason = f"below top {n}"
        p["selected"], p["not_selected_reason"] = reason is None, reason
        if reason:
            skipped.append(p)
            continue
        final.append(p)
        for e in p["episodes"]:
            per_ep[e] = per_ep.get(e, 0) + 1
    n_passed_final = len(final)

    # fill to N with the CLOSEST near-misses (smallest gap, then highest AI score). Timing checks are never relaxed.
    near_pool = sorted([p for p in rejected if p["gap"] is not None and p["status"] != "fail"],
                       key=lambda x: (x["gap"], -x["score"]))
    if g.get("fill_with_near_misses"):
        for p in near_pool:
            if len(final) >= n:
                break
            if any(per_ep.get(e, 0) >= cap for e in p["episodes"]):
                continue
            if any(overlap_ratio(p, k) > pc["near_duplicate_overlap"] for k in final):
                continue
            p["near_miss"], p["selected"], p["not_selected_reason"] = True, True, None
            final.append(p)
            for e in p["episodes"]:
                per_ep[e] = per_ep.get(e, 0) + 1
    write_rejected(out_dir, rejected)
    for i, p in enumerate(final, 1):
        p["rank"] = i
    for i, p in enumerate(skipped, len(final) + 1):
        p["rank"] = i
    (out_dir / "all_ranked_promos.json").write_text(json.dumps(final + skipped, indent=1))
    ranked_all, ranked = ranked, final

    summary = {
        "show": show, "episodes_analyzed": len(episodes), "episodes_tagged": len(tags), "tag_failures": tag_failures,
        "candidates_built": len(cands),
        "candidates_by_kind": {k: sum(c["kind"] == k for c in cands) for k in ("single", "tightened", "multi")},
        "candidates_judged": len(judged),
        "gate": {
            "thresholds": g,
            "passed_both": len(gate_passed), "rejected": len(rejected),
            "failed_hook_test": sum(not p["gate"]["hook"]["pass"] for p in judged),
            "failed_cliffhanger_test": sum(not p["gate"]["cliffhanger"]["pass"] for p in judged),
            "failed_both": sum(not p["gate"]["hook"]["pass"] and not p["gate"]["cliffhanger"]["pass"] for p in judged),
            "next_line_checked": sum(1 for p in judged if p.get("next_check") and p["next_check"].get("reason") != "no later line in the provided episodes"),
            "next_line_resolved": sum(bool((p.get("next_check") or {}).get("next_line_resolves")) for p in judged),
            "fail_reason_counts": _reason_counts(judged),
            "final_list_short": n_passed_final < pc.get("final_promo_count", 0),
            "in_final_passed_both": n_passed_final,
            "in_final_near_misses": sum(p["near_miss"] for p in final),
            "near_misses_used": [{"id": p["id"], "gap": p["gap"], "score": p["score"],
                                  "failed": "; ".join(p["gate"]["hook"]["reasons"] + p["gate"]["cliffhanger"]["reasons"])}
                                 for p in final if p["near_miss"]],
            "near_miss_pool_by_gap": {str(k): sum(p["gap"] == k for p in near_pool) for k in sorted({p["gap"] for p in near_pool})},
        },
        "unique_after_dedupe": len(ranked_all), "promos_ranked": len(final),
        "final_by_kind": {k: sum(c["kind"] == k for c in final) for k in ("single", "tightened", "multi")},
        "not_selected": {r: sum(p["not_selected_reason"] == r for p in skipped) for r in {p["not_selected_reason"] for p in skipped}},
        "target_seconds": pc.get("target_seconds"), "max_promos_per_episode": cap, "final_promo_count": n, "duration_range": [pc["min_seconds"], pc["max_seconds"]],
        "final_durations": [p["duration"] for p in final],
        "episodes_covered": sorted({e for p in final for e in p["episodes"]}),
        "near_duplicates_removed": dupes,
        "validation_status_counts": {s: sum(p["status"] == s for p in ranked) for s in ("pass", "warn", "fail")},
        "issue_counts": {}, "llm_usage": llm.usage, "model": llm.active_model,
        "runtime_seconds": round(time.time() - t0, 1), "weights": w,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for p in ranked:
        for v in p["validation"]:
            summary["issue_counts"][v["code"]] = summary["issue_counts"].get(v["code"], 0) + 1
    write_outputs(out_dir, ranked, cands, tags, summary)
    say(1.0, f"Done: {len(ranked)} ranked promos in {summary['runtime_seconds']}s. Tokens: {llm.usage}")
    return {"promos": ranked, "summary": summary}


def gate_label(p) -> str:
    if p.get("gate_pass"):
        return "PASSED ALL CRITERIA"
    if p.get("near_miss"):
        return f"NEAR MISS (gap {p['gap']}): " + "; ".join(p["gate"]["hook"]["reasons"] + p["gate"]["cliffhanger"]["reasons"])
    return "FAILED GATE"


def _reason_counts(promos) -> dict:
    """Group failure reasons into short categories, e.g. 'hook: AI score < 4'."""
    labels = [("does not start within", "hook not in first 3 seconds"), ("AI hook score", "AI hook score below threshold"),
              ("zero context", "hook needs context"), ("AI cliffhanger score", "AI cliffhanger score below threshold"),
              ("not unresolved", "ending is not unresolved"), ("does not end within", "cut is not right after the cliffhanger beat"),
              ("resolves it", "next line in script resolves it"), ("not judged", "not judged"), ("could not run", "next-line check failed")]
    out = {}
    for p in promos:
        for test in ("hook", "cliffhanger"):
            for r in p["gate"][test]["reasons"]:
                key = f"{test}: " + next((lab for k, lab in labels if k in r), r)
                out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def write_rejected(out_dir, rejected):
    with open(out_dir / "rejected_promos.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "kind", "episodes", "segments", "duration_s", "ai_score", "hook_score", "cliffhanger_score",
                     "hook_test", "hook_fail_reasons", "cliffhanger_test", "cliffhanger_fail_reasons", "failed",
                     "hook_text", "cliffhanger_text", "unresolved_type", "last_line_of_promo", "next_lines_in_script",
                     "next_line_check", "gap_to_passing", "used_in_final_as_near_miss"])
        for p in sorted(rejected, key=lambda x: (x["gap"] if x["gap"] is not None else 99, -x["score"])):
            j, gt = p["judge"], p["gate"]
            failed = " + ".join(t for t in ("hook", "cliffhanger") if not gt[t]["pass"])
            nc = p.get("next_check")
            wr.writerow([p["id"], p["kind"], " ".join(map(str, p["episodes"])),
                         " | ".join(f"E{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]), p["duration"],
                         p["score"], j["hook"], j["cliffhanger"],
                         "PASS" if gt["hook"]["pass"] else "FAIL", "; ".join(gt["hook"]["reasons"]),
                         "PASS" if gt["cliffhanger"]["pass"] else "FAIL", "; ".join(gt["cliffhanger"]["reasons"]), failed,
                         j["hook_text"], j["cliffhanger_text"], j.get("unresolved_type", ""),
                         p["source_lines"][-1] if p["source_lines"] else "", " || ".join(p["next_lines"]),
                         "" if nc is None else ("resolves: " if nc.get("next_line_resolves") else "does not resolve: ") + nc.get("reason", ""),
                         "hard fail (timing)" if p["gap"] is None else p["gap"], "YES" if p.get("near_miss") else ""])


def write_outputs(out_dir, ranked, cands, tags, summary):
    (out_dir / "promos.json").write_text(json.dumps(ranked, indent=1))
    (out_dir / "all_candidates.json").write_text(json.dumps(cands, indent=1))
    (out_dir / "episode_tags.json").write_text(json.dumps({str(k): v for k, v in sorted(tags.items())}, indent=1))
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    with open(out_dir / "promos.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "gate_result", "id", "score", "status", "kind", "episodes", "segments", "duration_s", "hook", "cliffhanger",
                     "why_it_works", "risks"] + CRITERIA + ["validation_issues"])
        for p in ranked:
            j = p["judge"]
            wr.writerow([p["rank"], gate_label(p), p["id"], p["score"], p["status"], p["kind"], " ".join(map(str, p["episodes"])),
                         " | ".join(f"E{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"]), p["duration"],
                         j["hook_text"], j["cliffhanger_text"], j["why_it_works"], j["risks"]] + [j[k] for k in CRITERIA]
                        + ["; ".join(f"{v['level']}:{v['code']}" for v in p["validation"])])
    # validation report
    vr = [f"# Validation report\n\nGenerated {summary['generated_at']} - {summary['promos_ranked']} ranked promos.\n",
          f"Status counts: {summary['validation_status_counts']}\n", f"Issue counts: {summary['issue_counts']}\n",
          f"Hook + cliffhanger gate: {summary['gate']['passed_both']} passed both, {summary['gate']['failed_hook_test']} failed hook test, "
          f"{summary['gate']['failed_cliffhanger_test']} failed cliffhanger test. Reasons: {summary['gate']['fail_reason_counts']}\n",
          f"Tagging failures: {summary['tag_failures'] or 'none'}\n",
          f"Near-duplicates removed ({len(summary['near_duplicates_removed'])}): " +
          ", ".join(f"{d['id']} (dup of {d['duplicate_of']})" for d in summary["near_duplicates_removed"]) + "\n",
          "\n## Per promo\n"]
    for p in ranked:
        vr.append(f"- #{p['rank']} {p['id']} [{p['status']}]" + ("" if not p["validation"] else ": " + "; ".join(f"{v['level']} {v['code']} - {v['msg']}" for v in p["validation"])))
    (out_dir / "validation_report.md").write_text("\n".join(vr) + "\n")
    # markdown
    good = [p for p in ranked if p["status"] != "fail"]
    md = [f"# Promos for '{summary['show']}'\n",
          f"{summary['episodes_analyzed']} episodes analyzed, {summary['candidates_built']} candidates, {summary['candidates_judged']} judged by AI, "
          f"{summary['unique_after_dedupe']} passed the strict gate (after removing near-duplicates); final list = {summary['promos_ranked']} "
          f"(target {summary['target_seconds']}s, allowed {summary['duration_range'][0]}-{summary['duration_range'][1]}s, "
          f"max {summary['max_promos_per_episode']} per episode). "
          f"Rejected candidates are in rejected_promos.csv. Model: {summary['model']}. Generated {summary['generated_at']}.\n",
          "Scores come from an AI judge reading screenplay TEXT only (no acting, visuals or sound). Every promo cites its source lines.\n",
          "## Strict hook + cliffhanger gate\n",
          f"Thresholds: AI hook >= {summary['gate']['thresholds']['hook_min_score']}/5, AI cliffhanger >= "
          f"{summary['gate']['thresholds']['cliffhanger_min_score']}/5, hook in first 3s, hook works with zero context, "
          f"ending unresolved, cut within {summary['gate']['thresholds']['cliff_end_window_seconds']}s of the cliffhanger beat, "
          "next script line does not resolve it.\n",
          f"Of {summary['candidates_judged']} judged candidates: **{summary['gate']['passed_both']} passed both tests**, "
          f"{summary['gate']['failed_hook_test']} failed the hook test, {summary['gate']['failed_cliffhanger_test']} failed the "
          f"cliffhanger test ({summary['gate']['failed_both']} failed both). Rejected promos and reasons: rejected_promos.csv.\n"]
    gs = summary["gate"]
    if gs["final_list_short"]:
        if gs["in_final_near_misses"]:
            md.append(f"**Only {gs['in_final_passed_both']} promos pass every criterion. To reach {summary['final_promo_count']}, the remaining "
                      f"{gs['in_final_near_misses']} slots are filled with the CLOSEST near-misses (smallest gap to passing, then AI score). "
                      "They are labelled NEAR MISS with exactly what they failed. Timing checks (hook in first 3s, cut right after "
                      "the cliffhanger, 27-33s, no mid-line cuts) were never relaxed.**\n")
        else:
            md.append(f"**Only {summary['promos_ranked']} promos made the final list - fewer than the {summary['final_promo_count']} requested. "
                      "The bar was NOT lowered to fill the list.**\n")
    md.append("## Top 5 to post first (from promos that passed every criterion)\n")
    for p in [x for x in good if not x.get("near_miss")][:5]:
        j = p["judge"]
        md.append(f"{p['rank']}. **{p['id']}** - score {p['score']}/100 - {p['duration']}s. {j['why_it_works']} Hook: {j['hook_text']}")
    md.append("\n## Full ranked list\n")
    for p in ranked:
        j = p["judge"]
        md.append(f"### #{p['rank']} [{gate_label(p)}] {p['id']} - {p['score']}/100 - {p['duration']}s - status: {p['status']}")
        for k, s in enumerate(p["segments"], 1):
            md.append(f"- Segment {k}: Episode {s['episode']} {s['start_ts']} -> {s['end_ts']}")
        md += [f"- Hook: {j['hook_text']}", f"- Cliffhanger: {j['cliffhanger_text']}", f"- Why it works: {j['why_it_works']}",
               f"- Risks: {j['risks']}", "- Scores: " + ", ".join(f"{k}={j[k]}" for k in CRITERIA),
               f"- Gate: Hook {'PASS' if p['gate']['hook']['pass'] else 'FAIL'}, Cliffhanger {'PASS' if p['gate']['cliffhanger']['pass'] else 'FAIL'}"
               f" (ending type: {j.get('unresolved_type', '')}; next script line: {(p.get('next_check') or {}).get('reason', 'n/a')})"]
        if p["validation"]:
            md.append("- Validation: " + "; ".join(f"{v['level']} {v['code']}" for v in p["validation"]))
        md.append("")
    (out_dir / "promos.md").write_text("\n".join(md))
