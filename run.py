"""Command line: python run.py               (all episodes of the show in config.yaml)
                python run.py --episodes 1-3
                python run.py --show "data/My New Show" --out outputs_newshow
"""
import argparse

from pipeline import run_pipeline


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", help="e.g. 1-3 or 1,4,7 (default: all)")
    ap.add_argument("--show", help="folder of screenplay .txt files (default: show_dir in config.yaml)")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    run_pipeline(parse_range(a.episodes) if a.episodes else None, a.out, a.show)
