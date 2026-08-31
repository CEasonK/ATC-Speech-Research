"""SimulStreaming 参数扫描：串行跑配置网格，每个配置独立 exp 目录并自动评测。

用法：python sweep_simul.py [--manifest PATH] [--audio KEY]
网格（s2 域内微调模型为主）：
  frame_threshold ∈ {12,25,50} × chunk ∈ {0.5}  （fp16）
  最优 ft 上追加 chunk ∈ {0.25,1.0}
  静态提示词 ON/OFF 对照；audio_max_len 缩短对照
汇总写入 results/sweep_summary.md 与 results/sweep_summary.json
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"
PY = sys.executable


def run_one(name, wav, manifest, args_list):
    out = RESULTS / "simul" / name
    if (out / "metrics.json").exists():
        print(f"[skip] {name}")
        return json.load(open(out / "metrics.json"))
    cmd = [PY, str(SRC / "run_simulstreaming.py"), str(wav), str(out)] + args_list
    print("[run]", name, " ".join(args_list), flush=True)
    r = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-2500:], sep="\n")
        return {"error": r.stderr[-500:]}
    subprocess.run([PY, str(SRC / "evaluate_run.py"), str(out), str(manifest)],
                   cwd=str(SRC), capture_output=True, text=True)
    m = json.load(open(out / "metrics.json"))
    meta = json.load(open(out / "meta.json"))
    m["cfg"] = {k: v for k, v in meta.items() if k in
                ("frame_threshold", "chunk", "beams", "rtf")}
    m["half"] = "--half" in args_list
    m["prompt"] = any(a == "--static_prompt" for a in args_list)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="CYYT_ATIS_a")
    a = ap.parse_args()
    man = RESULTS / "eval_assets" / f"eval_manifest_{a.audio}.json"
    wav = RESULTS / "eval_assets" / f"{a.audio}_evalK4.wav"
    MODEL_ATC = "../downloads/whisper-atc-openai.pt"
    PROMPT = str(SRC / "static_prompt_atc.txt")

    configs = []
    # 组1：fp16 基础网格 + 提示词全程开启（对齐上游 SimulStreaming 论文推荐做法）
    for ft in (12, 25, 50):
        configs.append((f"atc_ft{ft}_c0.5_p", [
            "--model", MODEL_ATC, "--frame_threshold", str(ft),
            "--chunk", "0.5", "--half", "--static_prompt", PROMPT]))
    # 组2：无提示词对照（ft=25）
    configs.append(("atc_ft25_c0.5_nop", [
        "--model", MODEL_ATC, "--frame_threshold", "25",
        "--chunk", "0.5", "--half"]))
    # 组3：短 buffer 对照（跨周期漂移抑制）
    configs.append(("atc_ft25_c0.5_ml16_p", [
        "--model", MODEL_ATC, "--frame_threshold", "25", "--chunk", "0.5",
        "--half", "--audio_max_len", "16", "--static_prompt", PROMPT]))

    rows = []
    for name, argv in configs:
        m = run_one(name, wav, man, argv)
        m["name"] = name
        rows.append(m)
        print(json.dumps(m, ensure_ascii=False)[:300], flush=True)

    ok = [m for m in rows if "error" not in m]
    ok.sort(key=lambda m: m.get("wer_token_level", 9))
    (RESULTS / "sweep_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    lines = ["| 配置 | WER | tok延迟中位 | p95 | LAG | RTF | half | prompt |",
             "|---|---|---|---|---|---|---|---|"]
    for m in ok:
        tl = m.get("token_latency") or {}
        lines.append(
            f"| {m['name']} | {m.get('wer_token_level')} | {tl.get('median')}"
            f" | {tl.get('p95')} | {m.get('lag_mean')} | {m.get('rtf')}"
            f" | {m['half']} | {m['prompt']} |")
    (RESULTS / "sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
