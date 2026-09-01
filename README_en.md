# ATC Speech Recognition Research (ATC-Speech-Research)

[简体中文](./README.md) | [English](./README_en.md)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![last commit](https://img.shields.io/github/last-commit/CEasonK/ATC-Speech-Research)

An air-traffic-control (ATC) speech research repository built on [FunASR](https://github.com/modelscope/FunASR).
Under the extreme constraint of **no ground-truth transcripts and only three recordings**, it delivers
the full pipeline of **authoritative transcription → true streaming recognition → domain-grade Chinese
translation**, backed by a **fully objective, reproducible evaluation system**
(acoustic likelihood / hard ICAO·METAR·ATIS grammar constraints / cross-system verification).

> Methodological bottom line: only objective judges are trusted — **no subjective scoring, ever**.
>
> **Status: in progress.** The deep and translate phases have shipped authoritative final results;
> the streaming phase has completed the P4 audit and frozen the L2/L1 tiers.
> **A clean re-run of L0 with the sanitized lexicon is underway** (see §10).

---

## 0. Results at a glance (real end-to-end output, not a mock-up)

```
🎧 CYYT_ATIS_a.wav (ATIS looped broadcast, varying pass quality)
        │  deep phase: frozen by four-fold objective evidence
        ▼
SAINT JOHNS INFORMATION FOXTROT WEATHER AT ZERO TWO ZERO ZERO ZULU
WIND TWO FOUR ZERO AT FIVE
        │  translate phase: term-constrained translation + audit loop (numeric fidelity 1.0 / term hit 1.0)
        ▼
圣约翰斯 信息F 天气世界协调时零二零零
风向二四零，风速五节
```

In streaming mode the same pipeline emits incremental drafts at ~1.7 s median word
delay with automatic end-of-utterance refinement: L2 WER 0.0 / L1 0.0845
(protocol and caveats in §4, §7).

## 1. Research corpus (`TT/audio/`, read-only — no experiment may modify it)

| Recording | Content | Language | Characteristics |
|---|---|---|---|
| `CYYT_ATIS_a.wav` | St. John's (CYYT) ATIS looped broadcast, ~270 s | English | Repeats every ~28 s (~5 passes), each pass with different channel quality |
| `CYYT_ATIS_b.wav` | Same station, different time (argued to be a **different date**) | English | Much weaker signal — automatic language detection misfires as "no speech"; clipped noise tail |
| `RJTT_CONTROL.wav` | Haneda (RJTT) control-frequency comms, multiple callsigns | EN/JA | Multi-speaker, real controller–pilot interaction |

## 2. Environment & dependencies

- GPU: RTX 3090 24 GB
- Conda env: `lingbot-map` on the research machine (torch 2.13.0+cu130 / transformers 4.57.6 / funasr 1.4.2 installed locally); on a new machine just create a fresh env per §5
- Network (measured on a mainland-China machine):
  - huggingface.co unreachable → `export HF_ENDPOINT=https://hf-mirror.com`
  - pip via the Tsinghua mirror; GitHub clones need a ghproxy mirror
- The FunASR framework at the repo root is upstream code used as a dependency, unmodified. All research work lives strictly inside `TT/`.

## 3. Repository layout

```
FunASR-main/
├── funasr/                               # upstream FunASR framework (dependency, unmodified; other upstream dirs pruned)
└── TT/                                     # ★ the entire research workspace
    ├── audio/            # raw recordings (read-only)
    ├── denoise/
    │   ├── methods/      #   denoisers: NN_name.py implementing denoise(y, sr) is auto-registered
    │   ├── output/       #   outputs <rec>__m<ID>__dn<ver>.wav (version auto-increments, never overwritten)
    │   ├── qc_report/    #   QC reports (JSON for machines + MD for humans, name-matched to outputs)
    │   └── legacy/       #   quarantined legacy artifacts
    ├── results/          # production-pipeline ASR results (best_pipeline / ATC_Whisper / Qwen3ASR / FunASR)
    ├── scripts/          # production scripts (line-by-line docs in TT/scripts/README.md, zh)
    ├── models/           # model weights location (weights not in git, see §5)
    ├── REVIEW_LOG.md     # incremental code-review log
    └── research/
        ├── deep/         #   Phase 1: offline authoritative transcription
        ├── streaming/    #   Phase 2: true streaming recognition (in progress)
        ├── translate/    #   Phase 3: EN→ZH domain translation
        └── refs/         #   third-party references (SimulStreaming, etc.)
```

Every sub-project follows the same triad: `PLAN.md` (protocol & red lines) →
`JOURNAL.md` (per-round experiment log incl. negative results and errata) →
`FINAL_REPORT.md` (conclusions + evidence file manifest). Method details in §7.

## 4. Evaluation system (the methodological core)

- **Three objective judges**:
  ① NLL acoustic likelihood (forced scoring — a hypothesis cannot "feel right" without surviving the audio);
  ② hard ICAO/METAR/ATIS grammar constraints (slot structure, NATO alphabet, number reading formats);
  ③ cross-system verification (cross-model-family consensus) + physical measurements (energy / period / anchor-window probes).
- **Prior tiers (streaming phase — prevents prior leakage from inflating metrics)**:
  - **L0 — true zero prior**: no day-specific text at all; pure acoustics + published-standard lexicon
  - **L1 — +text prompt**: ATIS text used as a decoding prompt (industry analogue: system holds the publicly broadcast text)
  - **L2 — +template fusion**: station-template evidence fusion (template text derived from the deep final drafts)
  - Every reported number must state its tier; metadata carries a self-describing `prior=` field;
    tokens coming from template output are tagged `src=tpl`.
- **Controlled K4 protocol**: token-WER computed against the corrected reference (284-token edition, J12 reference fix).
  The earlier protocol (phantom-repetition reference) is retired, with the change recorded in `JOURNAL.md`.

## 5. Quick start: model installation & first run

Weights are never committed, and **every script loads models from fixed paths** —
the Location column below must be followed exactly; "anywhere convenient" will not run.

**Prerequisites** (versions in §2):

```bash
conda create -n atc python=3.10 -y && conda activate atc
pip install -e .                # install the bundled funasr to match the research env
pip install -U huggingface_hub modelscope   # download tools
export HF_ENDPOINT=https://hf-mirror.com    # required in mainland China; skip if HF is reachable
```

**Model installation table** (Location = the path scripts read by default; wrong directories = immediate errors):

| # | Model | Download command | Location (repo-root relative) | Used by |
|---|---|---|---|---|
| 1 | whisper-large-v3-finetuned-for-ATC | `hf download jacktol/whisper-large-v3-finetuned-for-ATC --local-dir TT/models/whisper-large-v3-finetuned-for-ATC` | `TT/models/whisper-large-v3-finetuned-for-ATC/` | `run_best_asr` / `run_atc_whisper` / streaming primary engine |
| 2 | Qwen3-ASR-1.7B | `modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir TT/models/Qwen3-ASR-1.7B` | `TT/models/Qwen3-ASR-1.7B/` + **required** `export QWEN_ASR_MODEL=$PWD/TT/models/Qwen3-ASR-1.7B` | `run_qwen` / streaming side-witness worker (or pass `run_2pass.py --qwen_model`) |
| 3 | Qwen2.5-7B-Instruct | `hf download Qwen/Qwen2.5-7B-Instruct --local-dir TT/research/translate/models/qwen2.5-7b-instruct` | `TT/research/translate/models/qwen2.5-7b-instruct/` | primary translation model |
| 4 | m2m100_418M | `hf download facebook/m2m100_418M --local-dir TT/research/translate/models/m2m100_418M` | `TT/research/translate/models/m2m100_418M/` | translation baseline / back-translation |
| 5 | whisper-large-v3 (vanilla) | no manual step: `from_pretrained("openai/whisper-large-v3")` auto-fetches via mirror | `~/.cache/huggingface/` | cross-verification / streaming 2nd refiner |
| 6 | faster-whisper (CT2) builds | converted locally from #1/#5 via `TT/research/streaming/src/convert_hf_to_openai.py` | `TT/research/streaming/downloads/_ct2_atc/`, `_ct2_v3/` | streaming phase only |

**Environment variables & path dependencies**:

- `QWEN_ASR_MODEL`: overrides the legacy default path for #2 (without it, scripts point at a
  nonexistent `/siyuan/Qwen3_ASR/...` and fail immediately).
- `HF_ENDPOINT=https://hf-mirror.com`: needed by everything that touches HF (#5 and the deep reproductions).
- `TT/research/streaming/src/common.py` hard-codes `TT_ROOT` to the research machine's path —
  edit this one constant when deploying streaming reproductions elsewhere, or mirror the same path.

**Smoke test** (works as soon as #1 is installed):

```bash
cd TT && python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
# compare verbatim against the committed results/best_pipeline/CYYT_ATIS_a/result.txt
```

## 6. Production workflow (two steps)

```bash
cd TT

# ① denoise (optional — the default pipeline uses raw audio; see lessons below)
conda run -n lingbot-map python scripts/run_denoise.py            # all audios × all methods
conda run -n lingbot-map python scripts/run_denoise.py --list-methods
conda run -n lingbot-map python scripts/run_denoise.py --method 2 --audio CYYT_ATIS_a

# ② recognition (run_best_asr.py = frozen research conclusions; use this day-to-day)
conda run -n lingbot-map python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
#   --full      keep every loop pass (default: anchor-based dedup, keep the single cleanest pass)
#   --no-clean  raw decoder output only (post-processing off)
#   --anchor    change the loop anchor phrase (for other message types)

# single-model comparisons
conda run -n lingbot-map python scripts/run_atc_whisper.py --all        # 30 s sliding window + 2 s overlap
conda run -n lingbot-map python scripts/run_qwen.py CYYT_ATIS_a --lang English
conda run -n lingbot-map python scripts/qc_check.py audio/CYYT_ATIS_a.wav   # SNR / loudness / peak only
```

Outputs land in `results/<model>/<recording>/result.txt + result.json`.

**Key engineering decisions already frozen** (each backed by experiments; read `TT/scripts/README.md` before changing):
- Raw audio passes through with **no denoising**: denoising helps the noisy recording (a) but hurts the quiet one (b);
  no single parameter set wins on both. Confirmed again by streaming experiment R7 (denoiser front-end worsens WER
  by erasing weak-speech acoustic cues).
- ATIS loop dedup uses **whole-pass terminology scoring**, not per-word voting (per-word consensus fails 44% of the time).
- Hallucination filtering uses Whisper's built-in no_speech VAD (energy VAD is useless on full-energy noise segments).
- Qwen3-ASR must be given an explicit `--lang English`: auto-detection returns empty on the weak-signal recording
  (measured lesson, 2026-08-21).

## 7. The three research phases

### Phase 1 · deep — authoritative transcription without ground truth (done)

**Task**: no reference text of any kind, only three recordings — get as close to ground
truth as possible, with evidence available for every single word.

**Method core: three judges × five classes of objective tools** ("it sounds right" is never a judge):

1. **Tri-judge system**: whisper-atc (strong domain prior, self-biased) / whisper-large-v3
   vanilla (neutral) / turbo-atcosim (third independent engine). Iron rule: **any field is
   frozen only with ≥2 independent evidence sources agreeing; a single judge never decides**.
2. **Paired-window adjudication**: competing hypotheses are forced-NLL scored inside the
   *same* anchor window, removing window-drift artifacts (early trap: scoring text against a
   silent window yields deceptively low NLL — the "silent-window illusion").
3. **LM-prior contamination calibration**: measured that single-word-insertion ΔNLL can reach
   1.2–1.4 nat purely from decoder language priors — within that band ΔNLL **must not decide
   alone** (case v11 WIND, Δ1.27, triggered exactly this rule).
4. **Slice decoding**: cut 14–16 s context-free snippets for independent free dictation by all
   engines, breaking long-audio context anchoring. Key finding: **unanimous absence ≠ acoustic
   absence** — reduced /ət/ and the nasal /wɪnd/ were missed 11/11 and 7/7 yet physically
   present → free-decode voting is positive evidence of presence only, never of absence.
5. **Energy-envelope physical probe (the tie-breaker)**: when NLL and decoding deadlock, a
   language-free 10 ms RMS envelope + burst detection gives the final ruling. Both hard cases
   were decided by it — a's reduced AT (voiced run at 57.88 s, RMS 0.06–0.14 vs 0.025–0.04 in
   genuine inter-word gaps) and b's WIND (nasal-plateau signature + 7 bursts mapped word by word).
6. **Grammar vetoes acoustics**: METAR/ATIS hard constraints (integer temperature, VHF
   118–137 MHz, QNH 28.xx format) outright reject "best-sounding" hypotheses that are illegal.

**Representative wins** (each reproducible via `exp/adjudicate_v*.py`):
- **The v4 retrial**: callsign SIERRA→SHANGHAI AIR (v3 free decode hit it independently in two
  segments + qwen paired-window agreement + domain prior — three sources; turbo's SIERRA kept
  on file as the rival hypothesis); ORANGE NINER→ORANGE LINER (Japanese-track
  オレンジライナー + qwen, two sources).
- **a/b relationship**: not re-recordings but broadcasts from **different dates** — the whole
  corpus differs in exactly two spots (QNH 3023 vs 3033; AS vs WHEN REQUESTED), both channel-level.
- **Trailing readback**: human listening + anchor-window physical probe confirm the last three
  lines of a are a genuine repetition, not model hallucination.

**Deliverables**: `results/a_final.txt`, `b_final.txt`, `rjtt_final.txt` (RJTT synthesized from
9 consensus segments with per-segment confidence tiers) — the reference standard for all
streaming / translate research. Full evidence chain in `research/deep/FINAL_REPORT.md`, Appendix A.

### Phase 2 · streaming — true streaming recognition (in progress)
- **Architecture**: SimulStreaming (AlignAtt) streaming engine on the ATC-finetuned whisper-large-v3 backbone;
  an RMS-CV modulation gate triggers offline refinement (beam=5) at utterance end;
  station-template evidence fusion (ATC CT2 primary refiner + Qwen3-ASR side-witness worker +
  low-evidence-rate guardrail); delivered 2-pass.
- **Current numbers** (K4 protocol):
  | Tier | a | b | Notes |
  |---|---|---|---|
  | L2 template fusion | 0.0 | 0.0 | word-for-word match with deep finals; caveat: template prior included, 35–39 % of output tokens tagged `src=tpl` |
  | L1 text prompt | 0.0845 | 0.1162 | template used as a text prompt (not zero-prior) |
  | L0 3-engine ROVER | ⚠ 0.1303 / 0.2711 → **retired, pending re-run** | | see the P4 erratum in §9 |
  - Latency: draft-track median word delay ~1.7–1.9 s (RTF ~0.5, meets the real-time constraint); final track ~14.5 s.
- **Negative results archived** (equally conclusions): a single side-witness — especially the same-family whisper-v3 —
  can drag the primary engine down; the m1 denoiser front-end worsens WER; the cross-period-consensus route does not hold;
  Qwen warm-up must happen at worker startup (first call costs 32 s).

### Phase 3 · translate — EN→ZH domain translation (done; quality tied to the input tier)
- **Architecture**: T3 rule-template translator (0 parameters; numeric-correctness judge + fallback) +
  T1 Qwen2.5-7B term-constrained translation (primary generator) + audit feedback loop
  (failing lines are re-fed with error reasons, ≤3 rounds); T4 M2M-100 as an independent baseline and zh→en back-translation.
- **Metrics**: numeric fidelity (English-spelled numbers → payload sequences must match exactly) and
  terminology hit rate (21 EN→ZH term pairs). Final drafts score **1.0 / 1.0** when fed the deep finals or L2 streaming output.
- **Established**: ATC translation is not free MT — it is a "terminology mapping + number-reading restoration" problem
  (bare M2M failure modes archived: WIND→"win", ALTIMETER→"maximum").
- **Known boundary**: feeding zero-prior ASR output end-to-end drops metrics sharply
  (a: numeric 0.359 / terms 0.833; b worse) — end-to-end quality depends on the prior tier.

## 8. Models overview

Download commands, exact locations and environment variables for all 6 model artifacts
are in the §5 installation table (weights are not in git).
One-liner: ATC-finetuned whisper (primary ASR) · whisper-large-v3 vanilla (cross-verification) ·
Qwen3-ASR (side witness) · Qwen2.5-7B (translation) · m2m100 (back-translation baseline) ·
SimulStreaming (streaming engine — code bundled in this repo).

## 9. Reproduction guide

1. **Production pipeline**: commands in §6 run out of the box (after fetching the weights in §8).
2. **deep finals**: follow the protocol in `research/deep/PLAN.md`, running `src/` scripts in order;
   diff every step's output against `results/*_final.txt` verbatim.
3. **streaming, all tiers**: `research/streaming/src/run_2pass.py <wav> <out> --chunk 1.0 --half --rover --no_prompt`
   for the L0 zero-prior tier; drop `--no_prompt` for L1; add template fusion for L2.
   Scoring always goes through `src/metrics.py` (K4 protocol).
4. **translate**: `research/translate/` ships standalone audit scripts that recompute numeric fidelity and
   terminology hit rate for any translation.
5. Doubt any number? Replay it round-by-round through `JOURNAL.md` (protocol changes and errata are all on record).

## 10. Current progress & TODO (2026-08-31)

- ✅ deep: authoritative finals for all three recordings, evidence chain closed
- ✅ translate: finals at 1.0/1.0; audit tooling independently reusable
- ✅ streaming: L1/L2 frozen; P1–P3 fixes + regression tests (`tests/test_p3_fixes.py`)
- 🔶 **streaming P4 audit erratum (2026-08-31)**: audit found that the R1–R9 lexicon/grammar rules had been
  selected by analyzing residual errors against the K4 evaluation-set ground truth (adaptation supervised by the gold
  reference). **L0 numbers 0.1303/0.2711/0.0211/0.2254 are all retired.** `atis_lexicon.py` has been sanitized to
  "derivable from published standards + never deletes engine tokens".
  **TODO: re-run a clean L0 with the sanitized rules and re-report** (both a/b tracks, full-file delivery and
  end-to-end translation linkage).
- ⬜ Planned but not yet landed: Kyutai STT and NeMo FastConformer streaming baselines (routes E3/E4 in the streaming PLAN)
- ⬜ Weak-channel robustness: residual errors on track b (period-seam misalignment, degraded-segment numeric collapse)
  remain open — an acoustic frontiers problem

## 11. License

This project is released under the **MIT License** (see [LICENSE_ATC.md](./LICENSE_ATC.md) in the repo root;
free to use, modify and redistribute for any purpose).

> Note: another `LICENSE` at the repo root is FunASR's original license
> (MIT, © 2025 FunASR) and applies only to its framework code. The ATC research
> content of this repo (`TT/` and this README) follows the project MIT License above.

## 12. Citation

If you use this project in a paper or report, please cite:

```bibtex
@misc{atc_speech_research,
  author = {CEasonK},
  title  = {ATC-Speech-Research: Air Traffic Control Speech Recognition via Fully Objective Evaluation},
  year   = {2026},
  url    = {https://github.com/CEasonK/ATC-Speech-Research}
}
```
