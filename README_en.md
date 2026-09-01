# ATC Speech Recognition Research (ATC-Speech-Research)

[简体中文](./README.md) | [English](./README_en.md)

> An air-traffic-control (ATC) speech research workspace built on top of the FunASR framework.
> Core question: under the extreme constraint of **no ground-truth transcripts and only three recordings**,
> can we deliver the full pipeline of **authoritative transcription → true streaming recognition →
> domain-grade Chinese translation**, backed by a **fully objective, reproducible evaluation system**
> (acoustic likelihood / hard ICAO·METAR·ATIS grammar constraints / cross-system verification)?
> No subjective scoring is ever used as the judge.
>
> **Status: in progress.** The deep and translate phases have shipped authoritative final results;
> the streaming phase has completed the P4 audit and frozen the L2/L1 tiers.
> **A clean re-run of L0 with the sanitized lexicon is underway** (see §10).

---

## 0. Results at a glance (real end-to-end output, not a mock-up)

```
🎧 CYYT_ATIS_a.wav (weak-channel ATIS broadcast, ~5 looped passes)
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
- Conda env: `lingbot-map` (torch 2.13.0+cu130 / transformers 4.57.6 / funasr 1.4.2 installed locally)
- Network (measured on a mainland-China machine):
  - huggingface.co unreachable → `export HF_ENDPOINT=https://hf-mirror.com`
  - pip via the Tsinghua mirror; GitHub clones need a ghproxy mirror
- The FunASR framework at the repo root is upstream code used as a dependency, unmodified. All research work lives strictly inside `TT/`.

## 3. Repository layout

```
FunASR-main/
├── funasr/ · runtime/ · examples/ ...      # upstream FunASR framework (dependency, unmodified)
└── TT/                                     # ★ the entire research workspace
    ├── audio/            # raw recordings (read-only)
    ├── denoise/
    │   ├── methods/      #   denoisers: NN_name.py implementing denoise(y, sr) is auto-registered
    │   ├── output/       #   outputs <rec>__m<ID>__dn<ver>.wav (version auto-increments, never overwritten)
    │   ├── qc_report/    #   QC reports (JSON for machines + MD for humans, name-matched to outputs)
    │   └── legacy/       #   quarantined legacy artifacts
    ├── results/          # production-pipeline ASR results (best_pipeline / ATC_Whisper / Qwen3ASR / FunASR)
    ├── scripts/          # production scripts (line-by-line docs in TT/scripts/README.md, zh)
    ├── models/           # model weights location (weights not in git, see §8)
    ├── REVIEW_LOG.md     # incremental code-review log
    └── research/
        ├── deep/         #   Phase 1: offline authoritative transcription
        ├── streaming/    #   Phase 2: true streaming recognition (in progress)
        ├── translate/    #   Phase 3: EN→ZH domain translation
        └── refs/         #   third-party references (SimulStreaming, etc.)
```

Every sub-project follows the same triad: `PLAN.md` (protocol & red lines) →
`JOURNAL.md` (per-round experiment log incl. negative results and errata) →
`FINAL_REPORT.md` (conclusions + evidence file manifest).

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

## 5. Quick start & deploying on a new machine

```bash
# ① clone
git clone git@github.com:CEasonK/ATC-Speech-Research.git && cd ATC-Speech-Research

# ② environment (versions in §2; install the bundled funasr so it matches the research env)
conda create -n atc python=3.10 -y && conda activate atc
pip install -e .
pip install torch transformers modelscope funasr noisereduce soundfile  # extend as needed

# ③ weights: fetch per the table in §8 into TT/models/ (if HF is unreachable: export HF_ENDPOINT=https://hf-mirror.com)

# ④ run your first transcription
cd TT && python scripts/run_best_asr.py audio/CYYT_ATIS_a.wav
# output: results/best_pipeline/CYYT_ATIS_a/result.txt
```

No extra deployment is needed to reproduce the research phases (deep / streaming /
translate) — just use the conda env described in §9.

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
- **Deliverables**: `results/a_final.txt`, `b_final.txt`, `rjtt_final.txt` — authoritative final drafts
  that serve as the reference standard for all later phases.
- **Representative findings**: a and b are broadcasts from different dates (only two channel-level differences:
  3023 vs 3033, AS vs WHEN REQUESTED); the three trailing lines of a are a genuine readback
  (human listening + independent anchor-window probes), not model hallucination.
- Reproduction: `research/deep/` (PLAN defines the four-fold evidence protocol; `src/` contains the scripts;
  Appendix A lists every evidence file).

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

## 8. Model dependencies (weights are not in git; fetch them into `TT/models/` or the HF cache)

| Model | Role | Source |
|---|---|---|
| [whisper-large-v3-finetuned-for-ATC](https://huggingface.co/jacktol/whisper-large-v3-finetuned-for-ATC) | primary ATC recognition engine | HF → `TT/models/` |
| openai/whisper-large-v3 | cross-verification / ROVER side-witness | `from_pretrained` (via hf-mirror) |
| Qwen/Qwen3-ASR-1.7B | side-witness ASR worker | ModelScope / HF |
| Qwen2.5-7B-Instruct | primary translation model | HF |
| facebook/m2m100_418M | back-translation baseline | HF |
| SimulStreaming (AlignAtt) | streaming decoding engine | bundled: `TT/research/refs/SimulStreaming-main` |

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

Upstream FunASR remains under its original MIT License; the research content under `TT/` is owned by the repository author.
