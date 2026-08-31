# 排行榜 v2（乱序基线归一 · Δ = NLL_shuffled − NLL_text，越大越好）

## CYYT_ATIS_a

| # | Δ_own | Δ_cross | 共识Δ | NLL@own | @t | 词数 | id | 来源 | 标签 |
|---|-------|---------|-------|---------|----|------|----|------|------|
| 1 | 3.842 | 2.833 | 3.338 | 0.189 | 85.0 | 69 | R050 | longpipe_CYYT_ATIS_a.txt | A pipeline+VAD(0.6) 去重单条(69词) |
| 2 | 3.842 | 2.833 | 3.338 | 0.189 | 85.0 | 69 | R052 | longpipe_CYYT_ATIS_a.txt | B pipeline无VAD(0.99) 去重单条(69词) |
| 3 | 3.901 | 2.679 | 3.290 | 0.147 | 0.0 | 70 | R032 | segmap_CYYT_ATIS_a.txt | chunk[0-30s] |
| 4 | 4.025 | 2.491 | 3.258 | 0.098 | 60.0 | 62 | R034 | segmap_CYYT_ATIS_a.txt | chunk[60-90s] |
| 5 | 3.705 | 2.622 | 3.164 | 0.371 | 0.0 | 75 | R063 | loopvote_CYYT_ATIS_a.txt | 第1实例单遍 |
| 6 | 3.777 | 2.464 | 3.120 | 0.179 | 90.0 | 72 | R035 | segmap_CYYT_ATIS_a.txt | chunk[90-120s] |
| 7 | 3.771 | 2.426 | 3.098 | 0.174 | 30.0 | 71 | R033 | segmap_CYYT_ATIS_a.txt | chunk[30-60s] |
| 8 | 3.420 | 2.318 | 2.869 | 0.452 | 60.0 | 75 | R027 | CYYT_ATIS_a_best.txt | CYYT_ATIS_a_best |
| 9 | 3.420 | 2.318 | 2.869 | 0.452 | 60.0 | 75 | R030 | CYYT_ATIS_a.txt | best_v3 |
| 10 | 3.200 | 2.253 | 2.726 | 0.550 | 85.0 | 73 | R054 | longpipe_CYYT_ATIS_a.txt | C 旧25s滑窗 去重单条(73词) |
| 11 | 2.809 | 2.086 | 2.447 | 2.558 | 0.0 | 77 | R000 | result_denoised.txt | result_denoised |
| 12 | 2.430 | 2.279 | 2.354 | 2.196 | 35.0 | 187 | R023 | result.txt | result |
| 13 | 2.463 | 2.048 | 2.256 | 1.457 | 0.0 | 75 | R062 | loopvote_CYYT_ATIS_a.txt | 投票结果 |
| 14 | 2.249 | 2.144 | 2.197 | 3.108 | 35.0 | 198 | R022 | qwen_result.txt | qwen_result |
| 15 | 1.724 | 1.601 | 1.663 | 4.161 | -1.0 | 257 | R005 | result_1.txt | result_1 |
| 16 | 1.685 | 1.591 | 1.638 | 4.217 | -1.0 | 257 | R008 | result_4.txt | result_4 |
| 17 | 1.586 | 1.542 | 1.564 | 4.191 | -1.0 | 258 | R006 | result_2.txt | result_2 |
| 18 | 1.608 | 1.494 | 1.551 | 2.774 | -1.0 | 305 | R026 | CYYT_ATIS_a_best.raw.txt | CYYT_ATIS_a_best.raw |
| 19 | 1.608 | 1.494 | 1.551 | 2.774 | -1.0 | 305 | R049 | longpipe_CYYT_ATIS_a.txt | A pipeline+VAD(0.6) 原始(305词) |
| 20 | 1.608 | 1.494 | 1.551 | 2.774 | -1.0 | 305 | R051 | longpipe_CYYT_ATIS_a.txt | B pipeline无VAD(0.99) 原始(305词) |
| 21 | 1.608 | 1.472 | 1.540 | 4.209 | -1.0 | 258 | R009 | result_5.txt | result_5 |
| 22 | 1.590 | 1.480 | 1.535 | 4.173 | -1.0 | 259 | R007 | result_3.txt | result_3 |
| 23 | 1.499 | 1.416 | 1.458 | 3.081 | -1.0 | 303 | R064 | loopvote_CYYT_ATIS_a.txt | 全部分块 |
| 24 | 1.444 | 1.311 | 1.378 | 4.408 | -1.0 | 254 | R004 | denoised_dn1_result.txt | denoised_dn1_result |
| 25 | 1.244 | 1.073 | 1.158 | 3.397 | -1.0 | 560 | R001 | result_vad.txt | result_vad |
| 26 | 0.684 | 0.640 | 0.662 | 3.958 | -1.0 | 362 | R053 | longpipe_CYYT_ATIS_a.txt | C 旧25s滑窗 原始(362词) |

## CYYT_ATIS_b

| # | Δ_own | Δ_cross | 共识Δ | NLL@own | @t | 词数 | id | 来源 | 标签 |
|---|-------|---------|-------|---------|----|------|----|------|------|
| 1 | 3.351 | 2.468 | 2.909 | 0.857 | 235.0 | 97 | R028 | CYYT_ATIS_b_best.raw.txt | CYYT_ATIS_b_best.raw |
| 2 | 3.351 | 2.468 | 2.909 | 0.857 | 235.0 | 97 | R055 | longpipe_CYYT_ATIS_b.txt | A pipeline+VAD(0.6) 原始(97词) |
| 3 | 3.351 | 2.468 | 2.909 | 0.857 | 235.0 | 97 | R056 | longpipe_CYYT_ATIS_b.txt | A pipeline+VAD(0.6) 去重单条(97词) |
| 4 | 3.351 | 2.468 | 2.909 | 0.857 | 235.0 | 97 | R057 | longpipe_CYYT_ATIS_b.txt | B pipeline无VAD(0.99) 原始(97词) |
| 5 | 3.351 | 2.468 | 2.909 | 0.857 | 235.0 | 97 | R058 | longpipe_CYYT_ATIS_b.txt | B pipeline无VAD(0.99) 去重单条(97词) |
| 6 | 3.572 | 2.238 | 2.905 | 0.185 | 210.0 | 30 | R048 | segmap_CYYT_ATIS_b.txt | chunk[210-240s] |
| 7 | 3.023 | 2.602 | 2.812 | 1.602 | 230.0 | 185 | R059 | longpipe_CYYT_ATIS_b.txt | C 旧25s滑窗 原始(185词) |
| 8 | 3.023 | 2.602 | 2.812 | 1.602 | 230.0 | 185 | R060 | longpipe_CYYT_ATIS_b.txt | C 旧25s滑窗 去重单条(185词) |
| 9 | 3.189 | 2.372 | 2.781 | 0.860 | 235.0 | 88 | R029 | CYYT_ATIS_b_best.txt | CYYT_ATIS_b_best |
| 10 | 2.702 | 2.163 | 2.433 | 1.582 | 230.0 | 93 | R025 | result.txt | result |
| 11 | 2.507 | 2.150 | 2.328 | 1.798 | 230.0 | 83 | R031 | CYYT_ATIS_b.txt | best_v3 |
| 12 | 2.399 | 1.836 | 2.117 | 2.710 | 230.0 | 102 | R024 | qwen_result.txt | qwen_result |
| 13 | 2.296 | 1.761 | 2.028 | 3.996 | 235.0 | 86 | R011 | result_1.txt | result_1 |
| 14 | 2.175 | 1.710 | 1.942 | 4.124 | 235.0 | 86 | R014 | result_4.txt | result_4 |
| 15 | 2.203 | 1.681 | 1.942 | 4.144 | 235.0 | 86 | R015 | result_5.txt | result_5 |
| 16 | 2.182 | 1.660 | 1.921 | 4.152 | 235.0 | 86 | R013 | result_3.txt | result_3 |
| 17 | 2.147 | 1.614 | 1.881 | 4.224 | 235.0 | 86 | R012 | result_2.txt | result_2 |
| 18 | 1.784 | 1.413 | 1.599 | 5.207 | 235.0 | 72 | R010 | denoised_dn1_result.txt | denoised_dn1_result |
| 19 | 1.497 | 1.494 | 1.496 | 3.765 | 235.0 | 112 | R003 | result_vad.txt | result_vad |

## RJTT_CONTROL

| # | Δ_own | Δ_cross | 共识Δ | NLL@own | @t | 词数 | id | 来源 | 标签 |
|---|-------|---------|-------|---------|----|------|----|------|------|
| 1 | 1.029 | - | - | 5.600 | 65.0 | 52 | R017 | result_1.txt | result_1 |
| 2 | 0.977 | - | - | 5.836 | 65.0 | 52 | R021 | result_5.txt | result_5 |
| 3 | 0.912 | - | - | 5.758 | 65.0 | 52 | R018 | result_2.txt | result_2 |
| 4 | 0.885 | - | - | 5.925 | 65.0 | 52 | R020 | result_4.txt | result_4 |
| 5 | 0.848 | - | - | 5.987 | 65.0 | 52 | R019 | result_3.txt | result_3 |
| 6 | 0.766 | - | - | 6.175 | 85.0 | 36 | R016 | denoised_dn1_result.txt | denoised_dn1_result |
