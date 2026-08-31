# 排行榜 v1（NLL 裁判 · ATC-whisper-large-v3）

排序键：双信道共识分（无交叉者用本信道分）。NLL 越低越贴合音频。

## CYYT_ATIS_a

| # | NLL@own | NLL@cross | 共识 | @t | 词数 | id | 来源 | 标签 |
|---|---------|-----------|------|----|------|----|------|------|
| 1 | 0.189 | 1.222 | 0.706 | 85.0 | 69 | R050 | longpipe_CYYT_ATIS_a.txt | A pipeline+VAD(0.6) 去重单条(69词) |
| 2 | 0.189 | 1.222 | 0.706 | 85.0 | 69 | R052 | longpipe_CYYT_ATIS_a.txt | B pipeline无VAD(0.99) 去重单条(69词) |
| 3 | 0.147 | 1.379 | 0.763 | 0.0 | 70 | R032 | segmap_CYYT_ATIS_a.txt | chunk[0-30s] |
| 4 | 0.179 | 1.500 | 0.840 | 90.0 | 72 | R035 | segmap_CYYT_ATIS_a.txt | chunk[90-120s] |
| 5 | 0.098 | 1.662 | 0.880 | 60.0 | 62 | R034 | segmap_CYYT_ATIS_a.txt | chunk[60-90s] |
| 6 | 0.174 | 1.613 | 0.894 | 30.0 | 71 | R033 | segmap_CYYT_ATIS_a.txt | chunk[30-60s] |
| 7 | 0.371 | 1.463 | 0.917 | 0.0 | 75 | R063 | loopvote_CYYT_ATIS_a.txt | 第1实例单遍 |
| 8 | 0.452 | 1.559 | 1.005 | 60.0 | 75 | R027 | CYYT_ATIS_a_best.txt | CYYT_ATIS_a_best |
| 9 | 0.452 | 1.559 | 1.005 | 60.0 | 75 | R030 | CYYT_ATIS_a.txt | best_v3 |
| 10 | 0.551 | 1.527 | 1.039 | 85.0 | 73 | R054 | longpipe_CYYT_ATIS_a.txt | C 旧25s滑窗 去重单条(73词) |
| 11 | 1.457 | 1.890 | 1.674 | 0.0 | 75 | R062 | loopvote_CYYT_ATIS_a.txt | 投票结果 |
| 12 | 2.196 | 2.357 | 2.277 | 35.0 | 187 | R023 | result.txt | result |
| 13 | 2.774 | 3.101 | 2.937 | -1.0 | 305 | R026 | CYYT_ATIS_a_best.raw.txt | CYYT_ATIS_a_best.raw |
| 14 | 2.774 | 3.101 | 2.937 | -1.0 | 305 | R049 | longpipe_CYYT_ATIS_a.txt | A pipeline+VAD(0.6) 原始(305词) |
| 15 | 2.774 | 3.101 | 2.937 | -1.0 | 305 | R051 | longpipe_CYYT_ATIS_a.txt | B pipeline无VAD(0.99) 原始(305词) |
| 16 | 2.558 | 3.349 | 2.953 | 0.0 | 77 | R000 | result_denoised.txt | result_denoised |
| 17 | 3.108 | 3.212 | 3.160 | 35.0 | 198 | R022 | qwen_result.txt | qwen_result |
| 18 | 3.081 | 3.324 | 3.202 | -1.0 | 303 | R064 | loopvote_CYYT_ATIS_a.txt | 全部分块 |
| 19 | 3.397 | 3.696 | 3.546 | -1.0 | 560 | R001 | result_vad.txt | result_vad |
| 20 | 3.958 | 4.133 | 4.045 | -1.0 | 362 | R053 | longpipe_CYYT_ATIS_a.txt | C 旧25s滑窗 原始(362词) |
| 21 | 4.173 | 4.364 | 4.269 | -1.0 | 259 | R007 | result_3.txt | result_3 |
| 22 | 4.161 | 4.383 | 4.272 | -1.0 | 257 | R005 | result_1.txt | result_1 |
| 23 | 4.191 | 4.361 | 4.276 | -1.0 | 258 | R006 | result_2.txt | result_2 |
| 24 | 4.209 | 4.427 | 4.318 | -1.0 | 258 | R009 | result_5.txt | result_5 |
| 25 | 4.217 | 4.424 | 4.321 | -1.0 | 257 | R008 | result_4.txt | result_4 |
| 26 | 4.408 | 4.660 | 4.534 | -1.0 | 254 | R004 | denoised_dn1_result.txt | denoised_dn1_result |

## CYYT_ATIS_b

| # | NLL@own | NLL@cross | 共识 | @t | 词数 | id | 来源 | 标签 |
|---|---------|-----------|------|----|------|----|------|------|
| 1 | 0.185 | 1.481 | 0.833 | 210.0 | 30 | R048 | segmap_CYYT_ATIS_b.txt | chunk[210-240s] |
| 2 | 0.860 | 1.668 | 1.264 | 235.0 | 88 | R029 | CYYT_ATIS_b_best.txt | CYYT_ATIS_b_best |
| 3 | 0.857 | 1.712 | 1.285 | 235.0 | 97 | R028 | CYYT_ATIS_b_best.raw.txt | CYYT_ATIS_b_best.raw |
| 4 | 0.857 | 1.712 | 1.285 | 235.0 | 97 | R055 | longpipe_CYYT_ATIS_b.txt | A pipeline+VAD(0.6) 原始(97词) |
| 5 | 0.857 | 1.712 | 1.285 | 235.0 | 97 | R056 | longpipe_CYYT_ATIS_b.txt | A pipeline+VAD(0.6) 去重单条(97词) |
| 6 | 0.857 | 1.712 | 1.285 | 235.0 | 97 | R057 | longpipe_CYYT_ATIS_b.txt | B pipeline无VAD(0.99) 原始(97词) |
| 7 | 0.857 | 1.712 | 1.285 | 235.0 | 97 | R058 | longpipe_CYYT_ATIS_b.txt | B pipeline无VAD(0.99) 去重单条(97词) |
| 8 | 1.602 | 1.983 | 1.793 | 230.0 | 185 | R059 | longpipe_CYYT_ATIS_b.txt | C 旧25s滑窗 原始(185词) |
| 9 | 1.602 | 1.983 | 1.793 | 230.0 | 185 | R060 | longpipe_CYYT_ATIS_b.txt | C 旧25s滑窗 去重单条(185词) |
| 10 | 1.582 | 2.107 | 1.845 | 230.0 | 93 | R025 | result.txt | result |
| 11 | 1.798 | 2.144 | 1.971 | 230.0 | 83 | R031 | CYYT_ATIS_b.txt | best_v3 |
| 12 | 2.710 | 3.272 | 2.991 | 230.0 | 102 | R024 | qwen_result.txt | qwen_result |
| 13 | 3.765 | 3.645 | 3.705 | 235.0 | 112 | R003 | result_vad.txt | result_vad |
| 14 | 3.996 | 4.498 | 4.247 | 235.0 | 86 | R011 | result_1.txt | result_1 |
| 15 | 4.124 | 4.568 | 4.346 | 235.0 | 86 | R014 | result_4.txt | result_4 |
| 16 | 4.144 | 4.648 | 4.396 | 235.0 | 86 | R015 | result_5.txt | result_5 |
| 17 | 4.152 | 4.649 | 4.401 | 235.0 | 86 | R013 | result_3.txt | result_3 |
| 18 | 4.224 | 4.737 | 4.480 | 235.0 | 86 | R012 | result_2.txt | result_2 |
| 19 | 5.207 | 5.534 | 5.371 | 235.0 | 72 | R010 | denoised_dn1_result.txt | denoised_dn1_result |

## RJTT_CONTROL

| # | NLL@own | NLL@cross | 共识 | @t | 词数 | id | 来源 | 标签 |
|---|---------|-----------|------|----|------|----|------|------|
| 1 | 5.600 | - | - | 65.0 | 52 | R017 | result_1.txt | result_1 |
| 2 | 5.758 | - | - | 65.0 | 52 | R018 | result_2.txt | result_2 |
| 3 | 5.836 | - | - | 65.0 | 52 | R021 | result_5.txt | result_5 |
| 4 | 5.925 | - | - | 65.0 | 52 | R020 | result_4.txt | result_4 |
| 5 | 5.987 | - | - | 65.0 | 52 | R019 | result_3.txt | result_3 |
| 6 | 6.175 | - | - | 85.0 | 36 | R016 | denoised_dn1_result.txt | denoised_dn1_result |
