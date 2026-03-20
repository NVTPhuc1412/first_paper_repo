# XAI Analysis Report

Explainable AI analysis of the top anomaly events using **Integrated Gradients** and **TimeSHAP** for the two selected models.

---

## Anomaly Transformer (`AT`)

### NVDA

#### 2024-08-05 — score: 0.826895

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000137 |
| 2 | f1_log_ret | 0.000047 |
| 3 | f6_rsi_14 | 0.000035 |
| 4 | f4_vol_z | 0.000029 |
| 5 | f2_gap_ret | 0.000022 |
| 6 | f8_bb_width | 0.000019 |
| 7 | f3_parkinson | 0.000018 |
| 8 | f5_rel_ret | 0.000016 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2024-08-05.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=91 | 0.002771 |
| 2 | t=88 | 0.001848 |
| 3 | t=51 | 0.001314 |
| 4 | t=54 | -0.001281 |
| 5 | t=48 | 0.001260 |

> **Recency concentration**: 32.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2024-08-05.png)

---

#### 2025-01-27 — score: 0.510169

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000062 |
| 2 | f6_rsi_14 | 0.000017 |
| 3 | f3_parkinson | 0.000016 |
| 4 | f4_vol_z | 0.000014 |
| 5 | f8_bb_width | 0.000013 |
| 6 | f1_log_ret | 0.000010 |
| 7 | f2_gap_ret | 0.000005 |
| 8 | f5_rel_ret | 0.000005 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2025-01-27.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=55 | 0.001152 |
| 2 | t=81 | 0.001041 |
| 3 | t=57 | 0.000971 |
| 4 | t=83 | -0.000961 |
| 5 | t=31 | -0.000953 |

> **Recency concentration**: 25.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2025-01-27.png)

---

#### 2024-05-28 — score: 0.274446

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000165 |
| 2 | f8_bb_width | 0.000061 |
| 3 | f6_rsi_14 | 0.000056 |
| 4 | f2_gap_ret | 0.000037 |
| 5 | f4_vol_z | 0.000031 |
| 6 | f5_rel_ret | 0.000030 |
| 7 | f3_parkinson | 0.000025 |
| 8 | f1_log_ret | 0.000009 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2024-05-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=69 | 0.009221 |
| 2 | t=95 | 0.003977 |
| 3 | t=67 | 0.002738 |
| 4 | t=62 | -0.002058 |
| 5 | t=17 | 0.002051 |

> **Recency concentration**: 11.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2024-05-28.png)

---

#### 2020-03-16 — score: 0.247893

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000011 |
| 2 | f7_macd_hist | 0.000007 |
| 3 | f2_gap_ret | 0.000005 |
| 4 | f6_rsi_14 | 0.000005 |
| 5 | f3_parkinson | 0.000003 |
| 6 | f4_vol_z | 0.000003 |
| 7 | f5_rel_ret | 0.000002 |
| 8 | f1_log_ret | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2020-03-16.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=90 | 0.000048 |
| 2 | t=81 | 0.000041 |
| 3 | t=95 | 0.000038 |
| 4 | t=80 | 0.000035 |
| 5 | t=79 | 0.000029 |

> **Recency concentration**: 65.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2020-03-16.png)

---

#### 2024-05-23 — score: 0.130792

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000141 |
| 2 | f3_parkinson | 0.000078 |
| 3 | f6_rsi_14 | 0.000078 |
| 4 | f8_bb_width | 0.000077 |
| 5 | f4_vol_z | 0.000057 |
| 6 | f2_gap_ret | 0.000055 |
| 7 | f5_rel_ret | 0.000040 |
| 8 | f1_log_ret | 0.000034 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2024-05-23.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=71 | 0.014098 |
| 2 | t=72 | -0.002172 |
| 3 | t=70 | -0.001883 |
| 4 | t=69 | 0.001777 |
| 5 | t=65 | -0.001638 |

> **Recency concentration**: 5.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2024-05-23.png)

---

#### 2015-08-07 — score: 0.083451

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000007 |
| 2 | f2_gap_ret | 0.000006 |
| 3 | f6_rsi_14 | 0.000005 |
| 4 | f5_rel_ret | 0.000004 |
| 5 | f4_vol_z | 0.000003 |
| 6 | f7_macd_hist | 0.000002 |
| 7 | f3_parkinson | 0.000001 |
| 8 | f1_log_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2015-08-07.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=32 | 0.000287 |
| 2 | t=33 | 0.000181 |
| 3 | t=44 | 0.000059 |
| 4 | t=31 | -0.000035 |
| 5 | t=28 | -0.000032 |

> **Recency concentration**: 16.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2015-08-07.png)

---

#### 2022-11-10 — score: 0.065259

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000005 |
| 2 | f8_bb_width | 0.000002 |
| 3 | f2_gap_ret | 0.000002 |
| 4 | f4_vol_z | 0.000002 |
| 5 | f3_parkinson | 0.000002 |
| 6 | f5_rel_ret | 0.000001 |
| 7 | f6_rsi_14 | 0.000000 |
| 8 | f1_log_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2022-11-10.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=46 | 0.000558 |
| 2 | t=95 | 0.000266 |
| 3 | t=47 | -0.000188 |
| 4 | t=89 | -0.000119 |
| 5 | t=42 | 0.000081 |

> **Recency concentration**: 26.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2022-11-10.png)

---

#### 2025-10-29 — score: 0.062309

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000017 |
| 2 | f6_rsi_14 | 0.000007 |
| 3 | f3_parkinson | 0.000006 |
| 4 | f8_bb_width | 0.000004 |
| 5 | f4_vol_z | 0.000004 |
| 6 | f1_log_ret | 0.000004 |
| 7 | f2_gap_ret | 0.000004 |
| 8 | f5_rel_ret | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2025-10-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=54 | 0.002357 |
| 2 | t=11 | -0.001400 |
| 3 | t=12 | -0.001322 |
| 4 | t=57 | 0.001312 |
| 5 | t=19 | 0.001149 |

> **Recency concentration**: 18.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2025-10-29.png)

---

#### 2018-11-19 — score: 0.054306

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000024 |
| 2 | f6_rsi_14 | 0.000019 |
| 3 | f8_bb_width | 0.000013 |
| 4 | f1_log_ret | 0.000011 |
| 5 | f7_macd_hist | 0.000008 |
| 6 | f4_vol_z | 0.000008 |
| 7 | f5_rel_ret | 0.000006 |
| 8 | f2_gap_ret | 0.000006 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2018-11-19.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=78 | 0.000094 |
| 2 | t=80 | 0.000083 |
| 3 | t=79 | 0.000070 |
| 4 | t=82 | 0.000065 |
| 5 | t=83 | 0.000057 |

> **Recency concentration**: 72.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2018-11-19.png)

---

#### 2022-01-24 — score: 0.053984

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000020 |
| 2 | f7_macd_hist | 0.000016 |
| 3 | f6_rsi_14 | 0.000011 |
| 4 | f2_gap_ret | 0.000008 |
| 5 | f4_vol_z | 0.000008 |
| 6 | f3_parkinson | 0.000007 |
| 7 | f1_log_ret | 0.000007 |
| 8 | f5_rel_ret | 0.000005 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2022-01-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=48 | 0.000220 |
| 2 | t=40 | 0.000137 |
| 3 | t=41 | -0.000121 |
| 4 | t=45 | 0.000098 |
| 5 | t=70 | 0.000092 |

> **Recency concentration**: 7.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2022-01-24.png)

---

### TSLA

#### 2020-03-16 — score: 0.261300

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f6_rsi_14 | 0.000007 |
| 2 | f1_log_ret | 0.000006 |
| 3 | f7_macd_hist | 0.000005 |
| 4 | f8_bb_width | 0.000004 |
| 5 | f5_rel_ret | 0.000004 |
| 6 | f3_parkinson | 0.000003 |
| 7 | f4_vol_z | 0.000003 |
| 8 | f2_gap_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2020-03-16.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=68 | -0.000135 |
| 2 | t=90 | 0.000131 |
| 3 | t=70 | 0.000087 |
| 4 | t=94 | 0.000087 |
| 5 | t=95 | 0.000084 |

> **Recency concentration**: 48.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2020-03-16.png)

---

#### 2023-01-26 — score: 0.250867

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000007 |
| 2 | f7_macd_hist | 0.000007 |
| 3 | f5_rel_ret | 0.000005 |
| 4 | f4_vol_z | 0.000004 |
| 5 | f6_rsi_14 | 0.000003 |
| 6 | f1_log_ret | 0.000003 |
| 7 | f2_gap_ret | 0.000002 |
| 8 | f3_parkinson | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2023-01-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=75 | 0.000151 |
| 2 | t=95 | 0.000137 |
| 3 | t=79 | -0.000116 |
| 4 | t=25 | 0.000096 |
| 5 | t=26 | 0.000095 |

> **Recency concentration**: 36.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2023-01-26.png)

---

#### 2025-03-10 — score: 0.244965

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000030 |
| 2 | f1_log_ret | 0.000022 |
| 3 | f8_bb_width | 0.000015 |
| 4 | f4_vol_z | 0.000011 |
| 5 | f6_rsi_14 | 0.000011 |
| 6 | f2_gap_ret | 0.000010 |
| 7 | f3_parkinson | 0.000005 |
| 8 | f5_rel_ret | 0.000003 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2025-03-10.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=19 | 0.000458 |
| 2 | t=95 | 0.000336 |
| 3 | t=17 | 0.000328 |
| 4 | t=22 | -0.000327 |
| 5 | t=18 | 0.000320 |

> **Recency concentration**: 25.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2025-03-10.png)

---

#### 2022-03-28 — score: 0.192251

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000012 |
| 2 | f2_gap_ret | 0.000008 |
| 3 | f8_bb_width | 0.000003 |
| 4 | f6_rsi_14 | 0.000003 |
| 5 | f5_rel_ret | 0.000003 |
| 6 | f4_vol_z | 0.000003 |
| 7 | f3_parkinson | 0.000002 |
| 8 | f1_log_ret | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2022-03-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=73 | -0.000808 |
| 2 | t=71 | -0.000492 |
| 3 | t=54 | 0.000488 |
| 4 | t=75 | -0.000379 |
| 5 | t=94 | -0.000360 |

> **Recency concentration**: 21.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2022-03-28.png)

---

#### 2025-09-15 — score: 0.184143

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000099 |
| 2 | f4_vol_z | 0.000037 |
| 3 | f8_bb_width | 0.000029 |
| 4 | f5_rel_ret | 0.000028 |
| 5 | f6_rsi_14 | 0.000023 |
| 6 | f2_gap_ret | 0.000022 |
| 7 | f3_parkinson | 0.000016 |
| 8 | f1_log_ret | 0.000011 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2025-09-15.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=26 | 0.002777 |
| 2 | t=28 | 0.001521 |
| 3 | t=11 | -0.001432 |
| 4 | t=27 | 0.001235 |
| 5 | t=25 | 0.001013 |

> **Recency concentration**: 11.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2025-09-15.png)

---

#### 2021-03-05 — score: 0.138675

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000004 |
| 2 | f6_rsi_14 | 0.000003 |
| 3 | f1_log_ret | 0.000002 |
| 4 | f4_vol_z | 0.000002 |
| 5 | f3_parkinson | 0.000001 |
| 6 | f8_bb_width | 0.000001 |
| 7 | f5_rel_ret | 0.000001 |
| 8 | f2_gap_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2021-03-05.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=87 | 0.003292 |
| 2 | t=89 | 0.002723 |
| 3 | t=33 | -0.001152 |
| 4 | t=86 | 0.000967 |
| 5 | t=22 | -0.000941 |

> **Recency concentration**: 47.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2021-03-05.png)

---

#### 2021-10-25 — score: 0.109432

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000005 |
| 2 | f8_bb_width | 0.000003 |
| 3 | f6_rsi_14 | 0.000003 |
| 4 | f4_vol_z | 0.000002 |
| 5 | f1_log_ret | 0.000001 |
| 6 | f2_gap_ret | 0.000001 |
| 7 | f3_parkinson | 0.000001 |
| 8 | f5_rel_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2021-10-25.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=10 | 0.000090 |
| 2 | t=11 | 0.000051 |
| 3 | t=9 | 0.000039 |
| 4 | t=70 | 0.000033 |
| 5 | t=8 | 0.000032 |

> **Recency concentration**: 9.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2021-10-25.png)

---

#### 2024-07-02 — score: 0.107047

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f4_vol_z | 0.000000 |
| 2 | f7_macd_hist | 0.000000 |
| 3 | f1_log_ret | 0.000000 |
| 4 | f3_parkinson | 0.000000 |
| 5 | f8_bb_width | 0.000000 |
| 6 | f6_rsi_14 | 0.000000 |
| 7 | f2_gap_ret | 0.000000 |
| 8 | f5_rel_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2024-07-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=49 | -0.000133 |
| 2 | t=95 | 0.000075 |
| 3 | t=20 | 0.000053 |
| 4 | t=47 | -0.000048 |
| 5 | t=48 | 0.000036 |

> **Recency concentration**: 15.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2024-07-02.png)

---

#### 2023-10-19 — score: 0.086428

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000005 |
| 2 | f6_rsi_14 | 0.000002 |
| 3 | f4_vol_z | 0.000002 |
| 4 | f8_bb_width | 0.000001 |
| 5 | f3_parkinson | 0.000001 |
| 6 | f2_gap_ret | 0.000001 |
| 7 | f1_log_ret | 0.000001 |
| 8 | f5_rel_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2023-10-19.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=31 | 0.000091 |
| 2 | t=95 | 0.000067 |
| 3 | t=33 | 0.000059 |
| 4 | t=4 | -0.000059 |
| 5 | t=45 | 0.000058 |

> **Recency concentration**: 8.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2023-10-19.png)

---

#### 2015-08-24 — score: 0.061941

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.000004 |
| 2 | f6_rsi_14 | 0.000003 |
| 3 | f8_bb_width | 0.000003 |
| 4 | f5_rel_ret | 0.000002 |
| 5 | f4_vol_z | 0.000002 |
| 6 | f7_macd_hist | 0.000002 |
| 7 | f3_parkinson | 0.000001 |
| 8 | f1_log_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2015-08-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=83 | 0.000416 |
| 2 | t=63 | 0.000088 |
| 3 | t=15 | -0.000086 |
| 4 | t=84 | 0.000062 |
| 5 | t=82 | -0.000017 |

> **Recency concentration**: 57.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2015-08-24.png)

---

### INTC

#### 2024-08-02 — score: 1.002813

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f6_rsi_14 | 0.000001 |
| 2 | f3_parkinson | 0.000000 |
| 3 | f7_macd_hist | 0.000000 |
| 4 | f4_vol_z | 0.000000 |
| 5 | f5_rel_ret | 0.000000 |
| 6 | f1_log_ret | 0.000000 |
| 7 | f8_bb_width | 0.000000 |
| 8 | f2_gap_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2024-08-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -0.000668 |
| 2 | t=28 | 0.000482 |
| 3 | t=83 | -0.000277 |
| 4 | t=80 | -0.000214 |
| 5 | t=31 | 0.000150 |

> **Recency concentration**: 48.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2024-08-02.png)

---

#### 2020-03-16 — score: 0.506368

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f6_rsi_14 | 0.000012 |
| 2 | f4_vol_z | 0.000011 |
| 3 | f7_macd_hist | 0.000009 |
| 4 | f1_log_ret | 0.000008 |
| 5 | f8_bb_width | 0.000005 |
| 6 | f2_gap_ret | 0.000004 |
| 7 | f3_parkinson | 0.000004 |
| 8 | f5_rel_ret | 0.000004 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2020-03-16.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=89 | 0.000239 |
| 2 | t=90 | 0.000212 |
| 3 | t=95 | 0.000169 |
| 4 | t=86 | -0.000147 |
| 5 | t=38 | 0.000108 |

> **Recency concentration**: 57.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2020-03-16.png)

---

#### 2020-07-24 — score: 0.282781

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000082 |
| 2 | f3_parkinson | 0.000079 |
| 3 | f6_rsi_14 | 0.000075 |
| 4 | f7_macd_hist | 0.000053 |
| 5 | f2_gap_ret | 0.000041 |
| 6 | f5_rel_ret | 0.000041 |
| 7 | f1_log_ret | 0.000038 |
| 8 | f4_vol_z | 0.000017 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2020-07-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=4 | 0.000877 |
| 2 | t=2 | 0.000682 |
| 3 | t=95 | -0.000587 |
| 4 | t=1 | 0.000346 |
| 5 | t=5 | 0.000295 |

> **Recency concentration**: 16.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2020-07-24.png)

---

#### 2024-04-26 — score: 0.213286

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000022 |
| 2 | f7_macd_hist | 0.000015 |
| 3 | f2_gap_ret | 0.000014 |
| 4 | f1_log_ret | 0.000013 |
| 5 | f5_rel_ret | 0.000012 |
| 6 | f4_vol_z | 0.000011 |
| 7 | f6_rsi_14 | 0.000011 |
| 8 | f8_bb_width | 0.000007 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2024-04-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=58 | -0.000295 |
| 2 | t=61 | -0.000236 |
| 3 | t=78 | 0.000175 |
| 4 | t=59 | -0.000139 |
| 5 | t=11 | 0.000104 |

> **Recency concentration**: 19.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2024-04-26.png)

---

#### 2025-04-09 — score: 0.143298

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000004 |
| 2 | f6_rsi_14 | 0.000003 |
| 3 | f7_macd_hist | 0.000003 |
| 4 | f8_bb_width | 0.000003 |
| 5 | f1_log_ret | 0.000003 |
| 6 | f5_rel_ret | 0.000002 |
| 7 | f4_vol_z | 0.000002 |
| 8 | f2_gap_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2025-04-09.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=59 | -0.000181 |
| 2 | t=78 | 0.000136 |
| 3 | t=64 | -0.000125 |
| 4 | t=63 | 0.000093 |
| 5 | t=60 | -0.000082 |

> **Recency concentration**: 25.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2025-04-09.png)

---

#### 2023-01-27 — score: 0.131965

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f1_log_ret | 0.000018 |
| 2 | f7_macd_hist | 0.000011 |
| 3 | f2_gap_ret | 0.000009 |
| 4 | f6_rsi_14 | 0.000009 |
| 5 | f8_bb_width | 0.000009 |
| 6 | f3_parkinson | 0.000006 |
| 7 | f5_rel_ret | 0.000005 |
| 8 | f4_vol_z | 0.000005 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2023-01-27.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=34 | 0.000483 |
| 2 | t=95 | 0.000260 |
| 3 | t=33 | -0.000194 |
| 4 | t=32 | 0.000151 |
| 5 | t=36 | 0.000148 |

> **Recency concentration**: 17.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2023-01-27.png)

---

#### 2020-10-23 — score: 0.122931

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f1_log_ret | 0.000059 |
| 2 | f7_macd_hist | 0.000051 |
| 3 | f6_rsi_14 | 0.000040 |
| 4 | f3_parkinson | 0.000036 |
| 5 | f5_rel_ret | 0.000029 |
| 6 | f2_gap_ret | 0.000025 |
| 7 | f4_vol_z | 0.000021 |
| 8 | f8_bb_width | 0.000016 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2020-10-23.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=30 | -0.001029 |
| 2 | t=31 | 0.000853 |
| 3 | t=32 | 0.000410 |
| 4 | t=29 | -0.000408 |
| 5 | t=27 | -0.000386 |

> **Recency concentration**: 12.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2020-10-23.png)

---

#### 2022-10-28 — score: 0.121389

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000001 |
| 2 | f4_vol_z | 0.000001 |
| 3 | f6_rsi_14 | 0.000000 |
| 4 | f7_macd_hist | 0.000000 |
| 5 | f5_rel_ret | 0.000000 |
| 6 | f3_parkinson | 0.000000 |
| 7 | f2_gap_ret | 0.000000 |
| 8 | f1_log_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2022-10-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=31 | -0.000091 |
| 2 | t=80 | 0.000026 |
| 3 | t=95 | 0.000019 |
| 4 | t=62 | 0.000014 |
| 5 | t=76 | 0.000013 |

> **Recency concentration**: 28.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2022-10-28.png)

---

#### 2016-01-15 — score: 0.109946

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000001 |
| 2 | f1_log_ret | 0.000001 |
| 3 | f6_rsi_14 | 0.000001 |
| 4 | f4_vol_z | 0.000001 |
| 5 | f5_rel_ret | 0.000001 |
| 6 | f3_parkinson | 0.000001 |
| 7 | f8_bb_width | 0.000000 |
| 8 | f2_gap_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2016-01-15.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=89 | 0.000103 |
| 2 | t=31 | -0.000037 |
| 3 | t=27 | -0.000036 |
| 4 | t=93 | 0.000027 |
| 5 | t=24 | -0.000023 |

> **Recency concentration**: 37.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2016-01-15.png)

---

#### 2018-07-27 — score: 0.095604

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000002 |
| 2 | f8_bb_width | 0.000002 |
| 3 | f3_parkinson | 0.000002 |
| 4 | f1_log_ret | 0.000002 |
| 5 | f2_gap_ret | 0.000002 |
| 6 | f6_rsi_14 | 0.000001 |
| 7 | f4_vol_z | 0.000001 |
| 8 | f5_rel_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2018-07-27.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=13 | 0.000098 |
| 2 | t=0 | 0.000058 |
| 3 | t=9 | -0.000046 |
| 4 | t=67 | 0.000041 |
| 5 | t=1 | 0.000040 |

> **Recency concentration**: 13.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2018-07-27.png)

---

## TimesNet + TranAD (`TranAD`)

### NVDA

#### 2025-10-29 — score: 5.019600

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.353328 |
| 2 | f1_log_ret | 0.163472 |
| 3 | f2_gap_ret | 0.065317 |
| 4 | f4_vol_z | 0.027673 |
| 5 | f8_bb_width | 0.026549 |
| 6 | f3_parkinson | 0.016423 |
| 7 | f5_rel_ret | 0.016152 |
| 8 | f6_rsi_14 | 0.013159 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-10-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 24.282692 |
| 2 | t=94 | -7.206090 |
| 3 | t=93 | -1.931398 |
| 4 | t=57 | -1.018779 |
| 5 | t=20 | -0.871078 |

> **Recency concentration**: 78.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-10-29.png)

---

#### 2025-10-28 — score: 4.281147

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.222802 |
| 2 | f1_log_ret | 0.115291 |
| 3 | f8_bb_width | 0.021016 |
| 4 | f2_gap_ret | 0.016006 |
| 5 | f3_parkinson | 0.012945 |
| 6 | f4_vol_z | 0.012571 |
| 7 | f5_rel_ret | 0.011252 |
| 8 | f6_rsi_14 | 0.010428 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-10-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 13.379853 |
| 2 | t=94 | -2.624677 |
| 3 | t=93 | -1.017916 |
| 4 | t=69 | -0.757410 |
| 5 | t=91 | 0.688854 |

> **Recency concentration**: 73.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-10-28.png)

---

#### 2025-03-14 — score: 2.795005

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.134217 |
| 2 | f1_log_ret | 0.107521 |
| 3 | f6_rsi_14 | 0.019485 |
| 4 | f8_bb_width | 0.014815 |
| 5 | f5_rel_ret | 0.012549 |
| 6 | f3_parkinson | 0.009474 |
| 7 | f2_gap_ret | 0.007578 |
| 8 | f4_vol_z | 0.005996 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-03-14.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=91 | 1.580719 |
| 2 | t=94 | 1.129835 |
| 3 | t=95 | 1.095914 |
| 4 | t=92 | 0.994036 |
| 5 | t=93 | -0.977588 |

> **Recency concentration**: 66.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-03-14.png)

---

#### 2025-03-03 — score: 2.778372

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.229269 |
| 2 | f1_log_ret | 0.141227 |
| 3 | f8_bb_width | 0.026919 |
| 4 | f5_rel_ret | 0.020451 |
| 5 | f6_rsi_14 | 0.016041 |
| 6 | f3_parkinson | 0.012754 |
| 7 | f4_vol_z | 0.009907 |
| 8 | f2_gap_ret | 0.009015 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-03-03.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 6.785882 |
| 2 | t=93 | -4.718484 |
| 3 | t=92 | 1.704904 |
| 4 | t=71 | 1.557891 |
| 5 | t=87 | -0.287673 |

> **Recency concentration**: 71.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-03-03.png)

---

#### 2023-05-25 — score: 2.666274

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.063407 |
| 2 | f2_gap_ret | 0.034883 |
| 3 | f1_log_ret | 0.007726 |
| 4 | f5_rel_ret | 0.006027 |
| 5 | f4_vol_z | 0.005434 |
| 6 | f6_rsi_14 | 0.002540 |
| 7 | f8_bb_width | 0.001585 |
| 8 | f3_parkinson | 0.001504 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2023-05-25.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 6.973780 |
| 2 | t=31 | -0.727459 |
| 3 | t=9 | -0.190931 |
| 4 | t=12 | -0.146256 |
| 5 | t=13 | -0.136719 |

> **Recency concentration**: 66.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2023-05-25.png)

---

#### 2025-04-09 — score: 2.644044

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.185382 |
| 2 | f1_log_ret | 0.137509 |
| 3 | f6_rsi_14 | 0.084414 |
| 4 | f3_parkinson | 0.062410 |
| 5 | f2_gap_ret | 0.046516 |
| 6 | f8_bb_width | 0.035265 |
| 7 | f5_rel_ret | 0.029143 |
| 8 | f4_vol_z | 0.025533 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-04-09.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -5.553452 |
| 2 | t=94 | 4.344589 |
| 3 | t=92 | 3.595266 |
| 4 | t=91 | 1.231273 |
| 5 | t=93 | 0.423592 |

> **Recency concentration**: 72.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-04-09.png)

---

#### 2016-11-11 — score: 2.630502

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.018328 |
| 2 | f2_gap_ret | 0.017913 |
| 3 | f1_log_ret | 0.017581 |
| 4 | f3_parkinson | 0.013984 |
| 5 | f4_vol_z | 0.007806 |
| 6 | f8_bb_width | 0.000885 |
| 7 | f7_macd_hist | 0.000736 |
| 8 | f6_rsi_14 | 0.000666 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2016-11-11.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 4.094274 |
| 2 | t=31 | -0.195035 |
| 3 | t=94 | -0.137219 |
| 4 | t=93 | -0.078005 |
| 5 | t=91 | -0.072478 |

> **Recency concentration**: 83.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2016-11-11.png)

---

#### 2025-08-20 — score: 2.376601

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.208035 |
| 2 | f1_log_ret | 0.057728 |
| 3 | f8_bb_width | 0.017269 |
| 4 | f5_rel_ret | 0.014625 |
| 5 | f6_rsi_14 | 0.014073 |
| 6 | f4_vol_z | 0.013546 |
| 7 | f3_parkinson | 0.005781 |
| 8 | f2_gap_ret | 0.003367 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-08-20.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 3.557235 |
| 2 | t=93 | -1.462607 |
| 3 | t=0 | 1.202229 |
| 4 | t=94 | 1.168538 |
| 5 | t=92 | -0.906668 |

> **Recency concentration**: 55.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-08-20.png)

---

#### 2025-02-06 — score: 2.333233

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.193439 |
| 2 | f1_log_ret | 0.082138 |
| 3 | f8_bb_width | 0.021018 |
| 4 | f5_rel_ret | 0.012632 |
| 5 | f6_rsi_14 | 0.012268 |
| 6 | f3_parkinson | 0.008542 |
| 7 | f2_gap_ret | 0.007846 |
| 8 | f4_vol_z | 0.004493 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-02-06.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -6.041872 |
| 2 | t=93 | 3.413266 |
| 3 | t=92 | 2.217093 |
| 4 | t=94 | 2.133502 |
| 5 | t=91 | 0.738191 |

> **Recency concentration**: 81.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-02-06.png)

---

#### 2025-03-12 — score: 2.182489

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.226899 |
| 2 | f1_log_ret | 0.084933 |
| 3 | f8_bb_width | 0.036125 |
| 4 | f6_rsi_14 | 0.019786 |
| 5 | f5_rel_ret | 0.011902 |
| 6 | f4_vol_z | 0.005595 |
| 7 | f3_parkinson | 0.005531 |
| 8 | f2_gap_ret | 0.005452 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-03-12.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -6.269232 |
| 2 | t=93 | 4.941289 |
| 3 | t=94 | 2.725654 |
| 4 | t=91 | 1.026373 |
| 5 | t=64 | -1.026363 |

> **Recency concentration**: 76.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-03-12.png)

---

### TSLA

#### 2021-11-02 — score: 2.440174

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.169027 |
| 2 | f2_gap_ret | 0.031253 |
| 3 | f1_log_ret | 0.013853 |
| 4 | f5_rel_ret | 0.010316 |
| 5 | f8_bb_width | 0.007583 |
| 6 | f6_rsi_14 | 0.007459 |
| 7 | f3_parkinson | 0.005820 |
| 8 | f4_vol_z | 0.002395 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-11-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 14.800621 |
| 2 | t=94 | -2.489556 |
| 3 | t=92 | -1.869116 |
| 4 | t=89 | -1.697051 |
| 5 | t=91 | -1.102046 |

> **Recency concentration**: 80.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-11-02.png)

---

#### 2021-11-01 — score: 2.310388

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.163003 |
| 2 | f6_rsi_14 | 0.015941 |
| 3 | f8_bb_width | 0.012509 |
| 4 | f1_log_ret | 0.011573 |
| 5 | f5_rel_ret | 0.010373 |
| 6 | f4_vol_z | 0.006422 |
| 7 | f2_gap_ret | 0.004035 |
| 8 | f3_parkinson | 0.002035 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-11-01.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 10.256836 |
| 2 | t=93 | -2.308222 |
| 3 | t=90 | -1.386247 |
| 4 | t=92 | -1.204778 |
| 5 | t=94 | 1.150976 |

> **Recency concentration**: 83.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-11-01.png)

---

#### 2025-03-10 — score: 2.041117

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.139938 |
| 2 | f1_log_ret | 0.088884 |
| 3 | f3_parkinson | 0.025089 |
| 4 | f8_bb_width | 0.023686 |
| 5 | f5_rel_ret | 0.017934 |
| 6 | f6_rsi_14 | 0.015997 |
| 7 | f4_vol_z | 0.007097 |
| 8 | f2_gap_ret | 0.004628 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-03-10.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=93 | 1.523602 |
| 2 | t=94 | 1.456249 |
| 3 | t=95 | -0.511730 |
| 4 | t=91 | 0.505005 |
| 5 | t=4 | -0.245492 |

> **Recency concentration**: 69.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-03-10.png)

---

#### 2021-10-25 — score: 1.693685

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.078087 |
| 2 | f8_bb_width | 0.019499 |
| 3 | f5_rel_ret | 0.019111 |
| 4 | f3_parkinson | 0.018394 |
| 5 | f1_log_ret | 0.011383 |
| 6 | f6_rsi_14 | 0.010188 |
| 7 | f2_gap_ret | 0.010112 |
| 8 | f4_vol_z | 0.008078 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-10-25.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 7.999295 |
| 2 | t=93 | -1.052345 |
| 3 | t=94 | -0.736187 |
| 4 | t=92 | -0.272259 |
| 5 | t=46 | -0.230038 |

> **Recency concentration**: 73.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-10-25.png)

---

#### 2021-10-28 — score: 1.582624

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.105231 |
| 2 | f8_bb_width | 0.008425 |
| 3 | f1_log_ret | 0.007080 |
| 4 | f6_rsi_14 | 0.006891 |
| 5 | f2_gap_ret | 0.006264 |
| 6 | f3_parkinson | 0.003846 |
| 7 | f5_rel_ret | 0.001237 |
| 8 | f4_vol_z | 0.000790 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-10-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 7.955911 |
| 2 | t=92 | -1.660498 |
| 3 | t=94 | -1.292755 |
| 4 | t=93 | -1.000181 |
| 5 | t=91 | -0.221680 |

> **Recency concentration**: 83.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-10-28.png)

---

#### 2025-03-03 — score: 1.136966

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.093267 |
| 2 | f1_log_ret | 0.041527 |
| 3 | f8_bb_width | 0.011736 |
| 4 | f6_rsi_14 | 0.009447 |
| 5 | f3_parkinson | 0.009422 |
| 6 | f5_rel_ret | 0.006730 |
| 7 | f4_vol_z | 0.005812 |
| 8 | f2_gap_ret | 0.005608 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-03-03.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -2.273489 |
| 2 | t=92 | 1.068597 |
| 3 | t=93 | 0.884757 |
| 4 | t=94 | 0.800622 |
| 5 | t=91 | 0.618708 |

> **Recency concentration**: 75.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-03-03.png)

---

#### 2025-01-02 — score: 1.125827

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.120753 |
| 2 | f1_log_ret | 0.031862 |
| 3 | f8_bb_width | 0.007496 |
| 4 | f6_rsi_14 | 0.005704 |
| 5 | f3_parkinson | 0.004863 |
| 6 | f4_vol_z | 0.003494 |
| 7 | f5_rel_ret | 0.002885 |
| 8 | f2_gap_ret | 0.002576 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-01-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 4.285380 |
| 2 | t=94 | -0.987859 |
| 3 | t=93 | -0.588577 |
| 4 | t=92 | -0.499422 |
| 5 | t=91 | -0.271021 |

> **Recency concentration**: 77.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-01-02.png)

---

#### 2021-10-29 — score: 1.093910

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.096284 |
| 2 | f8_bb_width | 0.007352 |
| 3 | f6_rsi_14 | 0.006745 |
| 4 | f1_log_ret | 0.005711 |
| 5 | f3_parkinson | 0.002714 |
| 6 | f5_rel_ret | 0.001161 |
| 7 | f2_gap_ret | 0.000844 |
| 8 | f4_vol_z | 0.000696 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-10-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 4.145498 |
| 2 | t=93 | -0.822392 |
| 3 | t=91 | -0.384944 |
| 4 | t=92 | -0.303079 |
| 5 | t=32 | -0.154198 |

> **Recency concentration**: 77.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-10-29.png)

---

#### 2020-02-05 — score: 1.052270

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.011411 |
| 2 | f3_parkinson | 0.008240 |
| 3 | f1_log_ret | 0.007701 |
| 4 | f8_bb_width | 0.007004 |
| 5 | f5_rel_ret | 0.006516 |
| 6 | f6_rsi_14 | 0.002554 |
| 7 | f2_gap_ret | 0.001758 |
| 8 | f4_vol_z | 0.001316 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2020-02-05.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.588465 |
| 2 | t=25 | -0.078968 |
| 3 | t=93 | 0.067228 |
| 4 | t=26 | -0.057847 |
| 5 | t=3 | -0.047441 |

> **Recency concentration**: 79.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2020-02-05.png)

---

#### 2025-03-11 — score: 0.971735

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.083013 |
| 2 | f1_log_ret | 0.048972 |
| 3 | f8_bb_width | 0.020216 |
| 4 | f6_rsi_14 | 0.012541 |
| 5 | f3_parkinson | 0.007198 |
| 6 | f5_rel_ret | 0.004798 |
| 7 | f2_gap_ret | 0.003098 |
| 8 | f4_vol_z | 0.002812 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-03-11.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 0.931766 |
| 2 | t=3 | -0.726041 |
| 3 | t=12 | -0.328740 |
| 4 | t=92 | -0.306125 |
| 5 | t=0 | 0.300489 |

> **Recency concentration**: 37.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-03-11.png)

---

### INTC

#### 2025-09-18 — score: 2.055028

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.046327 |
| 2 | f5_rel_ret | 0.021730 |
| 3 | f1_log_ret | 0.018964 |
| 4 | f4_vol_z | 0.016161 |
| 5 | f3_parkinson | 0.015273 |
| 6 | f8_bb_width | 0.007979 |
| 7 | f7_macd_hist | 0.003945 |
| 8 | f6_rsi_14 | 0.003508 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2025-09-18.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 5.005689 |
| 2 | t=57 | -0.556897 |
| 3 | t=74 | -0.301557 |
| 4 | t=72 | -0.169061 |
| 5 | t=26 | -0.152888 |

> **Recency concentration**: 63.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2025-09-18.png)

---

#### 2024-08-02 — score: 1.627456

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.040945 |
| 2 | f4_vol_z | 0.019130 |
| 3 | f2_gap_ret | 0.016252 |
| 4 | f3_parkinson | 0.014171 |
| 5 | f1_log_ret | 0.011141 |
| 6 | f7_macd_hist | 0.009095 |
| 7 | f8_bb_width | 0.002764 |
| 8 | f6_rsi_14 | 0.002746 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2024-08-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 5.396450 |
| 2 | t=28 | -0.721494 |
| 3 | t=11 | -0.387175 |
| 4 | t=76 | -0.327474 |
| 5 | t=83 | -0.170898 |

> **Recency concentration**: 69.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2024-08-02.png)

---

#### 2020-07-24 — score: 1.237404

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.018882 |
| 2 | f4_vol_z | 0.016985 |
| 3 | f2_gap_ret | 0.013175 |
| 4 | f1_log_ret | 0.010542 |
| 5 | f3_parkinson | 0.005856 |
| 6 | f6_rsi_14 | 0.003251 |
| 7 | f8_bb_width | 0.002320 |
| 8 | f7_macd_hist | 0.001419 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2020-07-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 2.270994 |
| 2 | t=71 | -0.131697 |
| 3 | t=9 | -0.083495 |
| 4 | t=93 | 0.079062 |
| 5 | t=32 | -0.070944 |

> **Recency concentration**: 70.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2020-07-24.png)

---

#### 2021-10-22 — score: 0.649180

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.112257 |
| 2 | f3_parkinson | 0.032027 |
| 3 | f2_gap_ret | 0.022996 |
| 4 | f7_macd_hist | 0.022937 |
| 5 | f4_vol_z | 0.020181 |
| 6 | f5_rel_ret | 0.014303 |
| 7 | f6_rsi_14 | 0.012937 |
| 8 | f1_log_ret | 0.012852 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2021-10-22.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.542382 |
| 2 | t=7 | -0.120167 |
| 3 | t=89 | 0.119019 |
| 4 | t=73 | -0.090854 |
| 5 | t=50 | 0.090715 |

> **Recency concentration**: 47.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2021-10-22.png)

---

#### 2022-07-29 — score: 0.603087

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.013424 |
| 2 | f5_rel_ret | 0.012662 |
| 3 | f1_log_ret | 0.007414 |
| 4 | f3_parkinson | 0.005817 |
| 5 | f4_vol_z | 0.005801 |
| 6 | f8_bb_width | 0.004377 |
| 7 | f7_macd_hist | 0.004261 |
| 8 | f6_rsi_14 | 0.002959 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2022-07-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.423894 |
| 2 | t=33 | -0.108044 |
| 3 | t=94 | -0.081415 |
| 4 | t=69 | -0.071545 |
| 5 | t=60 | -0.070428 |

> **Recency concentration**: 57.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2022-07-29.png)

---

#### 2020-10-23 — score: 0.499188

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f4_vol_z | 0.007534 |
| 2 | f5_rel_ret | 0.006437 |
| 3 | f2_gap_ret | 0.003850 |
| 4 | f6_rsi_14 | 0.001658 |
| 5 | f1_log_ret | 0.001586 |
| 6 | f3_parkinson | 0.001563 |
| 7 | f7_macd_hist | 0.001402 |
| 8 | f8_bb_width | 0.000623 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2020-10-23.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.604186 |
| 2 | t=31 | -0.554236 |
| 3 | t=94 | -0.118056 |
| 4 | t=7 | -0.082002 |
| 5 | t=93 | 0.035809 |

> **Recency concentration**: 62.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2020-10-23.png)

---

#### 2025-02-19 — score: 0.465948

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.006633 |
| 2 | f7_macd_hist | 0.005174 |
| 3 | f5_rel_ret | 0.005169 |
| 4 | f2_gap_ret | 0.004992 |
| 5 | f1_log_ret | 0.004384 |
| 6 | f6_rsi_14 | 0.003163 |
| 7 | f3_parkinson | 0.002162 |
| 8 | f4_vol_z | 0.001146 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2025-02-19.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 0.840658 |
| 2 | t=94 | 0.198538 |
| 3 | t=23 | -0.052831 |
| 4 | t=74 | -0.040295 |
| 5 | t=92 | 0.031454 |

> **Recency concentration**: 68.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2025-02-19.png)

---

#### 2019-04-26 — score: 0.438478

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.006805 |
| 2 | f4_vol_z | 0.005559 |
| 3 | f2_gap_ret | 0.002974 |
| 4 | f1_log_ret | 0.002514 |
| 5 | f6_rsi_14 | 0.002301 |
| 6 | f3_parkinson | 0.001829 |
| 7 | f7_macd_hist | 0.001207 |
| 8 | f8_bb_width | 0.000812 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2019-04-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 0.882903 |
| 2 | t=32 | -0.113832 |
| 3 | t=31 | -0.093975 |
| 4 | t=89 | -0.049191 |
| 5 | t=66 | -0.046046 |

> **Recency concentration**: 66.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2019-04-26.png)

---

#### 2024-01-26 — score: 0.343203

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.008065 |
| 2 | f5_rel_ret | 0.005869 |
| 3 | f4_vol_z | 0.005174 |
| 4 | f1_log_ret | 0.003182 |
| 5 | f3_parkinson | 0.002937 |
| 6 | f6_rsi_14 | 0.001523 |
| 7 | f8_bb_width | 0.001287 |
| 8 | f7_macd_hist | 0.000820 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2024-01-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.023175 |
| 2 | t=34 | -0.096138 |
| 3 | t=48 | -0.050682 |
| 4 | t=93 | 0.039432 |
| 5 | t=74 | -0.038742 |

> **Recency concentration**: 64.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2024-01-26.png)

---

#### 2025-02-18 — score: 0.323612

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f1_log_ret | 0.012990 |
| 2 | f5_rel_ret | 0.004647 |
| 3 | f7_macd_hist | 0.003354 |
| 4 | f8_bb_width | 0.002782 |
| 5 | f3_parkinson | 0.002716 |
| 6 | f2_gap_ret | 0.002011 |
| 7 | f4_vol_z | 0.001529 |
| 8 | f6_rsi_14 | 0.000876 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2025-02-18.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.382133 |
| 2 | t=93 | 0.095415 |
| 3 | t=24 | -0.093012 |
| 4 | t=27 | -0.087817 |
| 5 | t=75 | -0.076106 |

> **Recency concentration**: 61.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2025-02-18.png)

---

## Cross-Model Comparison

### Feature Importance

![Feature Importance Comparison](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\comparison_feature_importance.png)

### Temporal Receptive Field

![Temporal Comparison](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\comparison_temporal_importance.png)
