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
| 3 | f6_rsi_14 | 0.000036 |
| 4 | f4_vol_z | 0.000030 |
| 5 | f2_gap_ret | 0.000022 |
| 6 | f5_rel_ret | 0.000016 |
| 7 | f8_bb_width | 0.000016 |
| 8 | f3_parkinson | 0.000015 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2024-08-05.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=46 | 0.002114 |
| 2 | t=89 | 0.001703 |
| 3 | t=54 | -0.001647 |
| 4 | t=88 | 0.001209 |
| 5 | t=90 | 0.001016 |

> **Recency concentration**: 30.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2024-08-05.png)

---

#### 2025-01-27 — score: 0.510169

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000062 |
| 2 | f6_rsi_14 | 0.000017 |
| 3 | f3_parkinson | 0.000017 |
| 4 | f8_bb_width | 0.000015 |
| 5 | f4_vol_z | 0.000015 |
| 6 | f1_log_ret | 0.000010 |
| 7 | f2_gap_ret | 0.000005 |
| 8 | f5_rel_ret | 0.000005 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2025-01-27.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=82 | 0.001385 |
| 2 | t=56 | -0.001361 |
| 3 | t=3 | 0.001269 |
| 4 | t=67 | 0.000651 |
| 5 | t=85 | -0.000497 |

> **Recency concentration**: 26.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2025-01-27.png)

---

#### 2024-05-28 — score: 0.274446

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000165 |
| 2 | f6_rsi_14 | 0.000056 |
| 3 | f8_bb_width | 0.000052 |
| 4 | f2_gap_ret | 0.000037 |
| 5 | f5_rel_ret | 0.000030 |
| 6 | f4_vol_z | 0.000030 |
| 7 | f3_parkinson | 0.000023 |
| 8 | f1_log_ret | 0.000009 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2024-05-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=69 | 0.009303 |
| 2 | t=95 | 0.003307 |
| 3 | t=70 | -0.002209 |
| 4 | t=6 | 0.001847 |
| 5 | t=23 | -0.001248 |

> **Recency concentration**: 11.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2024-05-28.png)

---

#### 2020-03-16 — score: 0.247893

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000010 |
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
| 1 | t=80 | 0.000058 |
| 2 | t=95 | 0.000053 |
| 3 | t=77 | -0.000044 |
| 4 | t=90 | 0.000040 |
| 5 | t=93 | 0.000040 |

> **Recency concentration**: 69.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2020-03-16.png)

---

#### 2024-05-23 — score: 0.130792

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000141 |
| 2 | f6_rsi_14 | 0.000078 |
| 3 | f3_parkinson | 0.000071 |
| 4 | f8_bb_width | 0.000067 |
| 5 | f2_gap_ret | 0.000055 |
| 6 | f4_vol_z | 0.000054 |
| 7 | f5_rel_ret | 0.000040 |
| 8 | f1_log_ret | 0.000034 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2024-05-23.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=71 | 0.013454 |
| 2 | t=10 | 0.006258 |
| 3 | t=36 | -0.002778 |
| 4 | t=14 | -0.002503 |
| 5 | t=32 | -0.001727 |

> **Recency concentration**: 7.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2024-05-23.png)

---

#### 2015-08-07 — score: 0.083451

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000009 |
| 2 | f2_gap_ret | 0.000006 |
| 3 | f6_rsi_14 | 0.000005 |
| 4 | f5_rel_ret | 0.000004 |
| 5 | f4_vol_z | 0.000003 |
| 6 | f3_parkinson | 0.000002 |
| 7 | f7_macd_hist | 0.000001 |
| 8 | f1_log_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2015-08-07.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=32 | 0.000370 |
| 2 | t=33 | 0.000111 |
| 3 | t=10 | -0.000036 |
| 4 | t=31 | -0.000035 |
| 5 | t=93 | -0.000021 |

> **Recency concentration**: 11.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2015-08-07.png)

---

#### 2022-11-10 — score: 0.065259

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000005 |
| 2 | f2_gap_ret | 0.000002 |
| 3 | f4_vol_z | 0.000002 |
| 4 | f8_bb_width | 0.000002 |
| 5 | f3_parkinson | 0.000001 |
| 6 | f5_rel_ret | 0.000001 |
| 7 | f6_rsi_14 | 0.000000 |
| 8 | f1_log_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2022-11-10.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=46 | 0.000251 |
| 2 | t=53 | -0.000211 |
| 3 | t=95 | 0.000164 |
| 4 | t=48 | 0.000151 |
| 5 | t=47 | -0.000124 |

> **Recency concentration**: 20.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2022-11-10.png)

---

#### 2025-10-29 — score: 0.062309

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000017 |
| 2 | f6_rsi_14 | 0.000007 |
| 3 | f3_parkinson | 0.000006 |
| 4 | f8_bb_width | 0.000005 |
| 5 | f4_vol_z | 0.000004 |
| 6 | f1_log_ret | 0.000004 |
| 7 | f2_gap_ret | 0.000004 |
| 8 | f5_rel_ret | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2025-10-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=84 | -0.001757 |
| 2 | t=57 | 0.000866 |
| 3 | t=82 | -0.000804 |
| 4 | t=55 | -0.000792 |
| 5 | t=53 | 0.000697 |

> **Recency concentration**: 40.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2025-10-29.png)

---

#### 2018-11-19 — score: 0.054306

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000023 |
| 2 | f6_rsi_14 | 0.000019 |
| 3 | f8_bb_width | 0.000012 |
| 4 | f1_log_ret | 0.000011 |
| 5 | f7_macd_hist | 0.000009 |
| 6 | f4_vol_z | 0.000007 |
| 7 | f5_rel_ret | 0.000006 |
| 8 | f2_gap_ret | 0.000006 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2018-11-19.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=82 | 0.000088 |
| 2 | t=78 | 0.000066 |
| 3 | t=77 | 0.000057 |
| 4 | t=79 | 0.000049 |
| 5 | t=76 | 0.000047 |

> **Recency concentration**: 65.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_TimeSHAP_2018-11-19.png)

---

#### 2022-01-24 — score: 0.053984

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000018 |
| 2 | f7_macd_hist | 0.000016 |
| 3 | f6_rsi_14 | 0.000011 |
| 4 | f2_gap_ret | 0.000008 |
| 5 | f3_parkinson | 0.000007 |
| 6 | f4_vol_z | 0.000007 |
| 7 | f1_log_ret | 0.000007 |
| 8 | f5_rel_ret | 0.000005 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_AT_IG_2022-01-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=37 | 0.000140 |
| 2 | t=48 | 0.000133 |
| 3 | t=46 | 0.000130 |
| 4 | t=45 | 0.000126 |
| 5 | t=69 | 0.000117 |

> **Recency concentration**: 11.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

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
| 4 | f5_rel_ret | 0.000004 |
| 5 | f3_parkinson | 0.000004 |
| 6 | f8_bb_width | 0.000003 |
| 7 | f4_vol_z | 0.000003 |
| 8 | f2_gap_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2020-03-16.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=68 | -0.000214 |
| 2 | t=37 | 0.000202 |
| 3 | t=90 | 0.000192 |
| 4 | t=70 | 0.000122 |
| 5 | t=67 | -0.000120 |

> **Recency concentration**: 42.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2020-03-16.png)

---

#### 2023-01-26 — score: 0.250867

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000007 |
| 2 | f8_bb_width | 0.000007 |
| 3 | f5_rel_ret | 0.000005 |
| 4 | f4_vol_z | 0.000004 |
| 5 | f6_rsi_14 | 0.000004 |
| 6 | f1_log_ret | 0.000003 |
| 7 | f2_gap_ret | 0.000002 |
| 8 | f3_parkinson | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2023-01-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=18 | 0.000238 |
| 2 | t=77 | -0.000116 |
| 3 | t=75 | 0.000113 |
| 4 | t=88 | 0.000112 |
| 5 | t=20 | 0.000099 |

> **Recency concentration**: 34.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2023-01-26.png)

---

#### 2025-03-10 — score: 0.244965

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000030 |
| 2 | f1_log_ret | 0.000022 |
| 3 | f8_bb_width | 0.000012 |
| 4 | f4_vol_z | 0.000011 |
| 5 | f6_rsi_14 | 0.000011 |
| 6 | f2_gap_ret | 0.000010 |
| 7 | f3_parkinson | 0.000004 |
| 8 | f5_rel_ret | 0.000003 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2025-03-10.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=46 | -0.000503 |
| 2 | t=51 | 0.000495 |
| 3 | t=95 | 0.000424 |
| 4 | t=4 | 0.000370 |
| 5 | t=77 | 0.000368 |

> **Recency concentration**: 24.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2025-03-10.png)

---

#### 2022-03-28 — score: 0.192251

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000012 |
| 2 | f2_gap_ret | 0.000008 |
| 3 | f6_rsi_14 | 0.000003 |
| 4 | f5_rel_ret | 0.000003 |
| 5 | f4_vol_z | 0.000003 |
| 6 | f8_bb_width | 0.000003 |
| 7 | f3_parkinson | 0.000002 |
| 8 | f1_log_ret | 0.000002 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2022-03-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=73 | -0.001287 |
| 2 | t=51 | 0.000746 |
| 3 | t=37 | 0.000660 |
| 4 | t=0 | 0.000528 |
| 5 | t=75 | -0.000456 |

> **Recency concentration**: 8.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2022-03-28.png)

---

#### 2025-09-15 — score: 0.184143

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000099 |
| 2 | f4_vol_z | 0.000035 |
| 3 | f5_rel_ret | 0.000028 |
| 4 | f8_bb_width | 0.000027 |
| 5 | f6_rsi_14 | 0.000023 |
| 6 | f2_gap_ret | 0.000022 |
| 7 | f3_parkinson | 0.000015 |
| 8 | f1_log_ret | 0.000011 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2025-09-15.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=26 | 0.002641 |
| 2 | t=28 | 0.002096 |
| 3 | t=27 | 0.001441 |
| 4 | t=25 | 0.001194 |
| 5 | t=95 | 0.001109 |

> **Recency concentration**: 14.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

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
| 1 | t=87 | 0.004495 |
| 2 | t=22 | -0.001285 |
| 3 | t=89 | 0.001018 |
| 4 | t=95 | -0.000768 |
| 5 | t=65 | 0.000760 |

> **Recency concentration**: 50.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2021-03-05.png)

---

#### 2021-10-25 — score: 0.109432

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000005 |
| 2 | f8_bb_width | 0.000005 |
| 3 | f6_rsi_14 | 0.000002 |
| 4 | f4_vol_z | 0.000002 |
| 5 | f3_parkinson | 0.000001 |
| 6 | f1_log_ret | 0.000001 |
| 7 | f2_gap_ret | 0.000001 |
| 8 | f5_rel_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2021-10-25.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=10 | 0.000093 |
| 2 | t=9 | 0.000061 |
| 3 | t=12 | 0.000032 |
| 4 | t=13 | 0.000028 |
| 5 | t=70 | 0.000028 |

> **Recency concentration**: 17.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2021-10-25.png)

---

#### 2024-07-02 — score: 0.107047

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000000 |
| 2 | f4_vol_z | 0.000000 |
| 3 | f8_bb_width | 0.000000 |
| 4 | f7_macd_hist | 0.000000 |
| 5 | f1_log_ret | 0.000000 |
| 6 | f6_rsi_14 | 0.000000 |
| 7 | f2_gap_ret | 0.000000 |
| 8 | f5_rel_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2024-07-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=24 | 0.000120 |
| 2 | t=95 | 0.000110 |
| 3 | t=56 | -0.000109 |
| 4 | t=52 | -0.000097 |
| 5 | t=49 | -0.000068 |

> **Recency concentration**: 19.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

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
| 1 | t=95 | 0.000108 |
| 2 | t=10 | -0.000102 |
| 3 | t=14 | -0.000073 |
| 4 | t=39 | 0.000065 |
| 5 | t=36 | -0.000056 |

> **Recency concentration**: 17.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2023-10-19.png)

---

#### 2015-08-24 — score: 0.061941

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000005 |
| 2 | f2_gap_ret | 0.000004 |
| 3 | f6_rsi_14 | 0.000003 |
| 4 | f5_rel_ret | 0.000002 |
| 5 | f4_vol_z | 0.000002 |
| 6 | f7_macd_hist | 0.000002 |
| 7 | f3_parkinson | 0.000001 |
| 8 | f1_log_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_IG_2015-08-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=83 | 0.000280 |
| 2 | t=84 | 0.000077 |
| 3 | t=10 | 0.000068 |
| 4 | t=26 | 0.000057 |
| 5 | t=49 | -0.000053 |

> **Recency concentration**: 42.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_AT_TimeSHAP_2015-08-24.png)

---

### INTC

#### 2024-08-02 — score: 1.002813

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f6_rsi_14 | 0.000001 |
| 2 | f7_macd_hist | 0.000000 |
| 3 | f4_vol_z | 0.000000 |
| 4 | f3_parkinson | 0.000000 |
| 5 | f5_rel_ret | 0.000000 |
| 6 | f1_log_ret | 0.000000 |
| 7 | f2_gap_ret | 0.000000 |
| 8 | f8_bb_width | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2024-08-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=28 | 0.000378 |
| 2 | t=95 | -0.000349 |
| 3 | t=79 | -0.000202 |
| 4 | t=31 | 0.000182 |
| 5 | t=80 | -0.000167 |

> **Recency concentration**: 47.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

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
| 5 | f3_parkinson | 0.000005 |
| 6 | f8_bb_width | 0.000005 |
| 7 | f2_gap_ret | 0.000004 |
| 8 | f5_rel_ret | 0.000004 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2020-03-16.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=60 | 0.000213 |
| 2 | t=61 | -0.000123 |
| 3 | t=89 | 0.000123 |
| 4 | t=94 | 0.000105 |
| 5 | t=95 | 0.000098 |

> **Recency concentration**: 44.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2020-03-16.png)

---

#### 2020-07-24 — score: 0.282781

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000075 |
| 2 | f6_rsi_14 | 0.000073 |
| 3 | f8_bb_width | 0.000072 |
| 4 | f7_macd_hist | 0.000053 |
| 5 | f2_gap_ret | 0.000041 |
| 6 | f5_rel_ret | 0.000041 |
| 7 | f1_log_ret | 0.000038 |
| 8 | f4_vol_z | 0.000017 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2020-07-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=4 | 0.000853 |
| 2 | t=2 | 0.000633 |
| 3 | t=95 | -0.000591 |
| 4 | t=9 | -0.000476 |
| 5 | t=1 | 0.000470 |

> **Recency concentration**: 13.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2020-07-24.png)

---

#### 2024-04-26 — score: 0.213286

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000019 |
| 2 | f7_macd_hist | 0.000015 |
| 3 | f2_gap_ret | 0.000014 |
| 4 | f1_log_ret | 0.000013 |
| 5 | f5_rel_ret | 0.000012 |
| 6 | f6_rsi_14 | 0.000011 |
| 7 | f4_vol_z | 0.000011 |
| 8 | f8_bb_width | 0.000009 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2024-04-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=58 | -0.000148 |
| 2 | t=34 | 0.000100 |
| 3 | t=57 | -0.000092 |
| 4 | t=78 | 0.000087 |
| 5 | t=61 | -0.000083 |

> **Recency concentration**: 23.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2024-04-26.png)

---

#### 2025-04-09 — score: 0.143298

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f3_parkinson | 0.000004 |
| 2 | f6_rsi_14 | 0.000003 |
| 3 | f7_macd_hist | 0.000003 |
| 4 | f1_log_ret | 0.000003 |
| 5 | f8_bb_width | 0.000002 |
| 6 | f5_rel_ret | 0.000002 |
| 7 | f4_vol_z | 0.000002 |
| 8 | f2_gap_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2025-04-09.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=78 | 0.000133 |
| 2 | t=59 | -0.000092 |
| 3 | t=64 | -0.000087 |
| 4 | t=76 | 0.000086 |
| 5 | t=60 | -0.000071 |

> **Recency concentration**: 30.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

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
| 5 | f5_rel_ret | 0.000006 |
| 6 | f4_vol_z | 0.000005 |
| 7 | f8_bb_width | 0.000005 |
| 8 | f3_parkinson | 0.000004 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2023-01-27.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 0.000407 |
| 2 | t=34 | 0.000326 |
| 3 | t=33 | -0.000083 |
| 4 | t=36 | 0.000080 |
| 5 | t=44 | 0.000080 |

> **Recency concentration**: 30.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2023-01-27.png)

---

#### 2020-10-23 — score: 0.122931

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f1_log_ret | 0.000059 |
| 2 | f7_macd_hist | 0.000051 |
| 3 | f6_rsi_14 | 0.000040 |
| 4 | f5_rel_ret | 0.000029 |
| 5 | f3_parkinson | 0.000029 |
| 6 | f2_gap_ret | 0.000026 |
| 7 | f4_vol_z | 0.000023 |
| 8 | f8_bb_width | 0.000017 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2020-10-23.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=31 | 0.001009 |
| 2 | t=30 | -0.000918 |
| 3 | t=32 | 0.000415 |
| 4 | t=27 | -0.000363 |
| 5 | t=28 | -0.000295 |

> **Recency concentration**: 13.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2020-10-23.png)

---

#### 2022-10-28 — score: 0.121389

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f4_vol_z | 0.000001 |
| 2 | f8_bb_width | 0.000000 |
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
| 1 | t=31 | -0.000040 |
| 2 | t=62 | 0.000034 |
| 3 | t=95 | 0.000023 |
| 4 | t=23 | -0.000017 |
| 5 | t=8 | 0.000017 |

> **Recency concentration**: 27.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2022-10-28.png)

---

#### 2016-01-15 — score: 0.109946

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.000001 |
| 2 | f1_log_ret | 0.000001 |
| 3 | f6_rsi_14 | 0.000001 |
| 4 | f5_rel_ret | 0.000001 |
| 5 | f4_vol_z | 0.000001 |
| 6 | f3_parkinson | 0.000001 |
| 7 | f8_bb_width | 0.000001 |
| 8 | f2_gap_ret | 0.000000 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2016-01-15.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=89 | 0.000106 |
| 2 | t=31 | -0.000046 |
| 3 | t=24 | -0.000029 |
| 4 | t=93 | 0.000025 |
| 5 | t=91 | 0.000025 |

> **Recency concentration**: 39.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2016-01-15.png)

---

#### 2018-07-27 — score: 0.095604

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.000003 |
| 2 | f7_macd_hist | 0.000002 |
| 3 | f3_parkinson | 0.000002 |
| 4 | f1_log_ret | 0.000002 |
| 5 | f2_gap_ret | 0.000002 |
| 6 | f4_vol_z | 0.000002 |
| 7 | f6_rsi_14 | 0.000002 |
| 8 | f5_rel_ret | 0.000001 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_IG_2018-07-27.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=13 | 0.000090 |
| 2 | t=9 | -0.000062 |
| 3 | t=10 | -0.000052 |
| 4 | t=15 | 0.000043 |
| 5 | t=72 | 0.000040 |

> **Recency concentration**: 14.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_AT_TimeSHAP_2018-07-27.png)

---

## TimesNet + TranAD (`TranAD`)

### NVDA

#### 2025-10-29 — score: 5.019600

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.353484 |
| 2 | f1_log_ret | 0.164744 |
| 3 | f2_gap_ret | 0.065291 |
| 4 | f8_bb_width | 0.037334 |
| 5 | f4_vol_z | 0.025911 |
| 6 | f5_rel_ret | 0.016057 |
| 7 | f3_parkinson | 0.015306 |
| 8 | f6_rsi_14 | 0.013202 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-10-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 19.110573 |
| 2 | t=94 | -5.765452 |
| 3 | t=20 | -0.962229 |
| 4 | t=60 | -0.742212 |
| 5 | t=54 | -0.593092 |

> **Recency concentration**: 77.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-10-29.png)

---

#### 2025-10-28 — score: 4.281147

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.222872 |
| 2 | f1_log_ret | 0.116317 |
| 3 | f8_bb_width | 0.028533 |
| 4 | f2_gap_ret | 0.016001 |
| 5 | f3_parkinson | 0.015540 |
| 6 | f4_vol_z | 0.013481 |
| 7 | f5_rel_ret | 0.011137 |
| 8 | f6_rsi_14 | 0.010517 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-10-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 15.696314 |
| 2 | t=94 | -3.915453 |
| 3 | t=93 | -1.297163 |
| 4 | t=83 | -0.826359 |
| 5 | t=91 | 0.821908 |

> **Recency concentration**: 74.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-10-28.png)

---

#### 2025-03-14 — score: 2.795005

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.133976 |
| 2 | f1_log_ret | 0.107853 |
| 3 | f6_rsi_14 | 0.019228 |
| 4 | f8_bb_width | 0.012999 |
| 5 | f5_rel_ret | 0.012517 |
| 6 | f3_parkinson | 0.008959 |
| 7 | f2_gap_ret | 0.007575 |
| 8 | f4_vol_z | 0.006066 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-03-14.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.662810 |
| 2 | t=91 | 1.579720 |
| 3 | t=94 | 1.396558 |
| 4 | t=93 | -1.238901 |
| 5 | t=92 | 0.908824 |

> **Recency concentration**: 70.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-03-14.png)

---

#### 2025-03-03 — score: 2.778372

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.229120 |
| 2 | f1_log_ret | 0.141008 |
| 3 | f8_bb_width | 0.021247 |
| 4 | f5_rel_ret | 0.020463 |
| 5 | f6_rsi_14 | 0.015754 |
| 6 | f3_parkinson | 0.011929 |
| 7 | f4_vol_z | 0.010109 |
| 8 | f2_gap_ret | 0.009017 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-03-03.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 9.918360 |
| 2 | t=93 | -6.343679 |
| 3 | t=92 | 2.187795 |
| 4 | t=71 | 1.778034 |
| 5 | t=88 | -0.599462 |

> **Recency concentration**: 71.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-03-03.png)

---

#### 2023-05-25 — score: 2.666274

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.063551 |
| 2 | f2_gap_ret | 0.034883 |
| 3 | f1_log_ret | 0.007718 |
| 4 | f5_rel_ret | 0.006024 |
| 5 | f4_vol_z | 0.005167 |
| 6 | f6_rsi_14 | 0.002576 |
| 7 | f3_parkinson | 0.001418 |
| 8 | f8_bb_width | 0.001359 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2023-05-25.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 5.685102 |
| 2 | t=31 | -0.542994 |
| 3 | t=9 | -0.127060 |
| 4 | t=16 | -0.112195 |
| 5 | t=13 | -0.110596 |

> **Recency concentration**: 69.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2023-05-25.png)

---

#### 2025-04-09 — score: 2.644044

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.185088 |
| 2 | f1_log_ret | 0.137431 |
| 3 | f6_rsi_14 | 0.082972 |
| 4 | f3_parkinson | 0.057691 |
| 5 | f2_gap_ret | 0.046513 |
| 6 | f8_bb_width | 0.034953 |
| 7 | f5_rel_ret | 0.029091 |
| 8 | f4_vol_z | 0.026061 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-04-09.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -7.049460 |
| 2 | t=94 | 5.273939 |
| 3 | t=92 | 3.262952 |
| 4 | t=91 | 2.181655 |
| 5 | t=85 | -0.461249 |

> **Recency concentration**: 70.8% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-04-09.png)

---

#### 2016-11-11 — score: 2.630502

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.018329 |
| 2 | f2_gap_ret | 0.017913 |
| 3 | f1_log_ret | 0.017583 |
| 4 | f3_parkinson | 0.013986 |
| 5 | f4_vol_z | 0.007536 |
| 6 | f8_bb_width | 0.000873 |
| 7 | f6_rsi_14 | 0.000671 |
| 8 | f7_macd_hist | 0.000659 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2016-11-11.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 3.987309 |
| 2 | t=94 | -0.159531 |
| 3 | t=31 | -0.139766 |
| 4 | t=91 | -0.084647 |
| 5 | t=93 | -0.075526 |

> **Recency concentration**: 84.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2016-11-11.png)

---

#### 2025-08-20 — score: 2.376601

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.207882 |
| 2 | f1_log_ret | 0.057389 |
| 3 | f8_bb_width | 0.020114 |
| 4 | f5_rel_ret | 0.014730 |
| 5 | f6_rsi_14 | 0.014261 |
| 6 | f4_vol_z | 0.012636 |
| 7 | f3_parkinson | 0.005218 |
| 8 | f2_gap_ret | 0.003367 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-08-20.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 16.597034 |
| 2 | t=94 | -7.780278 |
| 3 | t=93 | -1.938948 |
| 4 | t=92 | -1.865046 |
| 5 | t=0 | 1.475653 |

> **Recency concentration**: 81.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-08-20.png)

---

#### 2025-02-06 — score: 2.333233

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.193241 |
| 2 | f1_log_ret | 0.082501 |
| 3 | f8_bb_width | 0.018722 |
| 4 | f5_rel_ret | 0.012597 |
| 5 | f6_rsi_14 | 0.012103 |
| 6 | f3_parkinson | 0.008484 |
| 7 | f2_gap_ret | 0.007843 |
| 8 | f4_vol_z | 0.005872 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-02-06.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -4.270603 |
| 2 | t=92 | 2.076279 |
| 3 | t=93 | 1.788072 |
| 4 | t=91 | 1.412779 |
| 5 | t=94 | 1.229306 |

> **Recency concentration**: 73.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-02-06.png)

---

#### 2025-03-12 — score: 2.182489

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.226713 |
| 2 | f1_log_ret | 0.085203 |
| 3 | f8_bb_width | 0.032642 |
| 4 | f6_rsi_14 | 0.019584 |
| 5 | f5_rel_ret | 0.011895 |
| 6 | f2_gap_ret | 0.005452 |
| 7 | f4_vol_z | 0.005316 |
| 8 | f3_parkinson | 0.005048 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_IG_2025-03-12.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=94 | 1.964094 |
| 2 | t=95 | 1.825612 |
| 3 | t=93 | 1.257742 |
| 4 | t=64 | -1.178098 |
| 5 | t=86 | -0.586221 |

> **Recency concentration**: 58.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\NVDA_TranAD_TimeSHAP_2025-03-12.png)

---

### TSLA

#### 2021-11-02 — score: 2.440174

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.168926 |
| 2 | f2_gap_ret | 0.031080 |
| 3 | f1_log_ret | 0.013855 |
| 4 | f5_rel_ret | 0.010294 |
| 5 | f8_bb_width | 0.007901 |
| 6 | f6_rsi_14 | 0.007123 |
| 7 | f3_parkinson | 0.005351 |
| 8 | f4_vol_z | 0.002239 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-11-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 14.581870 |
| 2 | t=94 | -1.905135 |
| 3 | t=92 | -1.803157 |
| 4 | t=93 | -1.743672 |
| 5 | t=89 | -1.222389 |

> **Recency concentration**: 81.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-11-02.png)

---

#### 2021-11-01 — score: 2.310388

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.162902 |
| 2 | f6_rsi_14 | 0.015512 |
| 3 | f8_bb_width | 0.014493 |
| 4 | f1_log_ret | 0.011574 |
| 5 | f5_rel_ret | 0.010393 |
| 6 | f4_vol_z | 0.006054 |
| 7 | f2_gap_ret | 0.004087 |
| 8 | f3_parkinson | 0.002556 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-11-01.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 7.265747 |
| 2 | t=94 | 2.877999 |
| 3 | t=90 | -1.605055 |
| 4 | t=93 | -1.059239 |
| 5 | t=92 | -1.003604 |

> **Recency concentration**: 81.0% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-11-01.png)

---

#### 2025-03-10 — score: 2.041117

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.140040 |
| 2 | f1_log_ret | 0.088879 |
| 3 | f3_parkinson | 0.023189 |
| 4 | f8_bb_width | 0.018811 |
| 5 | f5_rel_ret | 0.017915 |
| 6 | f6_rsi_14 | 0.016511 |
| 7 | f4_vol_z | 0.006590 |
| 8 | f2_gap_ret | 0.004589 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-03-10.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=93 | 1.929140 |
| 2 | t=94 | 1.284863 |
| 3 | t=91 | 0.484217 |
| 4 | t=95 | -0.299549 |
| 5 | t=86 | -0.276861 |

> **Recency concentration**: 71.7% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-03-10.png)

---

#### 2021-10-25 — score: 1.693685

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.077946 |
| 2 | f8_bb_width | 0.031016 |
| 3 | f3_parkinson | 0.019547 |
| 4 | f5_rel_ret | 0.019135 |
| 5 | f1_log_ret | 0.011385 |
| 6 | f2_gap_ret | 0.010302 |
| 7 | f6_rsi_14 | 0.009583 |
| 8 | f4_vol_z | 0.008220 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-10-25.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 7.217529 |
| 2 | t=93 | -1.222266 |
| 3 | t=94 | -0.525103 |
| 4 | t=92 | -0.313624 |
| 5 | t=46 | -0.236770 |

> **Recency concentration**: 75.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-10-25.png)

---

#### 2021-10-28 — score: 1.582624

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.105153 |
| 2 | f8_bb_width | 0.010737 |
| 3 | f1_log_ret | 0.007081 |
| 4 | f6_rsi_14 | 0.006682 |
| 5 | f2_gap_ret | 0.006323 |
| 6 | f3_parkinson | 0.005478 |
| 7 | f5_rel_ret | 0.001238 |
| 8 | f4_vol_z | 0.000709 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-10-28.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 2.763242 |
| 2 | t=94 | 1.463269 |
| 3 | t=92 | -0.666467 |
| 4 | t=93 | -0.143199 |
| 5 | t=67 | -0.124724 |

> **Recency concentration**: 79.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-10-28.png)

---

#### 2025-03-03 — score: 1.136966

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.093342 |
| 2 | f1_log_ret | 0.041524 |
| 3 | f6_rsi_14 | 0.009826 |
| 4 | f8_bb_width | 0.008867 |
| 5 | f3_parkinson | 0.008442 |
| 6 | f5_rel_ret | 0.006717 |
| 7 | f2_gap_ret | 0.005595 |
| 8 | f4_vol_z | 0.005435 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-03-03.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | -1.698666 |
| 2 | t=93 | 1.235929 |
| 3 | t=92 | 0.606461 |
| 4 | t=91 | 0.480916 |
| 5 | t=94 | 0.425418 |

> **Recency concentration**: 74.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-03-03.png)

---

#### 2025-01-02 — score: 1.125827

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.120810 |
| 2 | f1_log_ret | 0.031860 |
| 3 | f8_bb_width | 0.005883 |
| 4 | f6_rsi_14 | 0.005536 |
| 5 | f3_parkinson | 0.004174 |
| 6 | f4_vol_z | 0.003454 |
| 7 | f5_rel_ret | 0.002880 |
| 8 | f2_gap_ret | 0.002570 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-01-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 14.540086 |
| 2 | t=94 | -5.308082 |
| 3 | t=93 | -3.265721 |
| 4 | t=92 | -2.095216 |
| 5 | t=60 | -0.530213 |

> **Recency concentration**: 85.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-01-02.png)

---

#### 2021-10-29 — score: 1.093910

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.096216 |
| 2 | f8_bb_width | 0.009146 |
| 3 | f6_rsi_14 | 0.006559 |
| 4 | f1_log_ret | 0.005711 |
| 5 | f3_parkinson | 0.002921 |
| 6 | f5_rel_ret | 0.001165 |
| 7 | f2_gap_ret | 0.000863 |
| 8 | f4_vol_z | 0.000647 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2021-10-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 4.529955 |
| 2 | t=91 | -0.666681 |
| 3 | t=93 | -0.477292 |
| 4 | t=92 | -0.291539 |
| 5 | t=94 | -0.109967 |

> **Recency concentration**: 78.3% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2021-10-29.png)

---

#### 2020-02-05 — score: 1.052270

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.011362 |
| 2 | f3_parkinson | 0.008145 |
| 3 | f1_log_ret | 0.007701 |
| 4 | f5_rel_ret | 0.006515 |
| 5 | f8_bb_width | 0.006496 |
| 6 | f6_rsi_14 | 0.002434 |
| 7 | f2_gap_ret | 0.001765 |
| 8 | f4_vol_z | 0.001257 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2020-02-05.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.588769 |
| 2 | t=94 | 0.113192 |
| 3 | t=25 | -0.086022 |
| 4 | t=26 | -0.048724 |
| 5 | t=3 | -0.048453 |

> **Recency concentration**: 79.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2020-02-05.png)

---

#### 2025-03-11 — score: 0.971735

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f7_macd_hist | 0.083072 |
| 2 | f1_log_ret | 0.048970 |
| 3 | f8_bb_width | 0.017066 |
| 4 | f6_rsi_14 | 0.012803 |
| 5 | f3_parkinson | 0.006413 |
| 6 | f5_rel_ret | 0.004790 |
| 7 | f2_gap_ret | 0.003106 |
| 8 | f4_vol_z | 0.002707 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_IG_2025-03-11.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=94 | 1.713228 |
| 2 | t=95 | -1.318201 |
| 3 | t=3 | -0.552743 |
| 4 | t=12 | -0.426772 |
| 5 | t=85 | -0.352229 |

> **Recency concentration**: 48.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\TSLA_TranAD_TimeSHAP_2025-03-11.png)

---

### INTC

#### 2025-09-18 — score: 2.055028

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.046355 |
| 2 | f5_rel_ret | 0.021761 |
| 3 | f1_log_ret | 0.018989 |
| 4 | f4_vol_z | 0.015783 |
| 5 | f3_parkinson | 0.013175 |
| 6 | f8_bb_width | 0.006155 |
| 7 | f7_macd_hist | 0.004015 |
| 8 | f6_rsi_14 | 0.003600 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2025-09-18.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 5.610976 |
| 2 | t=57 | -0.903907 |
| 3 | t=26 | -0.263050 |
| 4 | t=72 | -0.238518 |
| 5 | t=74 | -0.224455 |

> **Recency concentration**: 62.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2025-09-18.png)

---

#### 2024-08-02 — score: 1.627456

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.040944 |
| 2 | f4_vol_z | 0.018358 |
| 3 | f2_gap_ret | 0.016264 |
| 4 | f3_parkinson | 0.012956 |
| 5 | f1_log_ret | 0.011141 |
| 6 | f7_macd_hist | 0.009039 |
| 7 | f6_rsi_14 | 0.002687 |
| 8 | f8_bb_width | 0.002020 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2024-08-02.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 5.000049 |
| 2 | t=28 | -0.800799 |
| 3 | t=11 | -0.459608 |
| 4 | t=83 | -0.207969 |
| 5 | t=76 | -0.187043 |

> **Recency concentration**: 68.4% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2024-08-02.png)

---

#### 2020-07-24 — score: 1.237404

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.018875 |
| 2 | f4_vol_z | 0.016409 |
| 3 | f2_gap_ret | 0.013171 |
| 4 | f1_log_ret | 0.010522 |
| 5 | f3_parkinson | 0.005159 |
| 6 | f6_rsi_14 | 0.003254 |
| 7 | f8_bb_width | 0.002489 |
| 8 | f7_macd_hist | 0.001422 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2020-07-24.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 2.871233 |
| 2 | t=71 | -0.169944 |
| 3 | t=9 | -0.141977 |
| 4 | t=56 | -0.127980 |
| 5 | t=93 | 0.123800 |

> **Recency concentration**: 66.9% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2020-07-24.png)

---

#### 2021-10-22 — score: 0.649180

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.195423 |
| 2 | f3_parkinson | 0.057655 |
| 3 | f2_gap_ret | 0.023331 |
| 4 | f7_macd_hist | 0.022989 |
| 5 | f4_vol_z | 0.019705 |
| 6 | f5_rel_ret | 0.014290 |
| 7 | f1_log_ret | 0.012878 |
| 8 | f6_rsi_14 | 0.012712 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2021-10-22.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.637540 |
| 2 | t=7 | -0.156021 |
| 3 | t=31 | -0.134386 |
| 4 | t=56 | -0.119698 |
| 5 | t=47 | -0.108098 |

> **Recency concentration**: 47.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2021-10-22.png)

---

#### 2022-07-29 — score: 0.603087

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.013383 |
| 2 | f5_rel_ret | 0.012637 |
| 3 | f1_log_ret | 0.007393 |
| 4 | f4_vol_z | 0.005791 |
| 5 | f3_parkinson | 0.004772 |
| 6 | f7_macd_hist | 0.004258 |
| 7 | f8_bb_width | 0.003889 |
| 8 | f6_rsi_14 | 0.002898 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2022-07-29.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.466798 |
| 2 | t=67 | -0.133616 |
| 3 | t=33 | -0.130726 |
| 4 | t=94 | -0.123647 |
| 5 | t=40 | -0.115864 |

> **Recency concentration**: 47.6% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2022-07-29.png)

---

#### 2020-10-23 — score: 0.499188

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f4_vol_z | 0.007234 |
| 2 | f5_rel_ret | 0.006431 |
| 3 | f2_gap_ret | 0.003841 |
| 4 | f3_parkinson | 0.002055 |
| 5 | f6_rsi_14 | 0.001668 |
| 6 | f1_log_ret | 0.001599 |
| 7 | f7_macd_hist | 0.001415 |
| 8 | f8_bb_width | 0.000959 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2020-10-23.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.596598 |
| 2 | t=31 | -0.556951 |
| 3 | t=94 | -0.111817 |
| 4 | t=7 | -0.077698 |
| 5 | t=93 | 0.033417 |

> **Recency concentration**: 62.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2020-10-23.png)

---

#### 2025-02-19 — score: 0.465948

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f8_bb_width | 0.005663 |
| 2 | f7_macd_hist | 0.005212 |
| 3 | f5_rel_ret | 0.005150 |
| 4 | f2_gap_ret | 0.004973 |
| 5 | f1_log_ret | 0.004372 |
| 6 | f6_rsi_14 | 0.003191 |
| 7 | f3_parkinson | 0.001722 |
| 8 | f4_vol_z | 0.001137 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2025-02-19.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.036150 |
| 2 | t=94 | 0.149553 |
| 3 | t=23 | -0.060982 |
| 4 | t=74 | -0.053559 |
| 5 | t=93 | -0.042949 |

> **Recency concentration**: 67.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2025-02-19.png)

---

#### 2019-04-26 — score: 0.438478

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f5_rel_ret | 0.006799 |
| 2 | f4_vol_z | 0.005321 |
| 3 | f2_gap_ret | 0.002961 |
| 4 | f1_log_ret | 0.002499 |
| 5 | f6_rsi_14 | 0.002316 |
| 6 | f3_parkinson | 0.001598 |
| 7 | f8_bb_width | 0.001356 |
| 8 | f7_macd_hist | 0.001226 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2019-04-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 0.815132 |
| 2 | t=32 | -0.134617 |
| 3 | t=31 | -0.071853 |
| 4 | t=66 | -0.048699 |
| 5 | t=89 | -0.043493 |

> **Recency concentration**: 67.5% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2019-04-26.png)

---

#### 2024-01-26 — score: 0.343203

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f2_gap_ret | 0.008062 |
| 2 | f5_rel_ret | 0.005870 |
| 3 | f4_vol_z | 0.004941 |
| 4 | f1_log_ret | 0.003182 |
| 5 | f3_parkinson | 0.002326 |
| 6 | f6_rsi_14 | 0.001539 |
| 7 | f8_bb_width | 0.001357 |
| 8 | f7_macd_hist | 0.000800 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2024-01-26.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.015789 |
| 2 | t=34 | -0.118585 |
| 3 | t=50 | -0.044563 |
| 4 | t=48 | -0.044148 |
| 5 | t=93 | 0.040798 |

> **Recency concentration**: 63.1% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2024-01-26.png)

---

#### 2025-02-18 — score: 0.323612

**Integrated Gradients — Top Features**

| Rank | Feature | Mean |Attribution| |
|------|---------|---------------------|
| 1 | f1_log_ret | 0.013000 |
| 2 | f5_rel_ret | 0.004653 |
| 3 | f7_macd_hist | 0.003396 |
| 4 | f3_parkinson | 0.002452 |
| 5 | f8_bb_width | 0.002221 |
| 6 | f2_gap_ret | 0.002017 |
| 7 | f4_vol_z | 0.001458 |
| 8 | f6_rsi_14 | 0.000878 |

![IG Heatmap](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_IG_2025-02-18.png)

**TimeSHAP — Temporal Shapley Values**

| Rank | Timestep | Shapley Value φ(t) |
|------|----------|--------------------|
| 1 | t=95 | 1.389706 |
| 2 | t=75 | -0.104529 |
| 3 | t=93 | 0.094814 |
| 4 | t=24 | -0.091146 |
| 5 | t=28 | -0.070536 |

> **Recency concentration**: 61.2% of total |φ(t)| is concentrated in the last 20% of the window (timesteps 76–95).

![TimeSHAP](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\INTC_TranAD_TimeSHAP_2025-02-18.png)

---

## Cross-Model Comparison

### Feature Importance

![Feature Importance Comparison](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\comparison_feature_importance.png)

### Temporal Receptive Field

![Temporal Comparison](C:\Users\Zephyrus M16\Desktop\UNI\3rd - SP2026\ADY201m\Main project\results\xai\comparison_temporal_importance.png)
