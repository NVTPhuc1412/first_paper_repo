# Deep Learning Anomaly Detection with XAI and LLM Attribution

A comprehensive pipeline for financial time-series anomaly detection. This project combines state-of-the-art Deep Learning models (Anomaly Transformer, TimesNet + TranAD) with Explainable AI (Integrated Gradients, TimeSHAP) and Large Language Models (Gemini 3.1 lite) to not only detect market anomalies but also explain *why* they happened using real-world news headlines.

---

## Features

*   **Deep Learning Detectors**:
    *   **Anomaly Transformer**: Captures association discrepancies.
    *   **TimesNet + TranAD**: Combines 2D temporal variations with adversarial training.
*   **Explainable AI (XAI)**:
    *   **Integrated Gradients (IG)**: Identifies which technical indicators (features) drove the anomaly score.
    *   **TimeSHAP**: Provides temporal attribution, highlighting exactly *when* the anomalous behavior started in the sequence window.
*   **LLM News Attribution**:
    *   Automatically fetches concurrent financial news via **GDELT**.
    *   Uses **Google Gemini** to analyze headlines, rank their relevance to the price action, and generate human-readable summaries explaining the market context.
*   **Interactive Dashboard**:
    *   Clean, 16:9-optimized Streamlit dashboard.
    *   Interactive Plotly charts for price, anomaly scores, and XAI results.
    *   Collapsible detail expanders for full LLM reports and headline caches.

---

## Installation

1.  **Clone and set up the environment**:
    Ensure you have Python 3.10+ installed. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires PyTorch, Captum, Streamlit, Plotly, yfinance, and google-genai)*

2.  **API Keys**:
    To use the LLM News Attribution feature, you need a Google Gemini API key:
    ```bash
    # Windows Command Prompt
    set GEMINI_API_KEY=your_key_here

    # Linux / macOS
    export GEMINI_API_KEY=your_key_here
    ```

---

## Usage pipelines

The architecture is divided into two primary entry points: the **Analysis Pipeline** (which crunches the data) and the **Dashboard** (which visualizes it).

### 1. Run the Analysis Pipeline

The pipeline runs sequentially: Data Fetching → Feature Engineering → Model Scoring → XAI Analysis → GDELT Fetching → LLM Attribution.

```bash
# Run the full pipeline for Nvidia
python pipeline.py NVDA

# Run without LLM attribution (faster, no API key needed)
python pipeline.py TSLA --skip-llm

# Dry run (only fetch data and generate features)
python pipeline.py INTC --dry-run
```

All results are cached on disk under `results/pipeline/{TICKER}/` and `data/xai_cache/`.

### 2. Launch the Interactive Dashboard

Once the pipeline has completed for at least one ticker, you can explore the results visually:

```bash
python -m streamlit run demo.py
```

*   **Top row**: Anomaly scores overlaid on price history, alongside the top 3 most severe anomalies per model.
*   **Middle row**: XAI feature importance (IG) and temporal heatmaps (TimeSHAP).
*   **Bottom row**: Collapsible tabs containing the generated LLM summaries and the raw cached news headlines.

---

## Project Structure

```text
├── pipeline.py                 # Main CLI entry point for analysis
├── pipeline_config.py          # Centralized configuration (paths, hyperparameters)
├── demo.py                     # Streamlit interactive dashboard
├── extract_anomaly_scores.py   # Runs PyTorch models to generate scores
├── scrapers/                   # Data fetching & preprocessing modules
│   ├── fetch.py                # yfinance wrapper
│   └── feature_engineer_*.py   # Technical indicator generation
├── xai_analysis/               # Explainability & Attribution
│   ├── attribute.py            # Gemini LLM wrappers and prompt building
│   ├── news_fetcher.py         # GDELT API integration
│   ├── run_ig.py               # Captum Integrated Gradients
│   └── plot_paper_*.py         # High-res publication-style chart generation
├── models/                     # PyTorch model definitions (AT, TranAD, TimesNet)
├── data/                       # Cached models, news, and XAI arrays
└── results/                    # Pipeline outputs (CSVs, PNG plots, Markdown reports)
```

## License & Acknowledgments

*   Built with PyTorch and Captum.
*   News headlines provided by the [GDELT Project](https://www.gdeltproject.org/).
*   LLM Attribution powered by Google Gemini.
