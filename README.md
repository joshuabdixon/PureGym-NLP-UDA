# Unstructured Data Analysis Final Project — PureGym Trustpilot Reviews

A reproducible pipeline for the Unstructured Data Analysis module (Imperial College London, MSc Machine Learning and Data Science). 

The project processes Trustpilot reviews for PureGym, performs text preprocessing, language filtering, topic modelling (BERTopic, LDA, and BERTopic over Falcon-derived topics), emotion analysis (BERT/DistilRoBERTa models), and compiles tables and plots ready for inclusion in the final report.

## Context and Motivation: Why I chose this project

I approach this project as a consultancy-style case study, using public Trustpilot reviews for PureGym to explore how large volumes of unstructured customer feedback can be summarised and analysed using NLP techniques. The aim is to build a practical, end-to-end analysis pipeline that could be adapted to a workplace setting with minimal changes, for example by swapping in a different review source or business domain.

Due to data access and permission constraints, I am unable to use proprietary company data. Instead, I use a publicly available PureGym dataset as a realistic proxy. My interpretation of the results is informed by general familiarity with the fitness industry and first-hand experience as a PureGym customer, which helps ground the findings in a plausible operational context while keeping the analysis fully reproducible.

## What This Project Does
- Filters and cleans raw Trustpilot reviews and ensures language consistency.
- Applies robust text preprocessing tailored for topic modelling and sentiment/emotion analysis.
- Trains and evaluates topic models:
  - BERTopic (all, negative-only, emotion-sliced subsets)
  - LDA (negative and sadness-only subsets)
  - BERTopic over Falcon-extracted topics (LLM-derived topics)
- Runs emotion classification using two transformer-based models.
- Consolidates outputs into report-ready tables and plots.

## Data

The raw Trustpilot review data are provided in `data/` as `PureGym Customer Reviews.csv`. The dataset was obtained from a publicly available Kaggle repository: [https://www.kaggle.com/datasets/zackyboi/puregym-customer-reviews-sentiment-analysis](https://www.kaggle.com/datasets/zackyboi/puregym-customer-reviews-sentiment-analysis).


## Getting Started

### Prerequisites
- Python 3.10 (3.10.11 recommended)

### Key packages and versions
This project relies on standard Python data science and NLP libraries (for example pandas, NumPy, PyTorch, Hugging Face Transformers, BERTopic, Gensim, and UMAP). The full list of dependencies and pinned package versions used for the project environment can be found in `requirements.txt`.

### Example Setup (virtual environment + dependencies)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration
Paths and analysis settings in `config.ini`.
- `[DATA]`: input/output filenames in `data/`
- `[OUTPUT]`: folders for plots, tables, and models (under `output/`)
- `[FILTERING]`: selected columns, language, and country code
- `[ANALYSIS_DATES]`: date range and date column
- `[REPRODUCIBILITY]`: random seed for consistent results
- `[MODELS]`: canonical names for saved model folders

Here’s a polished version with a couple of grammar fixes and smoother phrasing (not changing meaning):

## Usage (run notebooks in order)

Initially, the repository was structured with the stages below modularised across multiple notebooks, with supporting classes and functions implemented in separate scripts and imported as needed.

To simplify grading, these components have been consolidated into two main notebooks (**01to06_core_analysis.ipynb** and **07_topic_modelling_falcon.ipynb**), along with a final optional notebook (**08_compile_results_for_report.ipynb**) that collates key results in a single location and generates LaTeX tables for the report. This final notebook does not perform additional analysis and may be skipped.

Outside of the notebooks, two key shared scripts used across the main pipeline are **text_preprocessor.py** and **bertopic_runner.py**. Their locations are shown in the **Repo Structure** section below.


1. **notebooks/01to06_core_analysis.ipynb**
   - Performs end-to-end steps for data filtering, preprocessing, topic modelling (BERTopic, LDA), and emotion analysis.
   - Writes intermediate and final artifacts into `data/` and `output/`.

      **01_preprocessing**
      - Setup and config for paths/resources.
      - Load raw reviews and basic quality checks.
      - Filter by language, dates, and ratings.
      - Preprocess text for topics and emotions.
      - Save processed datasets for downstream use.

      **02_eda_and_split_negative_review_data**
      - Configure plotting utilities and settings.
      - Visualise word frequencies (all/negative/non-negative).
      - Create and save split datasets for analysis.

      **03_topic_modelling_bertopic**
      - Fit BERTopic on target datasets (negative/emotion slices).
      - Export topics, top words, visuals; save models/plots.

      **04a_emotion_analysis_roberta**
      - Run DistilRoBERTa emotion classifier; aggregate counts.
      - Check token lengths to validate inputs.
      - Save emotion outputs.

      **04b_emotion_analysis_bert_base_uncased**
      - Run BERT-base emotion classifier.
      - Save emotion outputs and summaries.

      **05_topic_modelling_bertopic_per_emotion_neg_reviews**
      - Configure emotion-sliced subsets (e.g., sadness-only) on the results from **4a**.
      - Fit BERTopic; generate topics and plots.
      - Save results for reporting.

      **06_topic_modelling_lda_gensim**
      - Build corpus/dictionary from preprocessed text.
      - Train LDA on negative and emotion subsets.
      - Export topics and evaluation summaries.

2. **notebooks/07_topic_modelling_falcon.ipynb**
      - Runs BERTopic over Falcon-extracted topics (LLM-derived) for negative reviews.
      - Outputs under `output/models/07_topic_modelling_falcon/` and related plot/table folders.

3. **notebooks/08_compile_results_for_report.ipynb**
      - Consolidates selected CSV tables and PNG plots into `output/tables_report/` and `output/plots_report/`.
      - Generates combined LaTeX tables for inclusion in the final report.
      - This notebook summarises outputs from earlier notebooks and does not perform any additional analysis. It may be skipped if not required.


## Repo Structure
```
Assignment 2/
├── config.ini
├── requirements.txt
├── __init__.py
├── data/
│   ├── PureGym Customer Reviews.csv 
│   ├── PureGym Customer Reviews_raw_filtered.csv
│   ├── PureGym Customer Reviews_preprocessed.csv
│   ├── PureGym Customer Reviews_preprocessed_sentiment.csv
│   ├── PureGym Customer Reviews_preprocessed_emotion.csv
│   ├── PureGym Customer Reviews_preprocessed_negative.csv
│   ├── PureGym Customer Reviews_preprocessed_negative_emotion.csv
│   ├── PureGym Customer Reviews_preprocessed_non_negative.csv
│   └── PureGym Customer Reviews_preprocessed_non_negative_emotion.csv
├── notebooks/
│   ├── 01to06_core_analysis.ipynb
│   ├── 07_topic_modelling_falcon.ipynb
│   └── 08_compile_results_for_report.ipynb
├── utils/
│   ├── data_management/
│   │   ├── data_io.py
│   │   └── __init__.py
│   └── processing/
│       ├── text_preprocessor.py
│       └── __init__.py
├── modelling/
│   └── bertopic/
│       └── bertopic_runner.py
└── output/
    ├── models/
    ├── plots/
    ├── plots_report/
    ├── tables/
    └── tables_report/
```

## Additional notes
- Notebooks are written to resolve the project/repo root automatically as. If running from a different working directory, open notebooks from the repo root.
- Outputs are reproducible where supported (seeded; greedy decoding for Falcon). 


