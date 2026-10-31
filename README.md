# Fake Review Detection: Classical Models vs DistilBERT

An undergraduate research project comparing classical and
transformer-based models for review classification, writing-style
analysis, and cross-dataset transfer.

**Author:** Loganathan Kajamugan  
**Institution:** South Eastern University of Sri Lanka  
**Supervisor:** Prof HMM Naleer

## Research Questions

1. How do TF-IDF Logistic Regression and Linear SVM compare with DistilBERT?
2. Does adding writing-style features improve classification?
3. How well do models transfer between product and hotel review datasets?

## Datasets

| Dataset | Cleaned reviews | Labels |
|---|---:|---|
| Product reviews | 40,404 | Original / Computer-generated |
| Hotel reviews | 1,596 | Truthful / Deceptive |

Product reviews were split using related-review groups to reduce leakage.
Hotel splits kept hotels separate across training, validation, and testing.

Dataset source and setup documentation will be added separately.
Datasets and model weights are not included in this repository.

## Models

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + eight writing-style features + Linear SVM
- DistilBERT, evaluated across three training seeds

The style features measure review length, average word and sentence
length, punctuation, capitalization, word repetition, and a fixed
lexicon of exaggerated language.

## Final Test Results

Scores below are **macro-F1 percentages**, not accuracy.
DistilBERT values are the mean across three seeds.

| Model | Product → Product | Hotel → Hotel | Product → Hotel | Hotel → Product |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.64 | 89.37 | 35.38 | 35.93 |
| TF-IDF SVM | 94.21 | **89.69** | 34.71 | 35.98 |
| TF-IDF + Style SVM | 94.57 | 88.43 | 33.33 | 41.48 |
| DistilBERT | **97.82** | 87.26 | 34.28 | **60.62** |

The source dataset appears before the arrow; the evaluation dataset
appears after it.

## Main Findings

- DistilBERT performed best on product reviews.
- TF-IDF SVM achieved the highest observed hotel test score.
- Strong within-dataset performance did not guarantee successful transfer.
- Writing-style features provided mixed benefits.
- Classical models had faster inference in the measured deployment setup.

An exploratory paired duplicate-group bootstrap supported DistilBERT's
product-test advantage over the combined SVM. The smaller improvement
from adding style features to the product SVM had an interval containing
zero, so a clear benefit was not established.

## Evaluation and Analysis

- Exact and near-duplicate checks
- Group-aware product splits and hotel-disjoint splits
- Validation-based model selection
- Feature ablation experiments
- Three-seed DistilBERT evaluation
- Bidirectional cross-dataset evaluation
- Per-category and per-hotel analysis
- Qualitative error inspection and truncation checks
- Exploratory paired group-bootstrap comparisons
- Inference benchmarking
- Saved-prediction consistency checks for all 24 final evaluations

## Notebook

Open [the experiment notebook](fake_review_detection_experiments_github.ipynb)
to inspect the workflow and retained results.

The notebook is a documented Google Colab experiment record.
It contains upload prompts and paths to artifacts created during
multiple sessions. It is not yet a verified one-click reproduction.

To adapt it for execution:

1. Open a working copy in Google Colab.
2. Obtain the required datasets and respect their licence terms.
3. Configure Google Drive paths for your own files.
4. Supply or generate the saved splits and model artifacts required
   by each section.
5. Use a GPU for transformer training.
6. Read the section instructions before running cells.

Do not use "Run all" on your only copy of saved research artifacts.

## Environment

Experiments used Python, pandas, NumPy, scikit-learn, PyTorch,
Hugging Face Transformers, Datasets, Accelerate, and Google Colab.

Recorded transformer package versions:

- transformers: 5.20.2
- datasets: 4.8.5
- accelerate: 1.15.0

Transformer training used a Tesla T4 GPU.

## Limitations

- The datasets have different label meanings: generated-text detection
  and deception detection are different tasks.
- Cross-dataset results reflect changes in both domain and task meaning.
- The study does not establish annotation scarcity as the cause of
  transfer failure or implement a domain-adaptation solution.
- The hotel test set contains only four held-out hotels.
- Bootstrap intervals are exploratory and conditional on saved models;
  they do not include retraining uncertainty.
- Speed measurements compare classical models on CPU with DistilBERT
  on a T4 GPU, rather than identical hardware.
- Saved predictions reproduce the reported metrics, but a complete
  clean-session rerun has not yet been verified.

## Project Status

The comparison experiments and analyses are complete.
Repository documentation and reproducibility improvements are ongoing.
