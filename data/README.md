# Dataset Sources and Preparation

This repository contains analysis code and aggregate results.
Raw review datasets and trained model weights are not included.

## 1. Product Reviews

### Sources

Download source used for this project:
https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

Original research:
*Creating and Detecting Fake Reviews of Online Products*,
by Joni Salminen, Chandrashekhar Kandpal, Ahmed Mohamed Kamel,
Soon-Gyo Jung, and Bernard J. Jansen.

Authors' research repository:
https://github.com/joolsa/FakeReviews

The Kaggle uploader is the download provider; the original
researchers should also be credited.

### File and labels

Original file: `fake reviews dataset.csv`

Required columns:
- `category`
- `rating`
- `label`
- `text_`

Label mapping:
- `OR` → 0: Original
- `CG` → 1: Computer-generated

These labels represent text origin, not independently verified
truthfulness or deception.

### Cleaning

The downloaded CSV contained 40,432 reviews.

Preparation removed:
- 20 additional exact-duplicate rows.
- 7 additional duplicates after case and whitespace normalization.
- 1 review containing no letters or numbers.

The cleaned dataset contained 40,404 reviews:
- Original: 20,215
- Computer-generated: 20,189

Original review text was preserved. Normalized text was used
for duplicate comparison.

Near-duplicate screening used MinHash candidates followed by
exact Jaccard similarity on sets of consecutive three-word phrases,
with a final threshold of 0.90. It identified 58 pairs forming
46 related-review groups containing 98 reviews.

Related reviews were grouped rather than automatically removed.

### Splits

| Split | Reviews | Original | Computer-generated |
|---|---:|---:|---:|
| Training | 28,282 | 14,174 | 14,108 |
| Validation | 6,064 | 3,022 | 3,042 |
| Testing | 6,058 | 3,019 | 3,039 |

GroupShuffleSplit with random_state=42 was used in two stages:
approximately 70% training, then an approximately equal division
of the remaining reviews into validation and testing.

No duplicate groups or review IDs were shared across splits.

## 2. Hotel Reviews

### Source

Dataset: Deceptive Opinion Spam Corpus v1.4

Archive used:
https://myleott.com/op_spam_v1.4.zip

Original research:

- Ott, Choi, Cardie, and Hancock (2011).
  *Finding Deceptive Opinion Spam by Any Stretch of the Imagination.*
  https://aclanthology.org/P11-1032/

- Ott, Cardie, and Hancock (2013).
  *Negative Deceptive Opinion Spam.*
  https://aclanthology.org/N13-1053/

### Labels and composition

Original archive: `op_spam_v1.4.zip`

Label mapping:
- `truthful` → 0
- `deceptive` → 1

The original corpus contained 1,600 reviews across 20 hotels,
covering positive and negative sentiment.

The deceptive reviews are human-written; they are not the
computer-generated class used in the product dataset.

### Cleaning

Four additional normalized duplicate rows were removed.

The cleaned dataset contained 1,596 reviews:
- Truthful: 796
- Deceptive: 800

No additional hotel near-duplicate pairs were found using
three-word-set Jaccard similarity at or above 0.90.

### Splits

Original folds were preserved:

| Split | Original folds | Hotels | Reviews | Truthful | Deceptive |
|---|---|---:|---:|---:|---:|
| Training | 1, 2, 3 | 12 | 956 | 476 | 480 |
| Validation | 4 | 4 | 320 | 160 | 160 |
| Testing | 5 | 4 | 320 | 160 | 160 |

Hotels, review IDs, and normalized texts were disjoint across splits.

This is a fixed hotel-disjoint holdout, not five-fold cross-validation.

## 3. Cross-Dataset Overlap Checks

All saved splits were checked across the product and hotel datasets.

Results:
- Normalized exact-match pairs: 0
- Three-word-set Jaccard pairs at or above 0.90: 0
- Product reviews excluded from the phrase-set check for having
  fewer than three words: 1
- Hotel reviews excluded: 0

The short product review was still included in exact matching.
These checks do not rule out paraphrases with different wording.

## 4. Reproduction Notes

- Obtain datasets from the sources above.
- Follow the notebook's upload and preparation instructions.
- Preserve review IDs, duplicate groups, and original hotel folds.
- Use the saved splits consistently across model comparisons.
- Fit text vectorizers and feature scalers on training data only.
- Select models using validation data, not test performance.
- DistilBERT uses a maximum input length of 256 tokens.
- The notebook retains historical Drive paths that must be
  adapted to your own saved files.

## 5. Interpretation and Usage

Cross-dataset evaluation maps both datasets to numerical labels 0/1,
but their meanings are different:

- Product: original versus computer-generated.
- Hotel: truthful versus deceptive.

Transfer results therefore reflect changes in both domain and
classification task. They do not isolate domain shift or demonstrate
that annotation scarcity caused the performance decline.

Consult each dataset's source page and accompanying documentation
for applicable licence and usage terms. Any licence for this
repository's code does not replace third-party dataset terms.
