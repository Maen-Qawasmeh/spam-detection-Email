# Spam Detection with NLP and Unsupervised Clustering

A notebook-based natural language processing project that explores spam and non-spam messages, cleans their text, extracts TF-IDF features, and compares five clustering algorithms. The project uses known training labels to evaluate the clusters and creates prediction CSV files for three models.

The complete workflow is implemented in **`spam-detection-competition.ipynb`**.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Project Workflow](#project-workflow)
- [Models and Results](#models-and-results)
- [Outputs](#outputs)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Spam detection separates unwanted messages from legitimate content. This project investigates whether text similarity can group messages into two clusters corresponding to spam and non-spam.

The labels are:

- **`0`**: non-spam.
- **`1`**: spam.

The main objectives are to:

- Explore label distributions and common words.
- Clean and tokenize English message text.
- Convert text into numerical TF-IDF representations.
- Compare MiniBatchKMeans, KMeans, BIRCH, Spectral Clustering, and Gaussian Mixture.
- Evaluate precision, recall, F1 score, accuracy, and a confusion matrix.
- Export cluster-based predictions for the external test records.

The models are fitted without class labels. Labels are used afterward for evaluation and, for several models, to determine whether cluster IDs should be reversed. This is an **unsupervised clustering experiment with label-based interpretation**, rather than a supervised classifier trained on spam labels.

## Dataset

The notebook requires two CSV files in its working directory:

| File | Required columns | Purpose |
| --- | --- | --- |
| `Train_set.csv` | `Body`, `Label` | Message text and known spam labels. |
| `Test_set.csv` | `Body`, `id` | External message text and identifiers for prediction export. |

Columns beginning with `Unnamed` are removed from both files.

The supplied notebook does not include these CSV files or document the original competition URL, dataset source, or license. Obtain the matching data from its original provider before running the project.

### Saved Dataset Statistics

| Item | Count |
| --- | ---: |
| Training messages | 13,153 |
| Non-spam training messages | 8,400 |
| Spam training messages | 4,753 |
| External test messages | 5,638 |
| TF-IDF features in the saved run | 1,000 |

The saved training set is approximately **63.86% non-spam** and **36.14% spam**. These counts come from the notebook's stored matrix shapes and classification reports.

## Project Structure

```text
spam-detection-clustering/
├── README.md
├── spam-detection-competition.ipynb
├── Train_set.csv                # Required input; obtain separately
├── Test_set.csv                 # Required input; obtain separately
├── miniKmeans_Submission.csv    # Generated MiniBatchKMeans predictions
├── Kmeans_Submission.csv        # Generated KMeans predictions
└── Birch_Submission.csv         # Generated BIRCH predictions
```

No standalone application, serialized model, or dependency lock file is included.

## Requirements

The notebook metadata records **Python 3.12.12**. Exact package versions are not recorded.

| Package | Purpose |
| --- | --- |
| `numpy` | Numerical operations. |
| `pandas` | CSV loading, text processing, and export. |
| `matplotlib` | Charts and image display. |
| `seaborn` | Statistical plots and confusion matrix heatmap. |
| `nltk` | Tokenization and English stopwords. |
| `wordcloud` | Spam and non-spam word clouds. |
| `scikit-learn` | TF-IDF, clustering, and evaluation. |
| `jupyterlab` | Local notebook interface. |

The standard-library modules `re`, `string`, `itertools`, and `collections` need no separate installation.

The first notebook cell downloads the NLTK resources **`punkt`**, **`punkt_tab`**, and **`stopwords`**. Internet access is needed if those resources are not already installed.

## Project Workflow

### 1. Load and Explore the Data

The notebook loads both CSV files, removes `Unnamed` columns, and visualizes training label counts with bar and pie charts. It also creates separate word clouds for raw spam and non-spam text.

### 2. Clean the Text

The same text transformations are applied to training and external test messages:

1. Replace missing message bodies with empty strings.
2. Convert text to lowercase.
3. Normalize whitespace and trim leading and trailing spaces.
4. Remove digit sequences.
5. Tokenize using NLTK's `word_tokenize`.
6. Remove English stopwords and tokens matching entries in `string.punctuation`.
7. Identify the ten most frequent remaining words in the training corpus and remove them from both datasets.
8. Join tokens back into strings and normalize whitespace again.

The top-20 word chart uses counts calculated **before** the ten most common words are removed. `PorterStemmer` is imported, but stemming is not performed. The notebook does not explicitly remove URLs or HTML.

### 3. Extract TF-IDF Features

The vectorizer uses:

```python
TfidfVectorizer(
    ngram_range=(1, 5),
    smooth_idf=False,
    max_features=1000,
    sublinear_tf=False
)
```

It considers word n-grams containing one to five words and keeps up to 1,000 features. The vectorizer is fitted on training text and reused to transform external test text.

The saved feature matrix shapes are:

- Training: **`(13153, 1000)`**.
- External test: **`(5638, 1000)`**.

`CountVectorizer` is imported but not used.

### 4. Fit Clustering Models

| Model | Main settings |
| --- | --- |
| MiniBatchKMeans | Two clusters, `init='k-means++'`, `random_state=42`, `tol=1e-4`. |
| KMeans | Two clusters, random initialization, ten initializations, Lloyd algorithm, 300 maximum iterations, `random_state=42`. |
| BIRCH | Two final clusters, `threshold=0.01`. |
| Spectral Clustering | Two clusters; no fixed random seed is specified. |
| Gaussian Mixture | Two components, diagonal covariance, `random_state=42`; uses dense features. |

### 5. Interpret Cluster IDs

Cluster IDs are arbitrary: cluster `0` does not automatically mean non-spam. For KMeans, BIRCH, Spectral Clustering, and Gaussian Mixture, the notebook compares training accuracy under the original IDs and the reversed IDs (`1 - cluster_id`). It chooses the better mapping.

The KMeans and BIRCH exports apply the selected mapping to external predictions. MiniBatchKMeans is evaluated and exported with its original cluster IDs, without this mapping check.

### 6. Evaluate and Export

The notebook prints classification reports on the **same training messages used to fit the clusters**. It compares spam-class precision in a bar chart and plots a MiniBatchKMeans confusion matrix.

Three models produce external test CSV files. Spectral Clustering and Gaussian Mixture do not have external prediction export sections in the supplied notebook.

## Models and Results

These values come from **saved notebook outputs** and have not been independently reproduced. Accuracy is approximate because the stored classification reports round it to two decimal places.

| Model | Training Accuracy | Spam Precision | Spam Recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: |
| MiniBatchKMeans | Approximately **81%** | **0.66** | 0.97 | **0.79** |
| KMeans, after label mapping | Approximately 77% | 0.61 | **0.99** | 0.76 |
| BIRCH, after label mapping | Approximately 77% | 0.63 | 0.86 | 0.73 |
| Spectral Clustering | Approximately 63% | 0.00 | 0.00 | 0.00 |
| Gaussian Mixture, after label mapping | Approximately 75% | 0.62 | 0.77 | 0.69 |

### Interpretation

- **MiniBatchKMeans** has the highest displayed spam precision, spam F1, and overall accuracy among these saved experiments. Its reported weighted precision is **0.86**, which differs from spam-only precision of **0.66**.
- **KMeans** has the highest displayed spam recall after reversing its cluster IDs. Before that mapping, its displayed training accuracy is approximately 23%.
- **Spectral Clustering** has spam precision, recall, and F1 rounded to zero in the saved report.

These are **training-set results**. They do not establish performance on unseen messages or represent a verified competition score.

## Outputs

### Prediction Files

| File | Model |
| --- | --- |
| `miniKmeans_Submission.csv` | MiniBatchKMeans, original cluster IDs. |
| `Kmeans_Submission.csv` | KMeans, with the selected training-derived label mapping. |
| `Birch_Submission.csv` | BIRCH, with the selected training-derived label mapping. |

All three files use the same column structure:

```csv
ID,Label
```

The output `ID` comes from the input test column `id`. Each row corresponds to an external test message, and the CSV is written without a DataFrame index. Confirm cluster meanings before interpreting the exported labels as `0 = non-spam` and `1 = spam`.

### Visualizations

- ## Class distribution bar and pie charts.
  <img width="389" height="409" alt="image" src="https://github.com/user-attachments/assets/340d6e36-f303-49cd-81f3-ff45e6fbe893" />
  
- ## Spam and non-spam word clouds.
  <img width="819" height="819" alt="image" src="https://github.com/user-attachments/assets/ef6ba615-670e-499c-8372-c99d6e6b9c29" />
  <img width="819" height="819" alt="image" src="https://github.com/user-attachments/assets/de42489a-c6cf-4812-8dba-39844b3e4956" />
- ## Top-20 word frequency chart.
  <img width="1058" height="552" alt="image" src="https://github.com/user-attachments/assets/7b45c3b5-1d72-4c3b-89ac-c70fc1dd4d3d" />
- ## Spam precision comparison across five models.
  <img width="984" height="583" alt="image" src="https://github.com/user-attachments/assets/268aaca9-0d28-4251-8fa2-6760e12adc87" />
- ## MiniBatchKMeans confusion matrix.
  <img width="567" height="384" alt="image" src="https://github.com/user-attachments/assets/31e5cd55-9aed-4734-ad22-b067a43c0099" />


Plots are displayed in the notebook rather than explicitly saved as image files. Models and the TF-IDF vectorizer are not serialized.

## Current Limitations

- **No held-out evaluation:** clustering and reporting use the same labeled training corpus. A separate evaluation set is needed to estimate generalization.
- **Label mapping uses training labels:** reported classification scores depend on interpreting clusters with known labels. The selected mapping must be retained for future predictions.
- **MiniBatchKMeans mapping is unchecked:** its original IDs happen to align with labels in the saved output, but that alignment is not guaranteed across runs or datasets.
- **Spectral Clustering typo:** `yht_sc` is undefined and can stop execution when the flip branch is entered.
- **Resource-intensive experiments:** Spectral Clustering can construct a large affinity representation, and Gaussian Mixture explicitly densifies text features.
- **Potential loss of useful signals:** digit removal and removal of the ten most common words can discard information useful for spam detection. These choices are not tested against alternatives.
- **Missing-text handling:** empty or fully filtered messages may produce zero TF-IDF vectors; their frequency and impact are not analyzed.
- **Partial reproducibility:** Spectral Clustering and word clouds lack explicit seeds, and package versions are not pinned.
- **Limited submission coverage:** only MiniBatchKMeans, KMeans, and BIRCH export external predictions. No saved leaderboard score is provided.

## Future Improvements

- Create a stratified labeled validation split before learning common-word filters and TF-IDF vocabulary.
- Fit preprocessing and clustering on the training subset, retain a cluster-to-class mapping, and evaluate on unseen labeled messages.
- Add supervised baselines such as Multinomial Naive Bayes, Logistic Regression, and a linear SVM.
- Compare word n-grams with character n-grams and test different vocabulary sizes.
- Test whether preserving numbers, URLs, and common words improves results.
- Explore dimensionality reduction before resource-intensive clustering methods.
- Track zero-vector messages, duplicates, and preprocessing quality.
- Report exact accuracy, spam precision, recall, F1, and confusion matrices for every model.
- Save the fitted vectorizer, model, and label mapping for reusable inference.
- Fix random seeds, record working dependency versions, and document the dataset source and license.

## Project Status

**Experimental NLP and clustering notebook.** Text cleaning, TF-IDF extraction, five clustering comparisons, visualizations, and three CSV exports are implemented. Independent validation and consistent cluster-to-class mapping are the main next steps.
