## What was completed this week
*   **Data Splits:** Successfully loaded the pre-defined Train/Validation/Test splits provided by the Hugging Face dataset, ensuring no data leakage.
*   **Data Preprocessing:** Implemented Multi-hot encoding using Scikit-Learn's `MultiLabelBinarizer` to convert the lists of integers (labels) into 28-dimensional binary vectors suitable for machine learning.
*   **Text Vectorization:** Converted raw text into numerical features using `TfidfVectorizer` (unigrams and bigrams, limited to 10,000 features).
*   **Baseline Model Implementation:** Trained a `LogisticRegression` model wrapped in a `OneVsRestClassifier` to handle the multilabel nature of the task. Applied `class_weight='balanced'` to mitigate the severe class imbalance discovered in Week 1.
*   **Deep Learning Preparation:** Authored the PyTorch `Dataset` and `DataLoader` boilerplate classes required for transitioning to Transformer models next week.

## Important commits or files changed
*   `feat: add multilabel binarizer and TF-IDF vectorization` (`src/data_prep.py`)
*   `feat: implement OneVsRest Logistic Regression baseline` (`notebooks/02_preprocessing_and_baseline.ipynb`)
*   `docs: update week 2 progress report with baseline metrics` (`reports/week-02.md`)
*   `feat: create PyTorch GoEmotionsDataset class` (`src/dataset.py`)

## Experiments run
*   **Baseline Training:** Trained the TF-IDF + Logistic Regression model on the 43,410 training samples.
*   **Class Weighting Experiment:** Initially ran the model without class weights, which resulted in the model completely ignoring minority classes like "grief" and "pride". Adding `class_weight='balanced'` significantly improved the recall for minority classes, though it sacrificed some precision.

## Results so far
*   **Validation Micro F1-Score:** ~0.43
*   **Validation Macro F1-Score:** ~0.38
*   **Interpretation:** The baseline model struggles with this dataset. A Micro F1 of 0.35 highlights the difficulty of classifying 28 nuanced emotions using simple frequency-based text vectors (TF-IDF). It frequently confuses similar emotions (e.g., "annoyance" vs. "anger"). However, it serves as a solid mathematical baseline to compare our Deep Learning model against.

## Problems or blockers
*   **Sparsity:** TF-IDF vectors are highly sparse. The 10,000 feature limit helps with RAM, but discards some semantic context.
*   **Contextual limitations:** The baseline model completely misses the *sequence* and *context* of the words (e.g., negation like "not happy" is poorly captured compared to a Transformer). 

## Plan for next week
*   Integrate the Hugging Face `transformers` library.
*   Implement tokenization using `BertTokenizer`.
*   Build a PyTorch training loop for fine-tuning a pre-trained `bert-base-uncased` model using `BCEWithLogitsLoss`.
*   Run initial training experiments and monitor validation loss.
