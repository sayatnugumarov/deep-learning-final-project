# Week 1 Progress Report

**What was completed this week:**
* **Project Initialization:** Created the repository structure (`notebooks/`, `reports/`, `src/`) and finalized the project proposal.
* **Dataset Selection:** Confirmed the use of the Hugging Face `go_emotions` dataset (simplified version) for multi-label emotion classification.
* **Exploratory Data Analysis (EDA):** Developed a Python script to analyze the dataset's statistical properties, label distribution, and text characteristics.
* **Environment Setup:** Configured `requirements.txt` with necessary libraries: `pandas`, `matplotlib`, `seaborn`, and `datasets`.

**Experiments run:**
* Performed data loading and integrity checks on the training split.
* Calculated label cardinality (average number of labels per example).
* Visualized the frequency of each of the 28 emotion classes.
* Analyzed comment lengths to determine the distribution of input sequences.

**Results so far:**
* **Dataset Size:** Confirmed a robust training set of **43,410 samples**.
* **Multi-label Nature:** On average, each comment has **1.18 labels**, confirming that while most comments have one emotion, a significant portion has multiple, justifying the multi-label approach.
* **Class Imbalance:** Initial visualization shows a heavy skew. Frequent labels like 'neutral' and 'admiration' appear often, while others like 'grief' are extremely rare.
* **Sample Insights:** 
    * "My favourite food is anything I didn't have to cook myself." -> mapped to **neutral**.
* **Text Length:** Most comments are relatively short (under 20-30 words), which suggests that simple architectures or pretrained transformers should handle the context well.

**Problems or blockers:**
* **Imbalance Challenge:** The heavy presence of 'neutral' labels (as seen in sample data) might bias the model. I will need to focus on F1-Score rather than Accuracy in Week 2.
* **Multi-label Encoding:** The raw data uses ID lists; I've identified that `MultiLabelBinarizer` from `scikit-learn` will be necessary for the baseline.

**Plan for next week:**
* Implement the preprocessing pipeline (text cleaning and label binarization).
* Build a baseline model using TF-IDF and Logistic Regression (One-Vs-Rest).
* Establish the first performance metrics (Macro and Micro F1-scores).
