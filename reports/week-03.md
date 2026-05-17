# Week 3: Deep Learning and Transfer Learning for Emotion Detection

## 1. Executive Summary
In Week 3, we transitioned from classical Machine Learning algorithms (TF-IDF + Logistic Regression) to Deep Learning by applying **Transfer Learning**. We utilized a pre-trained Transformer model (`distilbert-base-uncased`) to perform multi-label classification across 28 emotion categories on the GoEmotions dataset. 

By utilizing a custom PyTorch training loop and specifically tracking the **Macro F1-score** to account for severe class imbalance, the optimal deep learning configuration achieved a **Peak Macro F1 of 0.4305**. This represents a significant improvement over the Week 2 baseline (~0.25) and demonstrates the power of contextual word embeddings in understanding complex internet language (Reddit comments).

## 2. Experimental Design & Architecture Choice
To scientifically evaluate the model's behavior and justify our architectural choices, we conducted three isolated experiments. All experiments utilized `BCEWithLogitsLoss` to handle the multi-label nature of the data.

### Experiment 1: Full Fine-Tuning (The Baseline DL)
* **Configuration:** Unfrozen backbone, Learning Rate = 2e-5, Epochs = 3.
* **Rationale:** Transformers require gentle updating of their attention weights to map general semantic knowledge (Wikipedia) to our highly specific domain (Reddit slang).
* **Result:** Successfully converged, reaching the highest performance.

### Experiment 2: Frozen Backbone (Linear Probing)
* **Configuration:** Base DistilBERT layers frozen (`requires_grad = False`); only the final classification head was trained. Learning Rate = 2e-5.
* **Rationale:** Testing if the pre-trained embeddings were already sufficient to separate 28 emotional boundaries using a single linear layer.
* **Result:** **Failed.** The Macro F1 score collapsed to **0.0195**. A simple linear classifier lacks the capacity to handle extreme class imbalance and complex multi-label boundaries without modifying the underlying feature representations.

### Experiment 3: Hyperparameter Stress Test (Aggressive LR)
* **Configuration:** Unfrozen backbone, Learning Rate = **5e-4**, Epochs = 3.
* **Rationale:** Demonstrating how learning rate tuning impacts gradient stability.
* **Result:** **Failed (Overfitting/Instability).** The massive learning rate caused *catastrophic forgetting*. The model's weights collapsed, forcing it to blindly predict the majority class for all inputs. The Macro F1 score dropped to **0.0000** by the third epoch.

## 3. Results Comparison

| Experiment Setup | Final Train Loss | Final Val Loss | Peak Macro F1 | Peak Micro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Exp 1: Full Fine-Tuning** | 0.0763 | 0.0852 | **0.4305** | **0.5666** |
| **Exp 2: Frozen Backbone** | 0.1369 | 0.1322 | 0.0195 | 0.1292 |
| **Exp 3: High LR (5e-4)** | 0.1506 | 0.1515 | 0.0002 | 0.0016 |

*Note: While a Macro F1 of 0.43 might seem low in binary classification, it is a highly competitive result for a 28-class highly imbalanced dataset, bordering on the human-agreement threshold (~0.55).*

## 4. Model Limitations & Deep Error Analysis
A critical part of deploying AI is understanding its blind spots. Using a dynamic scanning script, we extracted the most severe misclassifications (highest absolute probability deviation) from the validation set.

**Case Study 1: The Subjectivity of Positive Emotions**
> **Context:** *"congrats, darling! that is a monumentally meaningful accomplishment. i'm both happy for and proud of you < 3"*
> * **Model Predicted:** Admiration, Gratitude
> * **Human Ground Truth:** Excitement, Joy, Pride
> * **Analysis:** The model is not entirely "wrong." It accurately detected high-energy positive sentiment. However, the exact boundary between *Admiration/Gratitude* and *Excitement/Joy/Pride* is highly subjective. This highlights the limitation of human-annotated datasets—different assessors label similar text differently.

**Case Study 2: Contextual Blind Spots**
> **Context:** *"it’s a soulless facsimile of what made b movies actually fun."*
> * **Model Predicted:** Amusement
> * **Human Ground Truth:** Admiration, Approval, Neutral
> * **Analysis:** The model latched onto the trigger word *"fun"* and heavily skewed towards *Amusement*. It failed to grasp the broader syntactic negation and sarcastic critique ("soulless facsimile").

## 5. Conclusion
Full fine-tuning of DistilBERT yields vastly superior results compared to both classical ML models and frozen-backbone approaches. The primary limitation moving forward is not the model architecture, but the inherent ambiguity of human emotion and severe class imbalance. In Week 4, we will serialize these optimal weights into a functional Streamlit application to demonstrate real-time user interaction.
