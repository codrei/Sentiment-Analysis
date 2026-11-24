# Sentiment Analysis of McDonald's Reviews: Rule-Based vs Semantic

CS Elective 2 project by Marco Andrei R. Belen.

We classify Google reviews of McDonald's stores in the US as **positive or negative** in two different ways, and compare them on the same test reviews:

1. **Rule-based**: a dictionary of words with sentiment scores plus fixed rules (negation, "but", intensifiers, phrases, patterns). Nothing is trained.
2. **Semantic**: a transformer (RoBERTa) that learns what a review means from labeled examples, fine-tuned on our reviews.

## Results

Test set of 2,854 reviews (1-2 stars = negative, 4-5 stars = positive):

| Approach | Test accuracy |
|---|---|
| Plain VADER (rule-based tool, no changes) | 79.6% |
| VADER + 27 hand-picked restaurant words | 83.2% |
| **Our rule-based system** | **88.6%** |
| **Semantic: fine-tuned Twitter-RoBERTa** | **94.6%** |

- The rule-based system is fast, needs no GPU, and every score can be traced back to the words and rules that caused it. It struggles with meaning that comes from context, sarcasm, and reviews with no typical sentiment words.
- The semantic model understands whole sentences, which makes it about 6 points more accurate, at the cost of needing labeled data, a GPU to train, and being hard to explain.
- About 110 test reviews are wrong for both. Many of them are reviews whose star rating doesn't match the text.

## Notebooks

Run them in this order:

| # | Notebook | What's in it |
|---|---|---|
| 1 | `RuleBased_Presentation.ipynb` | Data cleaning, then improving VADER step by step: hand-picked words, a dictionary built from the training reviews, phrase rules, pattern rules |
| 2 | `RuleBased_Review.ipynb` | Study notes: how VADER calculates a score, each rule explained with examples, Q&A |
| 3 | `Semantic_Presentation.ipynb` | Fine-tuning RoBERTa, and the final comparison with the rule-based results |
| 4 | `Semantic_Review.ipynb` | Study notes: meaning as vectors, transfer learning, how the training connects to the lessons, Q&A |

`model/rule_based.json` contains the complete rule-based system: every added word, phrase and pattern with its score, and the cut-off.

## How to run

See **[SETUP.md](SETUP.md)** for step-by-step instructions for Google Colab and for a local PC with VS Code.

Short version:

```
python -m venv sentiment_env
source sentiment_env/bin/activate          # Windows: sentiment_env\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**The data is not included in this repository.** Put `McDonald_s_Reviews.csv` (the McDonald's store reviews dataset from Kaggle, 33,396 reviews) in this folder before running notebook 1. The notebooks create everything else: the cleaned data, the saved rules, and the fine-tuned model (about 480 MB, also not included).

The rule-based notebooks run on any computer. Fine-tuning the transformer takes about 2-3 minutes on a GPU (for example Colab's free T4) and 30-60 minutes on a CPU.

## Data preparation, in short

- The CSV isn't UTF-8, so it's read as `latin-1`, and broken characters (`ï¿½`) are repaired.
- 3-star reviews are left out because they're usually mixed.
- About 5,000 reviews are Google's placeholder text for star-only ratings ("Excellent", "Terrible"…), so duplicates are removed (one copy per text) to keep the same review out of both the training and the test data.
- The data is split 70 / 15 / 15 into train, validation and test. All rules and settings were chosen with train and validation, and the test set was used once at the end.
