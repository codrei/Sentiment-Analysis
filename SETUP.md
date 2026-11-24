# How to run this project on another computer

## The four notebooks (run in this order)

| # | Notebook | What it is | GPU? |
|---|---|---|---|
| 1 | `RuleBased_Presentation.ipynb` | Present: data cleaning + rule-based approach (VADER → our improved rules). Creates `mcdo_reviews_clean.csv` and saves the rules to `model/` | no |
| 2 | `RuleBased_Review.ipynb` | Study: how the rules work, by hand, plus Q&A and quiz | no |
| 3 | `Semantic_Presentation.ipynb` | Present: fine-tuned RoBERTa + final comparison with the rule-based results | for training |
| 4 | `Semantic_Review.ipynb` | Study: meaning as vectors, fine-tuning, lesson map, Q&A, quiz | no |

Notebook 3 needs notebook 1 to have run first (it uses the cleaned data and the saved rule-based predictions).

## What to bring

Copy this whole folder (Google Drive or USB), **except `sentiment_env/`**. A virtual environment only works on the computer that created it, so you make a new one on each computer.

| File / folder | Needed for |
|---|---|
| `McDonald_s_Reviews.csv` | notebook 1 (it creates `mcdo_reviews_clean.csv` itself) |
| `requirements.txt` | setting up the environment |
| `model/rule_based.json`, `model/rule_based_test_predictions.csv` | notebooks 2-3 (created by notebook 1) |
| `model/roberta_binary/` (~480 MB) | skipping the transformer training on a PC without a GPU, and Part 2 of notebook 4 |

---

## Option A: Google Colab (easiest, free GPU)

1. Upload the folder to Google Drive, e.g. `My Drive/Sentiment Analysis/`.
2. In Drive, right-click the notebook > Open with > Google Colaboratory.
3. **Runtime > Change runtime type > T4 GPU.** Without this, the transformer trains on the CPU and takes about an hour.
4. In the first code cell, check that the folder path matches your folder name (`DRIVE_FOLDER` in the presentation notebooks, the `os.chdir(...)` line in the review notebooks).
5. Runtime > Run all. Colab asks for permission to access Drive. Allow it.

Colab already has PyTorch and transformers, and the first cell installs VADER automatically. Fine-tuning takes about 2-3 minutes on the T4.

---

## Option B: School PC with VS Code

You need Python 3.10 or newer, and VS Code with the **Python** and **Jupyter** extensions.

Open a terminal in VS Code (Terminal > New Terminal) inside this folder.

**Windows:**
```
python -m venv sentiment_env
sentiment_env\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name sentiment_env --display-name "Python (sentiment_env)"
```

**Mac / Linux:**
```
python3 -m venv sentiment_env
source sentiment_env/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name sentiment_env --display-name "Python (sentiment_env)"
```

If the school PC has an **NVIDIA GPU**, use `.../whl/cu126` instead of `.../whl/cpu`, then check with:
```
python -c "import torch; print(torch.cuda.is_available())"
```

Then open a notebook and choose the kernel **Python (sentiment_env)** (top-right corner).

### No GPU? Skip the long training

On a normal CPU, fine-tuning the transformer takes about 30-60 minutes. To avoid that:

1. Run `Semantic_Presentation.ipynb` once on Colab or your own PC. It saves the trained model to `model/roberta_binary/`.
2. Copy the `model/` folder to the school PC.
3. In the first code cell, set `TRAIN_TRANSFORMER = False`.

The notebook then loads the trained model instead of training it. It still takes about 10 minutes on a CPU (maybe 15-20 on an older PC), mostly because the transformer has to make predictions for ~5,700 validation and test reviews. **Run it before your presentation starts**, then present from the finished outputs. Opening the notebook without running it also shows all the saved results.

---

## Common problems

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Install Python from python.org and tick "Add Python to PATH" |
| `activate` blocked on Windows PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt instead |
| Kernel "Python (sentiment_env)" not in the list | Re-run the `ipykernel install` line, then reload VS Code |
| `FileNotFoundError` for a CSV | Check you opened the right folder (Colab: check `DRIVE_FOLDER`) |
| `ModuleNotFoundError: vaderSentiment` | `pip install vaderSentiment` inside the activated environment |
| Downloading the RoBERTa model fails | The first run downloads ~500 MB from huggingface.co. A school network may block it; use Colab or bring the `model/` folder |
| Results slightly different from the saved notebook | The semantic model can vary by a few tenths of a percent across hardware. The rule-based results are always identical |
