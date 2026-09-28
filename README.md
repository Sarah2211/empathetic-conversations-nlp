# Empathetic Conversations: Predicting and Generating Emotion-Aware Dialogue

Two-part NLP project on empathetic conversations about news articles, using the WASSA 2024 conversation dataset (Track 2: turn-level emotion, emotional polarity and empathy).

* **Part 1: Prediction.** Given a conversation turn, predict its emotion intensity, emotional polarity and empathy, comparing four approaches from classic embeddings to a fine-tuned transformer and LLM prompting.
* **Part 2: Generation.** Given the first 5 turns of a conversation, generate the next 10, comparing a retrieval-based chatbot with few-shot prompting of an LLM.

Course project for **CS 421: Natural Language Processing**, University of Illinois Chicago (Fall 2025).

---

## Part 1: Emotion, polarity and empathy prediction

Each turn is scored on three targets at once, so every model is **multi-task**: two regression heads (emotion intensity and empathy, trained with MSE loss) and one classification head (emotional polarity, trained with cross-entropy).

| # | Approach | Key details |
|---|---|---|
| 1 | **Feed-forward ANN** on sentence embeddings | Averaged GloVe word vectors, two hidden ReLU layers with dropout, three output heads |
| 2 | **BiLSTM** | GloVe-initialized embedding layer, BiLSTM encoder, shared layers plus three heads, two-stage hyperparameter grid search |
| 3 | **Fine-tuned DeBERTa-v3-base** | Shared transformer encoder with three task heads, custom multi-task Hugging Face `Trainer`, **Optuna** hyperparameter search, FP16 training |
| 4 | **Zero-shot LLM prompting** | ChatGPT asked to score 5 dev conversations (50 turns), compared manually against gold labels |

### Results (dev set)

| Model | Emotion MAE ↓ | Empathy MAE ↓ | Polarity accuracy ↑ | Polarity F1 ↑ |
|---|---|---|---|---|
| ANN (GloVe) | 0.539 | 0.792 | 0.763 | 0.740 |
| BiLSTM* | 0.394 | 0.636 | 0.760 | 0.733 |
| **DeBERTa-v3 (Optuna-tuned)** | **0.472** | **0.671** | **0.867** | **0.866** |

\*The BiLSTM was validated on a curated 50-turn dev subset rather than the full dev set, so its scores are not directly comparable with the other two models.

**Takeaway:** the fine-tuned transformer was clearly best at polarity classification (+10 points of accuracy over the ANN), showing the value of contextual representations over averaged word vectors. The regression targets, especially empathy, stayed hard for every model, likely because empathy ratings are subjective and depend on the wider conversation, not just a single turn.

## Part 2: Dialogue generation

| Approach | How it works |
|---|---|
| **Retrieval-based chatbot** | Embeds the conversation history with `all-MiniLM-L6-v2` and retrieves the best training response using a weighted score: text similarity (0.5), emotion similarity (0.2), empathy similarity (0.2) and polarity similarity (0.1), without repeating responses |
| **Few-shot in-context learning** | Prompts `microsoft/Phi-3-mini-4k-instruct` (3.8B) with 3 or 5 example conversations, generates turns 6 to 15 with nucleus sampling (temperature 0.7, top-p 0.9), then parses the output with regex |

Generated responses were also scored with the Part 1 emotion, empathy and polarity models to compare how emotionally appropriate each method's responses were (`outputs/q3_analysis_with_predictions.csv`).

### Results (dev set)

| Method | ROUGE-1 | ROUGE-L | BLEU | BERTScore-F1 |
|---|---|---|---|---|
| **Retrieval chatbot** | **0.132** | **0.109** | **0.009** | **0.851** |
| Phi-3 ICL, 3-shot (30 conversations) | 0.086 | 0.067 | 0.000 | 0.666 |
| Phi-3 ICL, 5-shot (30 conversations) | 0.065 | 0.052 | 0.000 | 0.516 |

**Takeaway:** retrieval won on every automatic metric because it returns real human responses that sit close to the reference in meaning. The LLM produced more free-form replies, but lexical metrics punish that, and some generations came back empty or failed to parse, which pulled BERTScore down. Adding more examples did not help: 5-shot scored lower than 3-shot on the evaluation subset, even though the final test generations used the 5-shot prompt.

## Repository structure

```
├── part1_emotion_empathy_prediction/
│   ├── 01_ann_glove.ipynb
│   ├── 02_bilstm_multitask.ipynb
│   ├── 03_deberta_multitask.ipynb
│   ├── 04_llm_zero_shot_prompting.txt   # prompts and ChatGPT outputs
│   ├── predictions/                     # test and dev predictions for each model
│   └── figures/                         # training progress and hyperparameter search results
└── part2_dialogue_generation/
    ├── 01_retrieval_chatbot.ipynb
    ├── 02_few_shot_icl_phi3.ipynb
    └── outputs/                         # generated dialogues + emotion analysis
```

## Running it

The notebooks were built for **Google Colab** (a T4 GPU is enough).

1. Get the WASSA 2024 Track 2 files (`trac2_CONVT_train.csv`, `trac2_CONVT_dev.csv`, `trac2_CONVT_test.csv`). The data is not included in this repo.
2. Upload them to Colab or Google Drive and update the file paths at the top of each notebook.
3. Main packages: `torch`, `transformers`, `datasets`, `accelerate`, `optuna`, `gensim`, `sentence-transformers`, `scikit-learn`, `rouge-score`, `bert-score`, `evaluate`, `nltk`.

## Tech stack

Python · PyTorch · Hugging Face Transformers · Optuna · Sentence-Transformers · GloVe (Gensim) · Phi-3-mini · ROUGE / BLEU / BERTScore · Google Colab

## Team

**Sarah Zahir Syeda** and _[teammate names]_

**My contributions:** _[Fill in, e.g. "Built the DeBERTa multi-task model with Optuna tuning and the retrieval chatbot."]_
