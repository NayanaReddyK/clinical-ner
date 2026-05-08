# 🏥 Clinical NER — Named Entity Recognition from Patient Notes

> Automatically extract **Diseases**, **Drugs**, and **Dosages** from unstructured clinical text using a BiLSTM-CRF deep learning model.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![NLP](https://img.shields.io/badge/Domain-Clinical%20NLP-green)
![Model](https://img.shields.io/badge/Model-BiLSTM--CRF-orange)

---

## 📌 Overview

Electronic Health Records (EHRs) contain vast amounts of unstructured clinical text — doctor's notes, discharge summaries, radiology reports — where critical medical information is buried in free-form language.

This project builds a **complete end-to-end NLP pipeline** to extract three entity types from raw clinical notes:

| Entity | Tag | Example |
|---|---|---|
| Disease / Condition | `DISEASE` | *type 2 diabetes*, *hypertension*, *pneumonia* |
| Drug / Medication | `DRUG` | *metformin*, *lisinopril*, *amoxicillin* |
| Dosage / Strength | `DOSAGE` | *500mg*, *10mg twice daily*, *2 tablets* |

---

## 🎯 Demo

![Gradio Demo](assets/demo_screenshot.png)

> Type any clinical note → entities are highlighted in colour instantly.

Run the Gradio interface locally:
```bash
# After running the notebook, the last cell launches:
demo.launch(share=True)  # generates a public link
```

---

## 🧠 Model Architecture

```
Input Tokens → Embedding (128d) → BiLSTM (2-layer, 256 hidden) → Dropout → Linear → CRF → BIO Tags
```

**Why BiLSTM-CRF?**
- The **BiLSTM** captures left and right context for each token simultaneously
- The **CRF layer** enforces valid BIO tag transitions at decode time — preventing impossible sequences like `I-DRUG` without a preceding `B-DRUG`
- This is the **gold standard architecture** for sequence labeling NER tasks

---

## 🔧 NLP Pipeline

| Step | Technique | Clinical Consideration |
|---|---|---|
| Text Cleaning | Regex + abbreviation expansion | Expands `Pt`, `Hx`, `BID`, `PRN`, `IV` etc. |
| Tokenization | NLTK word tokenizer | Word-level for BIO tag alignment |
| Normalization | Unicode + selective lowercasing | Preserves `HIV`, `MRI`, `INR` |
| Stopword Removal | NLTK + clinical keep-list | Skipped for NER; used for encoding tasks |
| Lemmatization | WordNet Lemmatizer | Preferred over stemming — preserves clinical vocab |
| POS Tagging | NLTK averaged perceptron | Confirms entity-dense, noun-heavy clinical text |
| Dependency Parsing | spaCy | Extracts drug → disease relationship pairs |
| Encoding | BoW, TF-IDF, learned embeddings | Progression from sparse to dense representations |
| NER Model | BiLSTM-CRF (2-layer) | Sequence labeling with valid tag constraints |
| Evaluation | seqeval entity-level F1 | Span-based metric — correct for NER |

---

## 📁 Project Structure

```
clinical-ner-bilstm-crf/
│
├── Clinical_NER_NLP_Project.ipynb   # Main notebook (all sections)
├── clinical_ner_model.pt            # Saved model weights
├── requirements.txt                 # Dependencies
├── assets/
│   └── demo_screenshot.png          # Gradio interface screenshot
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/clinical-ner-bilstm-crf.git
cd clinical-ner-bilstm-crf
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the notebook
Open `Clinical_NER_NLP_Project.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

### 4. Launch the Gradio demo
The last cell in the notebook launches the interactive interface automatically.

---

## 📦 Requirements

```
torch>=2.0
pytorch-crf
nltk
spacy
seqeval
gradio
scikit-learn
matplotlib
seaborn
wordcloud
```

Or install all at once:
```bash
pip install torch pytorch-crf nltk spacy seqeval gradio scikit-learn matplotlib seaborn wordcloud
```

---

## 📊 Results

Evaluated using **seqeval** entity-level F1 (span-based — not inflated token accuracy):

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| DISEASE | — | — | — |
| DRUG | — | — | — |
| DOSAGE | — | — | — |

> ⚠️ Fill in your actual scores after running. Results are on a synthetic 80-sentence dataset — see Future Work for the path to production accuracy.

---

## 🔭 Future Work

1. **ClinicalBERT / BioBERT fine-tuning** — Replace trained embeddings with `dmis-lab/biobert-v1.1` for significantly higher F1 on real clinical text
2. **Larger datasets** — Fine-tune on [i2b2 2010](https://www.i2b2.org) or [BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr) (freely available)
3. **More entity types** — `SYMPTOM`, `PROCEDURE`, `LAB VALUE`, `BODY PART`
4. **Relation extraction** — Identify *drug treats disease*, *drug causes adverse effect* pairs
5. **Negation detection** — Handle *"no signs of pneumonia"* vs *"pneumonia"*
6. **API deployment** — Wrap as a FastAPI microservice to integrate with EHR systems

---

## 🏢 Real-World Applications

- Auto-populate structured EHR fields from free-text notes
- ICD-10 coding automation for insurance claims processing
- Adverse drug reaction mining from clinical narratives
- Clinical trial cohort identification

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

*Built as part of an NLP course project. Domain: Clinical / Biomedical NLP.*
