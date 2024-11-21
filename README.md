# Legal Clause Risk Analysis Tool

This project provides a tool for analyzing risks in legal clauses using a combination of:
- **Legal-BERT**: for legal-specific embeddings.
- **OpenAI GPT**: for detailed risk analysis.
- Additional tools for semantic similarity and correlation analysis.

The tool enables:
1. **Semantic similarity search** for relevant clauses from a database.
2. **Contextual and non-contextual risk analysis** using OpenAI GPT.
3. **Pearson correlation** between outputs to gauge similarity.

---

## Features
- **Legal-BERT embeddings**: Generate embeddings optimized for legal text.
- **Clause similarity search**: Find similar clauses using cosine similarity.
- **Risk analysis with GPT**: Perform risk analysis with or without contextual clauses.
- **Pearson correlation**: Compare outputs for deeper insight.

---

## Prerequisites
Make sure to install the following dependencies in your Python environment.

### Core Dependencies
The following libraries are required, along with their respective versions:
- `torch` == 2.2.2
- `transformers` == 4.39.3
- `sentence-transformers` == 3.1.0
- `scikit-learn` == 1.4.2
- `numpy` == 1.26.4
- `scipy` == 1.13.0
- `openai` == 1.51.2

### Environment Setup
The project is tested in the following Python environment:
- **Python**: 3.12.3

Other dependencies are listed below (from `conda list` output):
- `huggingface-hub` == 0.26.1
- `tqdm` == 4.66.5
- `filelock` == 3.13.4

---

## Installation
1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-repo/legal-clause-risk-analysis.git
   cd legal-clause-risk-analysis
