# Biodiversity Term Analysis with Neo4j and Embedding Similarity

This repository supports semantic analysis of biodiversity-related terminology within corporate transcripts using a Neo4j graph database and sentence embeddings. It provides a vector similarity-based pipeline to identify relevant text chunks, followed by statistical analyses to detect changes over time.

---

## 📁 Repository Structure

### `biodiversity_term_analysis.ipynb`

A Jupyter notebook that serves as the main analysis interface. It allows you to:
- Load semantic vector embeddings for glossary terms.
- Query Neo4j for transcript chunks semantically similar to the term "Biodiversity" or other custom terms.
- Apply similarity filtering across years or by fixed yearly indexes.
- Visualize changes in similarity using UMAP or cosine-based scatter plots.
- Optionally run DiD (Differences-in-Differences) regressions to evaluate shifts pre- and post-TNFD (Taskforce on Nature-related Financial Disclosures).

> 💡 Uses utility functions from `glossary_similarity.py`.

---

### `glossary_similarity.py`

This module provides the core logic for:
- Initializing Neo4j database drivers using Streamlit secrets or `.env` variables.
- Fetching and embedding glossary terms using [SentenceTransformers](https://www.sbert.net/).
- Performing vector-based semantic search in Neo4j using `db.index.vector.queryNodes`.
- Pre-filtering and refining relevant transcript chunks for a given term and year(s).
- Computing cosine similarities and optionally reducing dimensionality with UMAP.
- Visualizing similarity and deviation from glossary terms over time.
- Running term-specific regressions (e.g., `fit_and_compute_regression_by_terms`) to analyze treatment effects.

> Used both as a backend module and directly in notebooks.

---

### `differences_in_differences.py`

(Assumes presence of this file based on naming convention.)

This module implements Differences-in-Differences regression models for evaluating semantic changes over time. It's used to:
- Identify causal impacts of TNFD-related disclosures.
- Compare term usage across industries or sectors before and after TNFD events.
- Present p-values and R² values for treatment and control groups in clean, formatted outputs.

> Integrated into the notebook for downstream statistical analysis.

---

## 📦 Data Files

### `data/df_tfnd_glossary_2023_embedded.json`

- Contains glossary terms and their precomputed sentence embeddings.
- Used by `load_biodiversity_embed()` in `glossary_similarity.py` to retrieve the vector for "Biodiversity".
- Required for initializing vector search in Neo4j.

### Vector Indexes in Neo4j

- Neo4j must contain a `chunk_embeddings` index (or `chunk_embeddings_{year}` format) for fast vector similarity search.
- These indexes are queried using the term embedding to locate semantically relevant text chunks from transcripts.

---

## ⚙️ Dependencies

```bash
pip install -r requirements.txt