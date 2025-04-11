# Climate Policy Extractor: Technical Report

**AUTHOR:** 
Bilal Hashim

**DATE:** 
11/04/25

## 1. Introduction
The Climate Policy Extractor project aims to build a scalable, transparent, and accurate pipeline for extracting official 2030 emissions reduction targets from Nationally Determined Contribution (NDC) documents. These documents are rich, multilingual, and often structurally inconsistent, posing challenges for information retrieval. This project leverages a Retrieval-Augmented Generation (RAG) pipeline incorporating document annotation, semantic embeddings, vector search, and controlled prompt engineering to perform precise, context-aware extraction of country-level climate policy targets.


## 2. Document Analysis and Annotation
All documents were stored in a PostgreSQL database and parsed into chunked form using a structured PDF parsing workflow. The parsed output included not just raw text but associated metadata such as page number, bounding box coordinates, file origin, and inferred language(s).

Chunk-level metadata was extended to include:

page_number (for visual annotation)

coordinates (used for bounding box drawing)

languages (to support multilingual filtering)

w2v_cluster_labels and climate_bert_cluster_labels (manually verified topic labels)

Annotation methodology involved:

Validating each chunk for well-formedness, eliminating noise and formatting artifacts

Manual screening of clusters to identify "gold standard" categories (e.g. "GHG targets", "Policy statements")

Quality assurance via visual inspection and embedding similarity validation

Annotation of over 12,000 chunks with consistent metadata including visual anchor points for later PDF highlighting




## 3. Embedding Generation and Comparison
Two families of embeddings were tested:

Classic embeddings: Word2Vec-based centroids of chunk tokens

Transformer embeddings: Sentence-transformers (all-mpnet-base-v2) and domain-specific ClimateBERT

Exploratory analysis revealed:

Word2Vec clusters were semantically coherent but underpowered for fuzzy or cross-lingual queries

Transformer embeddings captured richer semantics and performed better at retrieving contextual but lexically divergent policy targets

We used pairwise cosine similarity, visual cluster validation (t-SNE/UMAP), and retrieval success rate to evaluate performance. Final implementation used ClimateBERT embeddings stored in PGVector for robust similarity search.

## 4. Information Extraction System
A RAG pipeline was implemented with the following steps:

Query embeddings were generated for the core research question: “What is the official 2030 emissions reduction target for [country]?”

Vector search retrieved the top 10–15 most semantically relevant chunks per country

Prompt engineering constructed tightly scoped prompts instructing the LLM to answer only using the retrieved chunks, explicitly disallowing external assumptions

Meta-LLaMA 3.1 70B was integrated via Nebius API with temperature and top_k fine-tuned to balance stability and creativity

Chunk-level visual tracing was enabled using bounding box highlights rendered onto the source PDFs using PyMuPDF

Where chunk quality was low, a second retrieval loop was introduced, expanding the number of retrieved chunks and loosening the filtering thresholds.

## 5. Evaluation
Retrieval performance: 67 of 106 countries returned meaningful, structured emissions targets. A second pass increased this to 70+ countries.

LLM accuracy: Most extracted targets contained clear values (MtCO2e), conditionality (unconditional vs. conditional), and baseline scenarios (e.g. BAU or 1990 levels)

Citation precision: Most answers correctly linked to specific chunk IDs, allowing for PDF-based validation

Error cases: Misses were mostly due to:

Poor source formatting (e.g., non-structured scans)

Targets embedded in tables

Ambiguously worded or fragmented policy statements

Location precision: Verified via automatic bounding box rendering; multiple targets were successfully traced back to visual chunks

## 6. Discussion
This project demonstrates how semantic search and large language models can extract structured information from highly variable policy documents at scale. A few highlights:

The two-stage querying strategy improved recall significantly

Visual QA added trust and interpretability to the system

Embedding selection was crucial; ClimateBERT outperformed other models in both clustering and search tasks

Limitations:

Some countries lack clearly stated numeric targets, making extraction inherently ambiguous

Retrieval remains limited by the scope of chunking granularity and document quality


## 7. Conclusion
The Climate Policy Extractor successfully combines NLP, information retrieval, and LLMs to extract critical climate targets with high precision and traceability. Despite data quality and document heterogeneity challenges, the system proved robust, interpretable, and modular.

Next steps could include:

OCR enhancement for image-based PDFs

Multilingual transformer fine-tuning

LLM fine-tuning with human-verified targets

## 8. References
DS205 Week 7–10 Lab Materials

HuggingFace Transformers Documentation

PyMuPDF / Fitz documentation

Sentence-Transformers & ClimateBERT

PostgreSQL & PGVector docs

IPCC & UNFCCC documentation on NDC targets

