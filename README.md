A tool for extracting climate policy information from National Determined Contributions (NDC) documents.
## Overview

This project provides a framework for analyzing climate policy information from NDC documents submitted to the UNFCCC.

The Climate Policy Extractor helps you:

1. **Collect National Determined Contributions (NDC) documents** from the [UNFCCC registry](https://unfccc.int/NDCREG)
2. **Extract and process text** from these documents using NLP techniques
3. **Generate embeddings** for document chunks to enable semantic search
4. **Build an information retrieval system** to extract specific climate policy information, such as emissions reduction targets

This tool aims to assist analysts in quickly finding and extracting relevant climate policy information from lengthy policy documents, making the assessment of climate commitments more efficient and accurate.

## Project Structure

```
climate-policy-extractor/
├── climate_policy_extractor/
│   ├── __init__.py
│   ├── items.py                       # Data models for scraped items
│   ├── settings.py                    # Scrapy settings
│   ├── pipelines.py                   # Processing pipelines
│   ├── spiders/
│   │   ├── __init__.py
│   │   └── ndc_spider.py              # Spider for scraping NDC documents
│   └── utils.py                       # Utility functions
├── notebooks/                         # Jupyter notebooks
├── data/
│   ├── pdfs/                          # Downloaded PDF documents
│   └── processed/                     # Processed document data
├── README.md                          # This file
├── CONTRIBUTING.md                    # Setup and contribution guidelines
├── REPORT.md                          # Technical report template
├── requirements.txt                   # Project dependencies
└── scrapy.cfg                         # Scrapy configuration
```

## Getting Started

For detailed setup instructions, including environment setup, database configuration, and troubleshooting tips, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Workflow

The project follows this general workflow:

1. **Data Collection**: Scrape NDC documents from the UNFCCC registry
2. **Document Processing**: Extract text and metadata from PDFs
3. **Text Analysis**: Analyze document structure and identify key sections
4. **Embedding Generation**: Create vector representations of text chunks
5. **Information Retrieval**: Build a system to extract specific climate policy information
6. **Evaluation**: Assess the accuracy and effectiveness of the extraction system

The included notebooks guide you through each step of this process.
