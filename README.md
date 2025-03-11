Information Extraction (IE) System

Overview

This project implements an Information Extraction (IE) System using spaCy to extract named entities and relationships from text data. The system identifies key entities (e.g., persons, organizations, dates) and their relationships (e.g., subject-verb-object dependencies) from input text.

Features

Named Entity Recognition (NER): Identifies entities like names, organizations, and dates.

Dependency Parsing: Extracts relationships between words in a sentence.

Interactive Output: Displays extracted information in a structured format.

Jupyter Notebook Implementation: Easy to test and extend.

Installation

# Clone the repository or copy the notebook files
# Install the required dependencies
pip install spacy pandas
python -m spacy download en_core_web_sm

Usage

import spacy
import pandas as pd

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_information(text):
    doc = nlp(text)
    entities = [{"Entity": ent.text, "Type": ent.label_} for ent in doc.ents]
    relationships = [{"Token": token.text, "Dependency": token.dep_, "Head": token.head.text} for token in doc if token.dep_ in ("nsubj", "dobj")]
    return entities, relationships

# Run the extraction on sample text
text = "Elon Musk is the CEO of Tesla. OpenAI was founded in 2015."
entities, relationships = extract_information(text)
print("Entities:", entities)
print("Relationships:", relationships)

Example Output

Named Entities:

| Entity     | Type  |
|------------|-------|
| Elon Musk  | PERSON |
| Tesla      | ORG  |
| OpenAI     | ORG  |
| 2015       | DATE |

Relationships:

| Token  | Dependency | Head  |
|--------|-----------|-------|
| Musk   | nsubj     | is    |
| CEO    | attr      | is    |
| OpenAI | nsubjpass | founded |
| 2015   | npadvmod  | founded |

Future Enhancements

Use Transformer-based models (e.g., en_core_web_trf for better accuracy).

Implement relationship visualization.

Extend with deep learning (BERT, GPT) for more advanced IE.
