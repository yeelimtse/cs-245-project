# Seed-based Weakly Supervised NER
---
**Requirements**
- `tqdm`
- `nltk==3.0` (from __future__ import print_function)
- pip3 install -U pluggy
- 
## Step 1: AutoPhrase Mining and Expansion
Expands the entity set by integrating **automatic phrase mining** and **dictionary matching** results. To be specific, candidate phrases are automatically added to the entity typeset for dictionary completion.

### Data Processing
The input corpus is generated from [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). We select the `"body text"` of nearly 30,000 documents and separate them such that one sentence in one line.

### Candidates
### Dictionary Matching
### Dictionary Expansion
#### [CatE](https://github.com/yumeng5/CatE)
You will need to first create a directory under datasets (e.g., `datasets/your_dataset`) and put two files in it:

- A text file of the corpus, e.g., datasets/your_dataset/text.txt. Note: When preparing the text corpus, make sure each line in the file is one document/paragraph.
- A text file with the category names/keywords for each category, e.g., datasets/your_dataset/topics.txt where each line contains the seed words for one category. You can provide arbitrary number of seed words in each line (at least 1 per category; if there are multiple seed words, separate them with whitespace characters). Note: You need to ensure that every provided seed word appears in the vocabulary of the corpus.

---
## Step 2: Group Entities
concate?
### Models used
- en_core_sci_sm: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz`
- en_ner_bc5cdr_md: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz`

---
## Step 3: Neural Networks and Typing System Construction
TODO: part 3

---