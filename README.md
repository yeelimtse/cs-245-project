# Seed-based Weakly Supervised NER
To better understand what is NER, we first need to understand what is **named entity**. A **named entity** is a word or a phrase that clearly identifies one item from a set of other items that are have similar attributes
---
**Requirements**
- `tqdm`
- `nltk==3.0` (from __future__ import print_function)
- pip3 install -U pluggy
- `pytorch>=1.0.0`
- `tensorboardX`

## Step 1: AutoPhrase Mining and Expansion
Expands the entity set by integrating **automatic phrase mining** and **dictionary matching** results. To be specific, candidate phrases are automatically added to the entity typeset for dictionary completion.

### Data Processing
The input corpus is generated from [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). We select the `"body text"` of nearly 30,000 documents and separate them such that one sentence in one line.

### Candidates
### Dictionary Matching
### Dictionary Expansion
**How to Determine if a word is NOUN:** 
- Since all named entities can only be *NOUN*, we need to examine the input data and exclude all other words (verb, adj, etc.). Here we use **PoS tagging** from `SpaCy`. Specifically, we load different models, pass in input data to these models, and then iterate the returned document object and check their `tag_` attribute (`'NN' in doc[i].tag_`). For those words whose entity does not match any from all models used, we simply exclude them from our results.

``` python
nouns = [d if 'NN' in d.tag_ for d in doc]
```
#### [CatE](https://github.com/yumeng5/CatE)
You will need to first create a directory under datasets (e.g., `datasets/your_dataset`) and put two files in it:

- A text file of the corpus, e.g., datasets/your_dataset/text.txt. Note: When preparing the text corpus, make sure each line in the file is one document/paragraph.
- A text file with the category names/keywords for each category, e.g., datasets/your_dataset/topics.txt where each line contains the seed words for one category. You can provide arbitrary number of seed words in each line (at least 1 per category; if there are multiple seed words, separate them with whitespace characters). Note: You need to ensure that every provided seed word appears in the vocabulary of the corpus.

---
## Step 2: Group Entities
concate?
### Models used
- en_core_sci_sm: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz`
- en_core_sci_md: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz`
- en_ner_bc5cdr_md: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz`
- en_ner_jnlpba_md: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_jnlpba_md-0.4.0.tar.gz`
- en_ner_bionlp13cg_md: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bionlp13cg_md-0.4.0.tar.gz`
- en_ner_craft_md: `pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_craft_md-0.4.0.tar.gz`


---
## Step 3: Neural Networks and Typing System Construction
You will need to first download [pre-trained GloVe embedding](http://nlp.stanford.edu/data/glove.6B.zip) into `./NER/data/` and unzip it.
### Models used
- LSTM
- BiLSTM
- LSTM+CRF
- BiLSTM+CRF

For config between LSTM/BiLSTM, change bidirectional option in [line 30 in models.py](https://github.com/yeelimtse/cs-245-project/blob/1dc22051b2ded72e658b7e64670915f2bfb4783d/NER/model.py#L30)

For training and evaluating models without CRF, run `python3 main.py --feature_extractor=lstm --use_crf=false`.
Otherwise, run with `python3 main.py --feature_extractor=lstm --use_crf=true`

Models will be saved in `./NER/data/model`