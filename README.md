# Seed-based Weakly Supervised NER
To better understand what is NER, we first need to understand what is **named entity**. A **named entity** is a word or a phrase that clearly identifies one item from a set of other items that are have similar attributes

---
**Requirements**
- `tqdm`
- `nltk==3.0`
- `SpaCy`
- `SciSpaCy`
- `pytorch>=1.0.0`
- `tensorboardX`

## Step 1: AutoPhrase Mining and Expansion
Expands the entity set by integrating **automatic phrase mining** and **dictionary matching** results. To be specific, candidate phrases are automatically added to the entity typeset for dictionary completion.

### Data Processing
The input corpus is generated from [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). We select the `"body text"` of nearly 30,000 documents and separate them such that one sentence in one line.

### [CatE](https://github.com/yumeng5/CatE)
You will need to first create a directory under datasets (e.g., `datasets/your_dataset`) and put two files in it:

- A text file of the corpus, e.g., `datasets/your_dataset/text.txt`. Note: When preparing the text corpus, make sure each line in the file is one document/paragraph.
- A text file with the category names/keywords for each category, e.g., datasets/your_dataset/topics.txt where each line contains the seed words for one category. You can provide arbitrary number of seed words in each line (at least 1 per category; if there are multiple seed words, separate them with whitespace characters). Note: You need to ensure that every provided seed word appears in the vocabulary of the corpus.

### Candidates
We created 5 topics as our seed entity types, which are `covid.txt, livestock.txt, viral_protein.txt, wildlife.txt`. In each file, we wrote 10 - 20 candidate entities that have shown in the original text. For example, in `covid.txt`, we have **covid-19, sars-cov, mers-cov**, etc. Note that it is not guaranteed that **CatE** will output a perfect expansion of entity types based on the given 10 to 20 mannually added seed entities for each topic, we also need to examine these output and sift out some more reasonable entities. For instance, in the expanded file via **CatE** `res_covid.txt`, category **covid-19**, we have entities such as `"wuhan", "mainland_china", "hubei_province"`, etc, which certainly makes no sense and should not be in this category.
### Seed-Dictionary Generation
After preprocessing the raw input text file, expanding based on given categories and cleaning up our expanded entity types, we can then build our seed dictionary which contains the mannually added seed entities and sifted-out expanded entities. We need to store the result into a **seed dictionary** so that we can do matching in the next step. 

**How to Determine if a word is NOUN:** Since all named entities can only be *NOUN*, we need to examine the candidates and exclude all other words (verb, adj, etc.). Here we use **PoS tagging** from `SpaCy`. Specifically, we load different models, pass in input data to these models, and then iterate the returned document object and check their `tag_` attribute (`'NN' in doc[i].tag_`). For those words whose entity does not match any from all models used, we simply exclude them from our results.

``` python
nouns = [d if 'NN' in d.tag_ for d in doc]
```
In `seed_generator.py`, we used the method just mentioned to exclude unnecessary entities and created a json object to store the mannually added entity types and the expanded entity types, and then save as `./datasets/covid/seed.txt`. 

---
## Step 2: Group Entities
Once we have the input data cleaned and seed dictionary ready, we can then start to import some **NER models** from `SpaCy` and `SciSpaCy`. Here are some models we used, 
- en_core_web_sm
- en_ner_bc5cdr_md
- en_ner_jnlpba_md
- en_ner_bionlp13cg_md
- en_ner_craft_md

The commands for installing these models are as below:
```bash
python3 -m spacy download en_core_web_sm
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bc5cdr_md-0.4.0.tar.gz
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_jnlpba_md-0.4.0.tar.gz
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_bionlp13cg_md-0.4.0.tar.gz
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_ner_craft_md-0.4.0.tar.gz
```

### Label Assignment
For these existing NER models, we simply needed to run them on the cleaned data. Note that the format of data is line by line, we therefore examine one line at one iteration. For the text which can be found a matching entity in the given model, simply add it to the output list. To avoid duplicates, we returned a set instead of a list. The detailed implementation is as below:
```python
def get_entity_list(model, data):
    output = []
    nlp = model.load()
    for datum in data:
        doc = nlp(datum)
        entity_and_label = set([(X.text, X.label_) for X in doc.ents])
        output.extend(entity_and_label)
    return set(output)
```

### Dictionary Matching
For the seed-dictionary though, instead of creating a customized NER model, we decided to do dictionary matching. As long as we found a noun that appears in one topic/entity_type, simply assign it with that label. Similar to other NER models, the `(text, label)` pair will be stored into a set and returned. The detailed implementation is as below:
```python
def label_based_on_seed(seed, data, model):
    nlp = model.load()
    seed_entities = set()
    for datum in data:
        doc = nlp(datum)
        for d in doc:
            if "NN" in d.tag_:
                entity_and_label = is_in_seed(d.text, seed)
                if entity_and_label is not None: 
                    seed_entities.add(entity_and_label)
    return seed_entities
```
Note that here we also need to determine if the input word is a **NOUN** using the method mentioned before.

### Entity & Label DataFrame
To better visualize the results, we created a dataframe for each model's output `(text, label)` pairs. The dataframe has three columns, `"Entity", "Label", "Model"`. For seed-dictionary, the model will be `"seed"`. By concating all dataframes, we then formed a dataframe including the labeling results of all models and our seed-dictionary. 


### Use Dict Matching Results to Train
With the matching results from all models used, we can then generate a training data with the following format.

`TRAIN_DATA = [(text, {"entities": [(start, end, label), ...]})]`

The detailed process is in [`training_data_generator.py`](https://github.com/yeelimtse/cs-245-project/blob/main/create-seed-model/training_data_generator.py), and the result file is in [`training_data.txt`](https://github.com/yeelimtse/cs-245-project/blob/main/create-seed-model/training_data.txt). In [`train_combined_model.ipynb`](https://github.com/yeelimtse/cs-245-project/blob/main/create-seed-model/train_combined_model.ipynb), we trained a **combined NER model** that includes all the entities and labels generated before. An annotation result using the combined model is in the figure.

![](./figures/annotation-res.png)

### IOB Tagging
For the training of the deep learning model (in step 3), we need not only the entities and labels pair, but also IOB tags. We need to separate each word to line by line format, together with tags and labels. e.g. `Micheal B-PERSON`.

---
## Step 3: Neural Networks and Typing System Construction
We separate the our produced labeled input into two parts. We add 3/4 of the pre-processed input to the existing CoNLL2003 training set (`covid.train`) and add the rest to the evaluating set (`eng.testa`). Both training and evaluating dataset contain half of the original CoNLL2003 data and half newly added COVID-19 related data

To run the model, you will need to first download [pre-trained GloVe embedding](http://nlp.stanford.edu/data/glove.6B.zip) into `./NER/data/` and unzip it.
### Models used
- LSTM
- BiLSTM
- LSTM+CRF
- BiLSTM+CRF

For config between LSTM/BiLSTM, change bidirectional option in [line 30 in models.py](https://github.com/yeelimtse/cs-245-project/blob/1dc22051b2ded72e658b7e64670915f2bfb4783d/NER/model.py#L30)

For training and evaluating models without CRF, run `python3 main.py --feature_extractor=lstm --use_crf=false --train_path=data/covid.train`.
Otherwise, run with `python3 main.py --feature_extractor=lstm --use_crf=true --train_path=data/covid.train`

Models will be saved in `./NER/data/model`


### Evaluation
Since it’s hard to have manually labeled ground truth labels for COVID-19 related data, 95% of the test dataset is from CoNLL2003 dataset while others are COVID-19 related ground truth. The COVID-19 related test data are stored in `./NER/data/test.txt` and we combined it into the `eng.testb`.
Following table shows the best performed model within 100 training epochs by F1 score.

Model|F1|
--|:--:   
LSTM|80.12 
LSTM+CRF|87.21
BiLSTM|88.55
BiLSTM+CRF|89.31

By comparing the performance of different models, we can see an improvement on precision, recall and F1 score when using bidirectional LSTM instead
of LSTM, with a CRF layer instead of without a CRF layer.