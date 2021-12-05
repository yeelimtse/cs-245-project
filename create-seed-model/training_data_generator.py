import json

dict_matching_res = json.load(open("../datasets/covid/dict-matching-res.txt", 'r'))
labels = json.load(open("../datasets/covid/labels.txt", 'r'))
labels = set(labels["LABEL"])
# TRAIN_DATA = [(text, {"entities": [(start, end, label), ...]})]

def generate_entities(doc):
    entities = doc['entities']
    entities_dict = {}
    entities_list = []
    for ent in entities:
        if ent['type'] in labels:
            entities_list.append((ent['start'], 
                                  ent['end'], 
                                  ent['type']))
    entities_dict['entities'] = entities_list
    return entities_dict

def generate_one_doc(doc):
    text = doc['title'] + doc['abstract'] + doc['body']
    entities_dict = generate_entities(doc)
    return (text, entities_dict)

def generate_training_data(docs):
    res = []
    for doc in docs:
        res.append(generate_one_doc(doc))
    return res

training_data = generate_training_data(dict_matching_res)
with open("./training_data.json", 'w') as outfile:
    json.dump(training_data, outfile)