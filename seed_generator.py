import en_ner_bc5cdr_md                 # to check if a word is noun
import json                             # to store entity type into file
nlp = en_ner_bc5cdr_md.load()           # load NLP
seed = {}                               # to store the result

# ========================== #
#    OPEN ENTITIES FILES     #
# ========================== #
print("Opening entities files...")
covid_file = open("./datasets/covid/res_covid.txt", "r")
livestock_file = open("./datasets/covid/res_livestock.txt", "r")
viral_protein_file = open("./datasets/covid/res_viral_protein.txt", "r")
wildlife_file = open("./datasets/covid/res_wildlife.txt", "r")


# ========================== #
#      GET RAW ENTITIES      #
# ========================== #
print("Getting raw data...")
covid_raw = covid_file.readlines()
livestock_raw = livestock_file.readlines()
viral_protein_raw = viral_protein_file.readlines()
wildlife_raw = wildlife_file.readlines()

# ========================== #
#    CLOSE ENTITIES FILES    #
# ========================== #
print("Closing entities files...")
covid_file.close()
livestock_file.close()
viral_protein_file.close()
wildlife_file.close()

def extract_category(raw_data, i):
    """
    Inputs:
        raw_data: the raw entity types data
        i: the index of one category
    Return:
        A list contains category and its expanded entities
    """
    category_text = raw_data[i]
    expanded_text = raw_data[i + 1].split(' ')
    entities = []

    category = category_text.split('(')[1].split(')')[0]
    expanded_entities = [str(entity) for expanded_word in expanded_text \
                                for entity in nlp(expanded_word) \
                                if "NN" in entity.tag_]
    expanded_entities.append(category)
    entities.extend(expanded_entities)
    return entities


def extract_entities(raw_data, entity_type, dict):
    """
    Inputs:
        raw_data: the raw entity types data
        entity_type: the ENTITY_TYPE
        dict: where to store ENTITIES
    Return: None
    """
    dict[entity_type] = []
    for i in range(0, len(raw_data), 2):
        dict[entity_type].extend(extract_category(raw_data, i))

# ========================== #
#       GENERATING SEED      #
# ========================== #
print("Generating seed...")
extract_entities(covid_raw, "COVID", seed)
extract_entities(livestock_raw, "LIVESTOCK", seed)
extract_entities(viral_protein_raw, "VIRAL_PROTEIN", seed)
extract_entities(wildlife_raw, "WILDLIFE", seed)

with open("./datasets/covid/seed.txt", 'w') as outfile:
    json.dump(seed, outfile)

print("Done generating seed.")