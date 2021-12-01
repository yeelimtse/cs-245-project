text_file = open("./datasets/covid/phrase_text.txt", 'r')
data = text_file.readlines()
text_file.close()
cleaned_text = open('./datasets/covid/cleaned_text.txt', 'w')

def remove_reference(datum):
    splitted = datum.split(' ')[:-1]
    not_ref_flag = True
    cleaned_datum = []
    for w in splitted:
        if w == '[':
            not_ref_flag = False
        if w == ']':
            not_ref_flag = True
            continue
        if not_ref_flag:
            cleaned_datum.append(w)
    output = ' '.join(cleaned_datum) + '\n'
    return output

cleaned_data = [remove_reference(datum) for datum in data]
cleaned_text.writelines(cleaned_data)
cleaned_text.close()