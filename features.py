import pandas as pd
import spacy
import textdescriptives as td


def get_features_text_ent(articles):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textdescriptives/all")
    features = []
    
    for article in articles:
        doc = nlp(article)
        dict = td.extract_dict(doc)
        del dict[0]['text']
        features.extend(dict)

    return pd.DataFrame(features)

def get_features_basic(articles):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textdescriptives/all")
    features = {
        'pos_prop_DET': [],
        'pos_prop_NOUN': [],
        'pos_prop_AUX': [],
        'pos_prop_VERB': [],
        'pos_prop_PUNCT': [],
        'pos_prop_PRON': [],
        'pos_prop_ADP': [],
        'pos_prop_SCONJ': [],
        'n_tokens': [],
        'n_unique_tokens': [],
        'n_characters': [],
        'n_sentences': [],
        'n_stop_words': []
    }
    
    for article in articles:
        doc = nlp(article)
        dict = td.extract_dict(doc)[0]
        features['pos_prop_DET'].append(dict['pos_prop_DET'])
        features['pos_prop_NOUN'].append(dict['pos_prop_NOUN'])
        features['pos_prop_AUX'].append(dict['pos_prop_AUX'])
        features['pos_prop_VERB'].append(dict['pos_prop_VERB'])
        features['pos_prop_PUNCT'].append(dict['pos_prop_PUNCT'])
        features['pos_prop_PRON'].append(dict['pos_prop_PRON'])
        features['pos_prop_ADP'].append(dict['pos_prop_ADP'])
        features['pos_prop_SCONJ'].append(dict['pos_prop_SCONJ'])
        features['n_tokens'].append(dict['n_tokens'])
        features['n_unique_tokens'].append(dict['n_unique_tokens'])
        features['n_characters'].append(dict['n_characters'])
        features['n_sentences'].append(dict['n_sentences'])
        features['n_stop_words'].append(dict['n_stop_words'])

    return pd.DataFrame(features)

# def get_features_basic(articles):
#     nlp = spacy.load("en_core_web_sm")
#     features = {
#         "word_count": [],
#         "char_count": [],
#         'sentence_count': []
#     }
    
#     for article in articles:
#         doc = nlp(article)

#         word_count = len(doc)
#         char_count = len(article)
#         sentence_count = len(list(doc.sents))

#         features["word_count"].append(word_count)
#         features["char_count"].append(char_count)
#         features["sentence_count"].append(sentence_count)
#         #entites

#     return pd.DataFrame(features)