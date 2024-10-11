import pandas as pd
import spacy
import textdescriptives as td
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_feature_entities_by_hand(doc):
    features = {
        "entity_number" : len(doc.ents),
        "entity_ratio" : len(doc.ents) / len(doc),
        "avg_entity_start" : sum([ent.start for ent in doc.ents]) / len(doc.ents) if len(doc.ents) > 0 else 0,
        "avg_entity_end" : sum([ent.end for ent in doc.ents]) / len(doc.ents) if len(doc.ents) > 0 else 0,
        "avg_entity_length" : sum([len(ent.text) for ent in doc.ents]) / len(doc.ents) if len(doc.ents) > 0 else 0
    }

    return features

def get_features_text_ent(articles, results):
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("textdescriptives/all")
    nlp.get_pipe('ner')
    tfidf_vectorizer = TfidfVectorizer()
    features = []
    
    for i, article in enumerate(articles):
        doc = nlp(article)
        dict = td.extract_dict(doc)
        del dict[0]['text']

        ents = [(e.text, "ent_" + e.label_) for e in doc.ents]
        ents_dict = {label: text for text, label in ents}

        print(ents_dict)

        filtered_values = [text for text in ents_dict.values() if text.strip()]
        if filtered_values:
            try:
                X_text = tfidf_vectorizer.fit_transform(filtered_values)  # Use filtered_values here
                final_dict = {label: X_text[i].toarray().flatten() for i, label in enumerate(ents_dict.keys())}
                averages_dict = {label: np.mean(vector) for label, vector in final_dict.items()}
                dict[0].update(averages_dict)

            except ValueError as e:
                print(f"Skipping article {i} due to ValueError: {e}")
                continue

        dict[0]["results"] = results[i]
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
