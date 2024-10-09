import pandas as pd
import spacy

def get_features(articles):
    nlp = spacy.load("en_core_web_sm")
    features = {
        "word_count": [],
        "char_count": []
    }
    
    for article in articles:
        doc = nlp(article)

        word_count = len(doc)
        char_count = len(article)

        features["word_count"].append(word_count)
        features["char_count"].append(char_count)

    return pd.DataFrame(features)
