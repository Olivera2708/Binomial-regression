import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df['simple_user_need'] = np.where(df['user_need'].isin(['Functional', 'Contextual']),'FactContext', 'EmotionAction')
df.to_csv('data.csv')

first_100_rows = df.head(100)
first_100_rows.to_csv('small.csv', index=False)