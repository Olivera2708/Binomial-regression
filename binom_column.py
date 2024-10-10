import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df['simple_user_need'] = np.where(df['user_need'].isin(['Functional', 'Contextual']),'FactContext', 'EmotionAction')
df.to_csv('data.csv')

sampled_df = df.groupby('simple_user_need', group_keys=False).apply(lambda x: x.sample(n=min(4000, len(x))))
sampled_df = sampled_df.reset_index(drop=True)
sampled_df.to_csv('small_equal.csv', index=False)