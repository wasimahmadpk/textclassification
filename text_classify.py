import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


# Loading TSV file
df_amazon = pd.read_csv ("datasets/amazon_alexa.tsv", sep="\t")

# Top 5 records
print(df_amazon.head())

# shape of dataframe
print(df_amazon.shape)

# View data information
df_amazon.info()

# Feedback Value count
print(df_amazon.feedback.value_counts())