import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omvw-1.4')

df = pd.read_csv(r'c:\Users\Dell\oneDrive\Desktop\Data_science\NPL\spam.csv', encoding='latin-1')

df.drop(columns=['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'], inplace=True)

print(df.head())
print(df.info())

df.rename(columns={
    'v1': 'label',
    'v2': 'message'
}, inplace=True)

print(df.head())
print(df['label'].value_counts)

df["label"].value_counts().plot(kind= 'bar')

plt.title("spam vs ham message")
plt.xlabel("label")
plt.ylabel("count")

plt.show()
