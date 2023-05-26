from transformers import pipeline
import pandas as pd
import os
import numpy as np
# visualizations 
from matplotlib import pyplot as plt
import seaborn as sns


def load_data():
    filename = os.path.join("in", "fake_or_real_news.csv")
    #docs = pd.read_csv(filename)["title"]
    data = pd.read_csv(filename, index_col=0)
    # turning the headlines into a list of strings 
    headlines = data['title'].astype(str).values.tolist()
    label = data['label']
    return filename, data, headlines, label

def classify(headlines):
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=False)
    emotion = classifier(headlines)
    return classifier, emotion

def emotion_score(headlines, emotion, label):
    # Create DataFrame with texts, predictions, labels, and scores
    df = pd.DataFrame(list(zip(headlines,label, emotion)), columns=['title', 'label', 'emotion'])
    df['emotion'] = df['emotion'].apply(lambda x: x['label'])
    return df


def visualize(df):
    # plotting in a bar graph 
    sns.countplot(x=df['emotion'])
    plt.savefig('out/emotions_for_ALL_headlines.png')
    df_fake = df[df['label'] == 'FAKE']
    sns.countplot(x=df_fake['emotion'])
    plt.savefig('out/emotions_for_FAKE_headlines.png')
    df_real = df[df['label'] == 'REAL']
    sns.countplot(x=df_real['emotion'])
    plt.savefig('out/emotions_for_REAL_headlines.png')