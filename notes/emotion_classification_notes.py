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
                        return_all_scores=True)
    emotion = classifier(headlines)
    return classifier, emotion

def emotion_score(headlines, emotion, label):
    anger = []
    disgust = []
    fear = []
    joy = []
    neutral = []
    sadness = []
    surprise = []

    # extract scores (as many entries as exist in pred_texts)
    for i in range(len(headlines)):
        anger.append(emotion[i][0].get("score"))
        disgust.append(emotion[i][1].get("score"))
        fear.append(emotion[i][2].get("score"))
        joy.append(emotion[i][3].get("score"))
        neutrneutral.append(emotion[i][4].get("score"))
        sadness.append(emotion[i][5].get("score"))
        surprise.append(emotion[i][6].get("score"))
    # Create DataFrame with texts, predictions, labels, and scores
    df = pd.DataFrame(list(zip(headlines,label, anger, disgust, fear, joy, neutral, sadness, surprise)), columns=['title', 'label', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    # adding a column for the highest rated emotion in a new column
    df['top_emotion'] = df[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].idxmax(axis=1)
    return df


def visualize(df):
    # plotting in a bar graph 
    df['top_emotion'].value_counts().plot(kind='barh')
    plt.savefig('out/emotions_for_ALL_headlines.png')
    df_fake = df[df['label'] == 'FAKE']
    df_fake['top_emotion'].value_counts().plot(kind='barh')
    plt.savefig('out/emotions_for_FAKE_headlines.png')
    df_real = df[df['label'] == 'REAL']
    df_real['top_emotion'].value_counts().plot(kind='barh')
    plt.savefig('out/emotions_for_REAL_headlines.png')


def main():
    # load the data
    filename, data, headlines, label = load_data()
    # create the classifer 
    classifer, emotion = classify(headlines)
    # label headlines as the top emotion 
    df = emotion_score(headlines, emotion, label)
    visualize(df)



if __name__=="__main__":
    main()