from transformers import pipeline
import pandas as pd
import os
import numpy as np
# visualizations 
from matplotlib import pyplot as plt
import seaborn as sns
# using functions from utils folder 
# to be able to use utils from the folder 
import sys
sys.path.append("utils")
from langutils import load_data
from langutils import classify
from langutils import emotion_score
from langutils import visualize




def main():
    # load the data
    filename, data, headlines, label = load_data()
    # create the classifer 
    classifer, emotion = classify(headlines)
    # label the headlines with top emotion
    df = emotion_score(headlines, emotion, label)
    visualize(df)



if __name__=="__main__":
    main()