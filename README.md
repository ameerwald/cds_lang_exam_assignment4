
# Assignment 4 - Using finetuned transformers via HuggingFace

## Github repo link

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_lang_exam_assignment4). 

## The data
The same dataset of fake and real news headlines used previously can be found again [here] (https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). It contains the headline, text of the story and label (fake or real). 

## Assignment description

In this assignemnt, we were asked to use ```HuggingFace``` and the *Fake or Real News* dataset completing the following tasts:

- Initalize a ```HuggingFace``` pipeline for emotion classification
- Perform emotion classification for every *headline* in the data
- Assuming the most likely prediction is the correct label, create tables and visualisations which show the following:
  - Distribution of emotions across all of the data
  - Distribution of emotions across *only* the real news
  - Distribution of emotions across *only* the fake news
- Comparing the results, discuss if there are any key differences between the two sets of headlines


# Repository 

| Folder         | Description          
| ------------- |:-------------:
| In      | Fake and real news data set 
| Notes | Jupyter notebook and script with notes       
| Out  | Visual representations of the data   
| Src  | Py script 
| Utils  |        


## To run the scripts 

1. Clone the repository, either on ucloud or something like worker2
2. From the command line, at the /cds_vis_exam_assignment1/ folder level, run the following lines of code. 

This will create a virtual environment, install the correct requirements.
``` 
bash setup.sh
```
While this will run the scripts and deactivate the virtual environment when it is done. 
```
bash run.sh
```

This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.

# Discussion of results 
In the graph showing all headlines, neutral is the most commonly classified emotion amongst the headlines. However when looking at the headlines split by fake and real, the most commonly classified emotion is disgust, in both. This is a bit puzzling. I tried coding it in another way with the same results so unsure at this time why that is occuring. Otherwise it appears that fear and joy were more present in the real headlines. 




