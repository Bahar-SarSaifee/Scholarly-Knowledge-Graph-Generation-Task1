# Scholarly-Knowledge-Graph-Generation-Task1

This is the first of three challenges in this competition: Research Theme Extraction

## Task #1 - Extracting Research Themes

For this task, teams will be asked to develop a model that can identify and label research papers with a research theme. There will be a total of 36 themes, each paper will be labelled with a single theme.

Competitors will be provided with 2 files, train.csv and test.csv. The train.csv file will contain the details for each paper in question and the class label (the research theme). The test.csv file should then be used for evaluating your trained models.

Task participants are required to:

- Develop methods addressing the task and submit the results via Kaggle
- Document and submit their method as a short paper as specified on the SDP 2022 website
- Provide source code for each method

## Data Sources

Competition data will be supplied by the CORE aggregator (Knoth and Zdrahal, 2012).
CORE is the world’s largest aggregator of open access scientific literature. Data in CORE is harvested from over 10,000 repositories using the Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH) protocol. 

- COMPETITION SPONSOR: Kaggle
- LINK OF COMPETITION: https://www.kaggle.com/competitions/sdp2022-scholarly-knowledge-graph-generation/overview
- LINK OF DATASET: https://www.kaggle.com/competitions/sdp2022-scholarly-knowledge-graph-generation/data

## Data Preprocessing

- Using Keyword Extraction Methods; Knowledge Graph, Yake
- Remove HTML tags/ Http tags/ punctuation/ Digits
- Remove Stopwords
- Remove Words that have less than 2 letters

## Creating the Model

- Tokenization Using BERT Tokenizer
- Using Pre-trained Glove Embedding
- A Model Based on CNN Networks

## Evaluation

- The evaluation metric for this competition is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision P and recall R

- The F1 metric weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

