# Sentiment-Analysis-of-Airline-Twitter-Data

## 1. Overview
In the airline industries it is much easier to get feedback from astute data source such as Twitter, for conducting a sentiment analysis on their respective customers. In the report, we go through the steps and methods on the analytics of 3400 rows of tagged airline Twitter data by
exhibiting results of machine learning algorithms using R. The tweets are pre-processed and
classified into negative and non-negative sentiments. Support Vector Machine and Naïve Bayes
models have been used and compared for classifying the sentiments of the tweets. Finally, we
adopt the best performed model on new Twitter data and report the precision of the classification.

## 2. Steps and methods
### Data preprocessing
Load the data set and import the libraries.
Set sentiment value and combine the two datasets.

### Construct Document-Term matrix
 a. Create corpus using tweet data

b. Vocabulary-based vectorization

Define preprocessing function and tokenization function using text2vec

c. Construct document-term matrix

d. Convert document term matrix to matrix and factorize the sentiment column

e. Get the random sample in order to separate the dataset into training set and testing set

### Model Training and Evaluation
In this part, I train the model using SVM and NB models, and select the best performed
model by comparing the model evaluation indexes.

a. Define evaluation function to get precision, recall, F1 score, and accuracy

b. Training and testing through Support Vector Machine model
![precision](image/svm.png)

c. Training and testing through Naïve Bayesion
![precision](image/final.png)

By comparing the performance, we choose SVM as the final model because of the higher
accuracy and F1 score.

### Predict on new data
a. Load data

b. Corpus construction and Vocabulary-based vectorization
![precision](image/dictionary.png)

c. Construct DTM and convert it to matrix
![precision](image/dtm.png)

d. Get classification using SVM


## 3. classification precision
After getting the output file, I evaluate whether the sentiment is correct and output my
evaluation result in the second column. 1 denotes the sentiment is correctly classified, and 0 is
the opposite.
![precision](image/tagging.png)
In the end, the precision of my classification is about 66.87% (222/332)
