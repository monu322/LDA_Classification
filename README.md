Optimum Topic number using HDP, LDA & Clustering

In this experiment, I demonstrate a method to find the optimum number of topics in a corpus and evaluate the result by performing a classification task. I use the Hierarchical Dirichlet process (HDP) to find the number of topics. I use this topic number and default hyperparameters to train the LDA model. Then I use it’s output, the document topic matrix to train multiple classifiers. A good classification performance will be a correct way to know if we have arrived at the right topic number. 

Dataset: 10000 Yelp reviews, with rating numbers. Reviews with rating above 4 are labelled as “Good” reviews and those with 2 and below are labelled as “Bad reviews”. Neutral reviews are discarded. That makes the final count to 8539 labelled reviews.

Pre-processing: Removed stop words(nltk english corpus+common words), replaced new lines, applied bigram phrase modeling using Gensim’s phrase models, used Gensim’s simple_preprocess() method for lower casing and tokenization. 

Model Codes: I use Gensim implementation in Python, for LDA and HDP. Gensim’s utilities are also used for preprocessing and bigram modelling. Sklearn is used for classification

Experiment setting: Classification of Yelp reviews into “Good” or “Bad”. Coding was done in Google Colab notebooks. 

LDA parameters: Default values are used for Alpha and Eta. Both default to a symmetric 1/num_topics prior. Number of topics = 20

The experiment: 
The dataset of  8593 Yelp reviews are labelled as “Good” and “Bad” reviews.

Pre-processing is done to the review text as described, lemmatized and the corpus is created. 

The corpus is then fed into an HDP model which gave the number of topics = 20. 

This topic number was then used to train an LDA Topic model with default values for alpha and eta(1/num_topics). 

Using the output of the LDA model, a document-topic matrix is created. This is fed into 3 classifiers. A Logistic regressor, an SGD linear classifier with log loss and another with modifier Huber loss. Classification was used because the dataset was labelled. All three classifiers performed well during 5 fold cross validation. Results are shown below.
 
Logistic Regression 0.840 +- 0.009
Linear Classifier SGD 0.835 +- 0.018
Linear Classifier SGD Huber Loss 0.896 +- 0.004

From results you can see, the classifier is doing a good job in classifying the reviews into “Good” and “Bad”. Hence I conclude that the LDA model trained with 20 topics is acceptable and that using HDP is a fast and effective method to get the number of topics in a corpus.
