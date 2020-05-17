# sentiment-analysis
Sentiment Analysis of Large Movie Dataset
Abstract
Sentiment analysis is an application of Natural Language Processing (NLP) and it has become a vital method for developing opinion mining systems. This application is particularly used in noting of customer reviews and nature. This helps in making classification or recommendation based on the customers (or users in particular). In our case, we performed the sentiment analysis on Large Movie Dataset (IMDB) such as to help to build a recommendation system that can predict the sentiment of the reviewer after watching a movie. 

1.	Introduction and Background
Our project is about sentiment analysis on IMDB movie reviews. Natural Language Processing or NLP is a field of Artificial intelligence. It lets the machine understand the natural language and derive meaning from it. Sentiment analysis is a subfield of NLP in AI and is done on text blogs, reviews, news to identify and extract meaningful opinions. It is one of the most popular applications in text analytics.
Deep Learning is a subfield of machine learning which is built using neural networks. A neural network takes in the input into the input layer, which is then processed in hidden layers where computations are performed using weights, biases, and activation functions. The model is trained on the movies train data and accuracy and loss metrics are measured which eventually is used to predict the test data.
Naïve Neural Network, long short-term memory (LSTM) Recurrent Neural Network, Convolutional Neural Network (CNN) models were created and applied on a dataset consisting of 50K movie reviews. The dataset is binary classified and is split into 25k train and test data each. The data contains a further 50% positive reviews and 50% negative reviews. Data preprocessing was done initially using Word2Vec and word embedding. 

The results showed that the LSTM model have outperformed the Naïve NN and singular CNN. LSTM has reported an accuracy of 89.2% while CNN has given the accuracy of 87.7%, while MLP and LSTM have reported accuracy of 86.74% and 86.64 respectively. 



1.1.	The problem you tried to solve

We tried to show the sentiment analysis on the IMDB review dataset so that users can get an idea of what the movie is about and how the movie is at a glance. We built a neural network with Keras which categorizes user reviews as positive or negative and different algorithms were used on the dataset to measure and compare the accuracy and the loss. 

1.2 Results from the literature
The textbook we referred to was “Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems”. It dives deep into the TensorFlow libraries and modules. Also, it shows the basic flow of how to prepare the dataset before applying any model. 
	There are many ways to implement sentiment analysis. It can be performed using the Natural Language Tool Kit (NLTK) libraries, Pandas, Scikit Learn. Choosing modules and packages to use is really dependent on the dataset and since the dataset is already labeled in our case, we chose TensorFlow over NLTK libraries.

1.3 Tools and programs 
The sentiment analysis uses Natural Language Processing methods, and algorithms, such as:
•	Rule-Based Systems: This method defines a set of rules using NLP techniques (such as tokenization and parsing) and manually crafted rules. This rule works in a way that the opposite word list is prepared for example positive words and negative words. This has a disadvantage of not being able to correctly indicate if the review is positive or not. For example, a sentence such as “This movie was not bad!”. It will not take the word sequences into the account and will fail to produce correct results.

•	Automatic Systems: This method relies on machine learning and deep learning techniques to learn from data. There are two stages i.e. Training and Prediction involved in implementing this system.

o	Training Stage: The model at this stage learns to correctly tag text as negative and positive using the dataset. Then the feature extraction is used to create a pair of features vectors and tags that are then fed into a machine learning algorithm or neural network to generate the model
o	Prediction Stage: In this stage, the feature extractor is used to transform the unseen review into feature vectors which are then fed to the generated models, and accordingly the sentiment predictions are done.

•	We have implemented Automatic System in our project. This system can be implemented using machine learning libraries such as NLTK, Pandas, and algorithms such as Naïve Bayes and Support Vector Machine. Also, it can be applied using deep learning techniques with neural networks and data preprocessing can be done using the Word2Vec method or Glove method. We have used TensorFlow and Keras upon the NLTK library because we wanted to generate the neural network models such as LSTM Recurrent Neural Network, Convolutional Neural Network and to convert the word into vector space so as to tokenize them using Word2Vec method.

2.	 Overview of the architecture

 
Figure 1- Basic working of the project

Step 1:   Data Collection and Preprocessing
Data is loaded using TensorFlow from TensorFlow Datasets (TDFS) and the data pre-processing such as tokenizing of the reviews is done. The data is cleaned by removing the null values. 
 
Figure 2 – Loading data
Step 2: Data Splitting
The imdb dataset is spited into train data and test data with 25K data. Then, the two arrays of train data (Training Sentence and Training Label) and for test data (Testing Sentence and Testing Label) is created. The positive is denoted by label 1 and negative by label 0.
 
Figure 3 Splitting data
Tokenizing:
The most important step is tokenizing the words in a review. This is done using TensorFlow library objects that are preprocessing and Tokenizer. Vocabulary or Dictionary is created with the size of 10,000 words which is used to store each word. Embedding matrix size is 16 and the maximum words are kept at 120.
Text to sequences:
Text to sequences is performed where each token in a sentence is converted into the sequence. Also, we have used token to represent an unidentified word in the train data.
Padding sequences:
Sequences are padded with zeroes so that the matrix is aligned. We do post padding and truncating. The post padding and truncating is done such that no information is lost.
 
Figure 4 Tokenizing and Padding
Step 3: Reverse Dictionary
The tokens are no longer represented as numbers. The dictionary is reversed and is mapped to the words that the numbers were actually representing. For example, word like ‘HELLO’ was tokenized and converted into sequence such as [H, E, L, L, O]  [1,2,3,4,5].Now it is time to represent the numbers again by words such as [1,2,3,4,5] [H, E, L, L, O]. 
Words represented in vector space using the Embedded Projector which is inbuilt TensorFlow tool 
 
Figure 5 Reverse Dictionary
Step 4: Generate Neural Network Model
In this step the neural network model is created, and the input is created for NN which is an Embedding matrix. The naive neural network is created using the Kera’s Sequential object and the layer is flatten using GlobalAveragePooling1D so that the dimensions can be reduced.
Dense seven layer is used. For the first six layers the Rectified Liner Unit (Relu) is used and finally the Sigmoid activation function is applied. The model is compiled using BinaryCrossEntropy Loss and accuracy is measured. 
 
Figure 5 Naïve Neural Network
Step 5: Model fit, and Model evaluate
Model is fitted on the training data and is evaluated to calculate the loss and accuracy for each iteration. The number of the iterations (epochs) can be customized. The accuracy versus validation accuracy graph is made using matplotlib library. Similarly, loss and validation loss are plotted.
 
Figure 6 Plot graph
Step 6: Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) 
The model is created using Bidirectional LSTM RNN. Similarly, the accuracy and loss are measured by fitting the data on the training data. 
 
Figure 7 Bidirectional LSTM
Step 7: Convolutional Neural Network (CNN)
The model is created using CNN. Similarly, the accuracy and loss are measured by fitting the data on the training data. 
 
Figure 8 CNN
Step 8: Model Prediction
Finally, for each model prediction is done using model. Predict. If the predictions are less than or equal to 0.5 then it is positive review and more than 0.5 it is negative review. 
Figure 9 Model Predict







2.1 Finished work: Running modules
 
Figure 10 Load Data 

 
Figure 11 Train data review and Labels
 
 
Figure 12 Decoding Review

Naïve Neural Network:
 
Figure 13 Naïve Neural Network
 
 

 
Figure 14 Naïve Neural Network Accuracy and Loss
Long Short-Term Memory Recurrent Neural Network (LSTM- RNN):
 
Figure 15 LSTM Recurrent Neural Network
 
 
 
Figure 16 LSTM Recurrent Neural Network Accuracy and Lose
Convolution Neural Network (CNN):
 
 
 
 
Figure 17 Convolutional Neural Network Accuracy and Loss
 
 Figure 18 Predictions
2.2	Work in progress: Modules designed but not implemented

We tried to implement the prediction on the test data. We were able to predict on the whole test data. But we are trying to figure to apply predict function on random or particular group of reviews.
 
Figure 19 Predictions on whole model
We also tried to apply other algorithms and neural network such as Hybrid Neural Network (LSTM and CNN) so as to improve the accuracy of the system while keeping in mind there is least loss to predict the value on test data.
2.3	Future work: Modules a future continuation may have

•	The project is focused to make recommendation system. We like to implement the predictions on more larger movie dataset. 
•	Also, we want to deploy the recommendation system using mobile application such that customers time of reading long reviews are saved by just a click.




3.	Results and Evaluation

Naïve neural network accuracy and loss @ epoch 10 and 50
In epoch=10, sharp rise in accuracy at first epoch can be observed then steadily increasing and similarly loss can be seen doing just the opposite of accuracy.
In epoch=50, a slow increase till 20 in accuracy and slight changes trying to attain stability can be seen after that and loss is decreasing very slowly.  

  
LSTM-RNN accuracy and loss @ epoch 10 and 50
In epoch=10, a slight increase and decrease observed between epoch 5 and epoch 7 and then sharp increase after that to achieve accuracy and on contrary loss was down and up then down to minimum. 
In epoch=50, a slow increase in accuracy and a slow decrease in loss. 
  



CNN accuracy and loss @ epoch 10 and 50

In epoch=10, accuracy has rapid increase can be noticed at epoch one and then a smooth increase and loss is vice versa.
In epoch=50, a smooth and steady increase of accuracy can be seen, at and after epoch 12 we can see data is attaining stability not changing, as well as loss but in contrast to accuracy.
  
Analysing from above 3 models and graphs, CNN has advantage over other models.

4.	Discussion and Conclusions
We are trying to build a recommendation system for the sentiment analysis of the Large Movie Dataset (IMDB). This application of Natural Language Processing can implement in several ways. We choose TensorFlow and Keras open source libraries for machine learning.
The main aim for choosing these libraries is they are machine learning-friendly libraries and can easily be implemented on GPU’S. Our method includes conversion of the words in a vector space using Word2Vec and applying the Naïve Neural Network, Long Short-Term Memory Recurrent Neural Networks and Convolutional Neural Network to the training data such that we can compile and evaluate our model and predict the reviews probability on testing data.
In the first step, we Tokenized the text and then converted it sequences. We then padded sequences and tried to show the words in vector space using the Word2Vec technique in the inbuilt Embedding projector offered by TensorFlow. We generated our respective Neural Network models, LSTM-RNN and CNN with all of them having an input layer of padded sequences and dictionary of size 10K words. We compiled our model and measured the Accuracy and Binary cross-entropy loss which is considered as the best for the binary classified data.
We compared each model’s accuracy and loss on the respective Training and Testing data by plotting the graphs in each case between the
•	accuracy and validation accuracy
•	loss and validation loss
Finally, we predicted each model to classify the review as a positive or negative review.
CONCLUSION:
•	We achieved the representation of word to vector space and display the clustering of the words alike by calculating the Euclidean and Cosine distances.
•	We applied the Training data to each model that are NN, LSTM and CNN and compared accuracy, loss and total parameters used by each model and as a
conclusion, we figured that fitting the model with the number of epochs as 50, the
accuracy of CNN model is the highest while LSTM and NN are neutral
•	CNN uses a smaller number of parameters as compared to LSTM
FUTURE EXTENTIONS AND IMPROVEMENTS:
We were successfully able to check accuracy and loss on NN, CNN, and LSTM-RNN. We will try to implement hybrid model of LSTM and CNN. Also, a multi-layer CNN to our model and figure the best model we can use to produce more accurate sentiment predictions of the reviews. In addition to it we want to predict the reviews for any random data and deploy it in the form of some mobile application.
5.	 References
•	Referred from Part-II chapter-9 Up and Running with TensorFlow Hands-On Machine Learning
•	with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
•	https://www.pythonforengineers.com/build-a-sentiment-analysis-app-with-movie-reviews/
•	https://builtin.com/data-science/how-build-neural-network-keras
