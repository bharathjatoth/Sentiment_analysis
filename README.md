# Sentiment_analysis
Sentiment analysis using both Traditional Neural Network and RNN

Here Neural_net.py is the main file which has trained the network. Here I have trained only with a one layer network you can change the network structure and the number of epoches and minibatch size according to your choice. 

Linked_list.py -> In this I've done the Preprocessing of the data. The model I've used is GLove-50D which converts each word to a 50-D vector. 

# For example we have a sentence "Food quality is very bad" 
-> We convert this sentence as 'Food' to a 50_D vec. We add all the 50-D vector of 5 words and divide then by the length of the sentence i.e., 5. This will be input of out Neural Network. 

We save the model with the name of my_test_model. We predict the new sentence by restoring the model and feeding model with new sentence with 50-D vector
