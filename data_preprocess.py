import pandas as pd
import csv
import numpy as np
import tensorflow as tf
ques_vec = []
load_model = False #make it True when you need to use word2Vec model
#In this version we used glove 50-D vector you can just place the location of the model in the below location place
words = pd.read_table('location of Glove', sep=" ",
                      index_col=0, header=None, quoting=csv.QUOTE_NONE)
if load_model is True:  #using Word2vec to convert to vectors
    def vec(w):
        x = os.path.join(os.getcwd(),'GoogleNews-vectors-negative300.bin')
        print(x)
        model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        return model[w]
def vec(w): #convert to vectors with glove 50-D or 100-D
    return words.loc[w].as_matrix()

x = (vec("hello").shape)

def oneh(x5):
    return tf.one_hot(x5,5,on_value=1.0,off_value=0.0,axis=-1)
  
def exact(sent):
    g1 = str(sent).split()
    # print(g1)
    g2 = np.zeros(x)
    for i in range(len(g1)):
        g2 += vec(g1[i])
    g2 = g2/len(g1)
    # g2 = np.array([g2])
    return g2
  
def here():
    feedback = []
    feedback_test = []
    print(len(ques_vec))
    book = pd.read_csv(r'data\train_emoji.csv')
    book1 = pd.read_csv(r'data\tesss.csv')
    rating = (book['rating'].tolist())
    rating1 = (book1['rating'].tolist())
    question = book['Question'].tolist()
    question1 = book1['Question'].tolist()
    # print(len(question))
    for i in range(len(question)):
        #clean the sentence here
        question[i] = str(question[i]).lower()
        curr = exact(question[i])
        # print(curr)
        ques_vec.append(curr)
        curr = ''
    # print(len(ques_vec))
    # print(ques_vec[0].shape)
    with tf.Session() as sess:
        for i in range(len(rating)):
            x3 = (sess.run(oneh(rating[i])))
            # print(x3)
            feedback.append(x3)
    # print(feedback[0])
    #tess data cleaning
    test_ques = []
    for i in range(len(question1)):
        #clean the sentence here
        question1[i] = str(question1[i]).lower()
        curr = exact(question1[i])
        test_ques.append(curr)
        curr = ''
    with tf.Session() as sess:
        for i in range(len(rating1)):
            x1 = (sess.run(oneh(rating1[i])))
            # print(x1)
            feedback_test.append(x1)
    # print(feedback_test[0])
    testing = "the food was very bad"
    new= []
    new.append(exact(testing.lower()))
    return ques_vec,feedback,test_ques,feedback_test,new
