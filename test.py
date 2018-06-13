import csv
import json
import numpy as np
import os
import re
import random
import operator
import csv
import json
import copy
from nltk.tokenize import TweetTokenizer
from collections import Counter
from bs4 import BeautifulSoup


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models.word2vec import Word2Vec

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

def extract_regions(tokens, filter_size):
    regions = []
    regions.append(' '.join(tokens[:filter_size]))
    for i in range(filter_size, len(tokens)):
        regions.append(' '.join(tokens[(i-filter_size+1):(i+1)]))
    return regions

class Arg:

    def __init__(self):
        self.path_root = ""
        self.path_to_IMDB = self.path_root + "data/"
        self.path_to_pretrained_wv = ""
        self.path_to_plot = self.path_root
        self.path_to_save = "my_model/"  
        # Parameter values
        self.use_pretrained = False
        self.name_save = "cnn_two_channels_glove.hdf5"
        self.max_features = int(2e4)
        self.stpwd_thd = 1 
        self.max_size = 200
        self.word_vector_dim = int(3e2)
        self.do_static = False
        self.nb_filters = 150
        self.filter_size_a = 3
        self.filter_size_b = 4
        self.drop_rate = 0.3
        self.batch_size = 64
        self.nb_epoch = 6
        self.my_optimizer = 'adam' 
        self.my_patience = 2
        self.show_each_result = False

            
class CNN_Model:

    def __init__(self, arg):
        self.arg = arg

    def test(self):
        
        model = load_model(self.path_to_save + self.arg.name_save)

        get_region_embedding_a = K.function([model.layers[0].input,K.learning_phase()],
                                            [model.layers[3].output])

        get_region_embedding_b = K.function([model.layers[0].input,K.learning_phase()],
                                            [model.layers[4].output])

        get_softmax = K.function([model.layers[0].input,K.learning_phase()],
                                 [model.layers[11].output])

        n_doc_per_label = 2

        idx_pos = [idx for idx,elt in enumerate(self.arg.y_test) if elt==1]
        idx_neg = [idx for idx,elt in enumerate(self.arg.y_test) if elt==0]

        my_idxs = idx_pos[:n_doc_per_label] + idx_neg[:n_doc_per_label]
        x_test_my_idxs = np.array([self.arg.x_test[elt] for elt in my_idxs])
        print(x_test_my_idxs.shape)
        y_test_my_idxs = [self.arg.y_test[elt] for elt in my_idxs]

        reg_emb_a = get_region_embedding_a([x_test_my_idxs,0])[0]
        reg_emb_b = get_region_embedding_b([x_test_my_idxs,0])[0]

        # predictions are probabilities of belonging to class 1
        predictions = get_softmax([x_test_my_idxs,0])[0] 
        # note: you can also use directly: predictions = model.predict(x_test[:100]).tolist()
        n_show = 3 # number of most predictive regions we want to display

        for idx, doc in enumerate(x_test_my_idxs):
            
            tokens = [self.arg.index_to_word[elt] for elt in doc if elt!=0] # the 0 index is for padding

            # extract regions (sliding window over text)
            regions_a = extract_regions(tokens, self.arg.filter_size_a)
            regions_b = extract_regions(tokens, self.arg.filter_size_b)

            print('\n *********')
            print('===== text: =====')
            print(' '.join(tokens))
            print('===== label:',y_test_my_idxs[idx],'=====')
            print('===== prediction:',predictions[idx],'=====')
            norms_a = np.linalg.norm(reg_emb_a[idx,:,:],axis=1)
            norms_b = np.linalg.norm(reg_emb_b[idx,:,:],axis=1)
            print('===== most predictive regions of size', self.arg.filter_size_a,': =====')
            print([elt for idxx,elt in enumerate(regions_a) if idxx in np.argsort(norms_a)[-n_show:]]) # 'np.argsort' sorts by increasing order
            print('===== most predictive regions of size', self.arg.filter_size_b,': =====')
            print([elt for idxx,elt in enumerate(regions_b) if idxx in np.argsort(norms_b)[-n_show:]])
  

def load_data(arg):

    with open("my_model/word_to_index.json", 'r') as f:
        word_2_index = json.load(f)

    reviews = []
    with open("data/tweets_200/infinity_war.json", 'r') as f: 
        for line in f:
            reviews.append(line)
    test_data_text_list = copy.deepcopy(reviews)

    for review in reviews:
        review = review.lower()

    tokenizer = TweetTokenizer()
    
    # transfer all reviews to all tokens
    reviews_token = []
    for review in reviews:
        tokens = tokenizer.tokenize(review)
        reviews_token.append(tokens)
    
    test_data_list = []
    for review_token in reviews_token:
        test_data = []
        for token in review_token:
            if token in word_2_index:
                test_data.append(word_2_index[token])
        test_data_list.append(test_data)
    

    test_data_list = [[num for num in rev if num <= arg.max_features and num >= arg.stpwd_thd] for rev in test_data_list]
    test_data_list = [rev[:arg.max_size] for rev in test_data_list]
    test_data_list = [rev + [0] * (arg.max_size-len(rev)) if len(rev) < arg.max_size else rev for rev in test_data_list]

    return test_data_list, test_data_text_list

def predict(arg, test_data_list, test_data_text_list):

    model = load_model(arg.path_to_save + arg.name_save)
    
    get_region_embedding_a = K.function([model.layers[0].input,K.learning_phase()],
                                        [model.layers[3].output])

    get_region_embedding_b = K.function([model.layers[0].input,K.learning_phase()],
                                        [model.layers[4].output])

    get_softmax = K.function([model.layers[0].input,K.learning_phase()],
                             [model.layers[11].output])


    test_data_list = np.array(test_data_list)
    #print(test_data_list)
    reg_emb_a = get_region_embedding_a([test_data_list,0])[0]
    reg_emb_b = get_region_embedding_b([test_data_list,0])[0]
    predictions = get_softmax([test_data_list,0])[0] 
    
    print("\n***********************************")
    print("\n******** Prediction Result: *******")
    print("\n***********************************")
    avg_prob = 0
    for i in range(len(test_data_list)):
        avg_prob += predictions[i][0]
    avg_prob /= len(test_data_list)
    print("****  Average probility: {:.2f}  ****".format(avg_prob))
    if avg_prob >= 0.5:
        print("****  This movie is blockbuster!  ****")
    else:
        print("****  People hate this movie!  ****")
    print("Total reviews: {}".format(len(test_data_list)))

    if not arg.show_each_result:
        return
    for i in range(len(test_data_list)):
        print("--------- text content: ---------")
        print(test_data_text_list[i])
        print()
        print("Probability: {:.2f}".format(predictions[i][0]))
        if predictions[i] > 0.5:
            print("Prediction result: Positive")
        else:
            print("Prediction result: Negative")

        '''tokens = [arg.index_to_word[elt] for elt in test_data_list[i] if elt!=0]
        regions_a = extract_regions(tokens, arg.filter_size_a)
        regions_b = extract_regions(tokens, arg.filter_size_b)
        norms_a = np.linalg.norm(reg_emb_a[i,:,:],axis=1)
        norms_b = np.linalg.norm(reg_emb_b[i,:,:],axis=1)
        print('===== most predictive regions of size', arg.filter_size_a,': =====')
        print([elt for idxx,elt in enumerate(regions_a) if idxx in np.argsort(norms_a)[-3:]]) # 'np.argsort' sorts by increasing order
        print('===== most predictive regions of size', arg.filter_size_b,': =====')
        print([elt for idxx,elt in enumerate(regions_b) if idxx in np.argsort(norms_b)[-3:]])'''
        print("---------------------------------")  
        print()

    return 
    
def extract_regions(tokens, filter_size):
    regions = []
    regions.append(' '.join(tokens[:filter_size]))
    for i in range(filter_size, len(tokens)):
        regions.append(' '.join(tokens[(i-filter_size+1):(i+1)]))
    return regions


def main():

  arg = Arg()
  test_data_list, test_data_text_list = load_data(arg)
  predict(arg, test_data_list, test_data_text_list)
  #cnn_model = CNN_Model(arg)
  #cnn_model.test()
  
if __name__== "__main__":
  main()
