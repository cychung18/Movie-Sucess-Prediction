import csv
import json
import numpy as np

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
        self.path_root = ''
        self.path_to_IMDB = self.path_root + 'data/'
        self.path_to_pretrained_wv = ''
        self.path_to_plot = self.path_root
        self.path_to_save = "my_model/"  
        # Parameter values
        self.use_pretrained = True
        self.name_save = self.path_to_save + "cnn_two_channels.hdf5"
        print('Model will be saved in {} folder'.format(self.path_to_save))   
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

        self.load_data()
        self.data_preprocess()
        self.embeddings = self.load_pretrained_word2vector()

    def load_data(self):
        with open('my_model/word_to_index.json', 'r') as my_file:
            self.word_to_index = json.load(my_file)

        with open(self.path_to_IMDB + 'training.csv', 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            x_train = list(reader)

        with open(self.path_to_IMDB + 'test.csv', 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            x_test = list(reader)

        with open(self.path_to_IMDB + 'training_labels.txt', 'r') as my_file:
            y_train = my_file.read().splitlines()

        with open(self.path_to_IMDB + 'test_labels.txt', 'r') as my_file:
            y_test = my_file.read().splitlines()

        # turn lists of strings into lists of integers
        self.x_train = [[int(elt) for elt in sublist] for sublist in x_train]
        self.x_test = [[int(elt) for elt in sublist] for sublist in x_test]  

        self.y_train = [int(elt) for elt in y_train]
        self.y_test = [int(elt) for elt in y_test]

        print("Data are loaded!")


    def data_preprocess(self):

        self.index_to_word = dict((v,k) for k, v in self.word_to_index.items())
        #print (' '.join([self.index_to_word[elt] for elt in self.x_train[4]]))
        # Stopwords and out-of-vocab words removal
        self.stpwds = [self.index_to_word[idx] for idx in range(1,self.stpwd_thd)]
        print('stopwords are: {}'.format(self.stpwds))

        self.x_train = [[elt for elt in rev if elt <= self.max_features and elt >= self.stpwd_thd] for rev in self.x_train]
        self.x_test =  [[elt for elt in rev if elt <= self.max_features and elt >= self.stpwd_thd] for rev in self.x_test]
        #print('pruning done')
        # Truncation and zero-padding

        self.x_train = [rev[:self.max_size] for rev in self.x_train]
        self.x_test = [rev[:self.max_size] for rev in self.x_test]

        self.x_train = [rev+[0]*(self.max_size-len(rev)) if len(rev)<self.max_size else rev for rev in self.x_train]

        self.x_test = [rev + [0] * (self.max_size - len(rev)) if len(rev) < self.max_size else rev for rev in self.x_test]

        print('Data preprocess done!')

    def load_pretrained_word2vector(self):
        # convert integer reviews into word reviews
        x_full = self.x_train + self.x_test
        x_full_words = [[self.index_to_word[idx] for idx in rev if idx!=0] for rev in x_full]
        all_words = [word for rev in x_full_words for word in rev]

        print("There are {} words.".format(len(all_words)))

        if not self.use_pretrained:
            print('Not using pre-trained embeddings')
            return

        print("Loading pre-trained word2vec.")
        print("It may take a few minutes...")
        # initialize word vectors
        word_vectors = Word2Vec(size = self.word_vector_dim, min_count = 1)

        # create entries for the words in our vocabulary
        word_vectors.build_vocab(x_full_words)

        # fill entries with the pre-trained word vectors
        word_vectors.intersect_word2vec_format(self.path_to_pretrained_wv + 'GoogleNews-vectors-negative300.bin.gz', binary=True)

        print('pre-trained word vectors loaded')

        norms = [np.linalg.norm(word_vectors[word]) for word in list(word_vectors.wv.vocab)] # in Python 2.7: word_vectors.wv.vocab.keys()
        idxs_zero_norms = [idx for idx, norm in enumerate(norms) if norm < 0.05]
        no_entry_words = [list(word_vectors.wv.vocab)[idx] for idx in idxs_zero_norms]
        print('# of vocab words w/o a Google News entry:',len(no_entry_words))

        # create numpy array of embeddings  
        embeddings = np.zeros((self.max_features + 1, self.word_vector_dim))
        for word in list(word_vectors.wv.vocab):
            idx = self.word_to_index[word]
            # word_to_index is 1-based! the 0-th row, used for padding, stays at zero
            embeddings[idx,] = word_vectors[word]
            
        print('embeddings created')
        return embeddings
            
class CNN_Model:

    def __init__(self, arg):

        self.arg = arg
        self.def_model()
    
    def def_model(self):
        my_input = Input(shape = (self.arg.max_size,)) # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument

        if self.arg.use_pretrained:
            embedding = Embedding(input_dim = self.arg.embeddings.shape[0], # vocab size, including the 0-th word used for padding
                                  output_dim = self.arg.word_vector_dim,
                                  weights = [self.arg.embeddings], # we pass our pre-trained embeddings
                                  input_length = self.arg.max_size,
                                  trainable = not self.arg.do_static,
                                  )(my_input)
        else:
            embedding = Embedding(input_dim = self.arg.max_features + 1,
                                  output_dim = self.arg.word_vector_dim,
                                  trainable = not self.arg.do_static,
                                  )(my_input)

        embedding_dropped = Dropout(self.arg.drop_rate)(embedding)

        # feature map size should be equal to max_size-filter_size + 1
        # tensor shape after conv layer should be (feature map size, nb_filters)
        print('branch A:', self.arg.nb_filters, 'feature maps of size', self.arg.max_size - self.arg.filter_size_a + 1)
        print('branch B:', self.arg.nb_filters, 'feature maps of size', self.arg.max_size - self.arg.filter_size_b + 1)

        # A branch
        conv_a = Conv1D(filters = self.arg.nb_filters,
                        kernel_size = self.arg.filter_size_a,
                        activation = 'relu',
                        )(embedding_dropped)

        pooled_conv_a = GlobalMaxPooling1D()(conv_a)

        pooled_conv_dropped_a = Dropout(self.arg.drop_rate)(pooled_conv_a)

        # B branch
        conv_b = Conv1D(filters = self.arg.nb_filters,
                        kernel_size = self.arg.filter_size_b,
                        activation = 'relu',
                        )(embedding_dropped)

        pooled_conv_b = GlobalMaxPooling1D()(conv_b)

        pooled_conv_dropped_b = Dropout(self.arg.drop_rate)(pooled_conv_b)

        concat = Concatenate()([pooled_conv_dropped_a, pooled_conv_dropped_b])

        concat_dropped = Dropout(self.arg.drop_rate)(concat)
       
        fully_conn = Dense(units = 32, activation = 'relu')(concat_dropped)
        # we finally project onto a single unit output layer with sigmoid activation
        prob = Dense(units = 1, activation = 'sigmoid',)(fully_conn)

        self.model = Model(my_input, prob)

        self.model.compile(loss = 'binary_crossentropy',
                      optimizer = self.arg.my_optimizer,
                      metrics = ['accuracy'])

        print('model compiled')

        self.model.summary()
        self.model.layers[4].output_shape # dimensionality of document encodings (nb_filters*2)

        print('total number of model parameters: {}'.format(self.model.count_params()))


        # Visualization of document embeddings before training
        # in test mode, we should set the 'learning_phase' flag to 0 (we don't want to use dropout)
        get_doc_embedding = K.function([self.model.layers[0].input,K.learning_phase()],
                                       [self.model.layers[9].output])

    def train(self):
        
        early_stopping = EarlyStopping(monitor = 'val_acc', # go through epochs as long as accuracy on validation set increases
                                       patience = self.arg.my_patience,
                                       mode = 'max')

        # make sure that the model corresponding to the best epoch is saved
        checkpointer = ModelCheckpoint(filepath = self.arg.name_save,
                                       monitor = 'val_acc',
                                       save_best_only = True,
                                       verbose = 0)

        self.model.fit(np.array(self.arg.x_train), 
                       np.array(self.arg.y_train),
                       batch_size = self.arg.batch_size,
                       epochs = self.arg.nb_epoch,
                       validation_data = (np.array(self.arg.x_test), np.array(self.arg.y_test)),
                       callbacks = [early_stopping, checkpointer])
        self.model.save(self.arg.name_save)
    def test(self):
        
        model = load_model(self.arg.name_save)
        # Visualizing and understanding CNN

        # Document embeddings after training

        # Predictive text regions

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
        y_test_my_idxs = [self.arg.y_test[elt] for elt in my_idxs]

        reg_emb_a = get_region_embedding_a([x_test_my_idxs,0])[0]
        reg_emb_b = get_region_embedding_b([x_test_my_idxs,0])[0]

        # predictions are probabilities of belonging to class 1
        predictions = get_softmax([x_test_my_idxs,0])[0] 
        # note: you can also use directly: predictions = model.predict(x_test[:100]).tolist()
        print("\n***********************************")
        print("\n******** Prediction Result: *******")
        print("\n***********************************")
        n_show = 3 # number of most predictive regions we want to display
        for idx, doc in enumerate(x_test_my_idxs):
                
            tokens = [self.arg.index_to_word[elt] for elt in doc if elt!=0] # the 0 index is for padding
            
            # extract regions (sliding window over text)
            print("--------- text content: ---------")
            print(' '.join(tokens))
            print('===== label:',y_test_my_idxs[idx],'=====')
            print('===== Probability:',predictions[idx][0],'=====')
            if predictions[idx] > 0.5:
                print("Prediction result: Positive")
            else:
                print("Prediction result: Negative")
            '''
            regions_a = extract_regions(tokens, filter_size_a)
            regions_b = extract_regions(tokens, filter_size_b)
            norms_a = np.linalg.norm(reg_emb_a[idx,:,:],axis=1)
            norms_b = np.linalg.norm(reg_emb_b[idx,:,:],axis=1)
            print('===== most predictive regions of size',filter_size_a,': =====')
            print([elt for idxx,elt in enumerate(regions_a) if idxx in np.argsort(norms_a)[-n_show:]]) # 'np.argsort' sorts by increasing order
            print('===== most predictive regions of size',filter_size_b,': =====')
            print([elt for idxx,elt in enumerate(regions_b) if idxx in np.argsort(norms_b)[-n_show:]])'''
            print("---------------------------------")  
            print()

  
def main():
  arg = Arg()
  cnn_model = CNN_Model(arg)
  cnn_model.train()
  cnn_model.test()
  
if __name__== "__main__":
  main()
