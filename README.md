# Movie-Sucess-Prediction
We want to build a system which can predict an overall rating of a movie according to movie reviews from twitter.




main.py: 
- for training

test.py: 
- for testing
- testing data is in data/tweets_200/

my_model:
- cnn_two_channels_google.hdf5:
- the model is trained by using google word2vec

cnn_two_channels_glove.hdf5
- the model is trained by using glove word vector

aclImdb:
- training and testing data

GoogleNews-vectors-negative300.bin:
- google word2vec model

glove.twitter.27B:
- glove word vector model
- They have different dimensions
- when we want to use it to train our CNN model:
1. transfer it to word2vec format 
(call transform_glove.py)
2. main.py: change model name in intersect_word2vec_format function
3. binary = False
