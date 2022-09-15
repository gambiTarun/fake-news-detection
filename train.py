
# coding: utf-8

# In[1]:


import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import nltk
import numpy as np
import re
from time import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS


# In[2]:


fake_news = pd.read_csv("Kaggle_dataset/fake_13k.csv" , usecols=["uuid","author","title","text","country","type","language"])


# In[4]:


fake_news.head(5)


# In[3]:


fake_news.groupby(["type"]).size()


# In[5]:


# Categorizing all data as 'fake'
fake_news['type'] = 'fake'


# In[7]:


fake_news.groupby(["type"]).size()


# In[6]:


fake_news.shape


# In[8]:


# removing all non-english articles
fake_news = fake_news[fake_news.language == "english"]
fake_news = fake_news[['title','text','type']].rename(columns={'title':'headline','text':'body'})


# In[10]:


def filter_nan(dataset):   
    # changing the NaN values in titles to ""
    # It is managable without the title so do not delete such datapoints
    dataset.headline.fillna(value="",inplace=True)   

    # deleting the row with NaN text
    # No text body, datapoint useless
    dataset.dropna(axis=0, inplace=True, subset=["body"])

    # deleting the row having less than 100 characters in body
    # No text body, datapoint useless
    dataset = dataset[np.array([len(i.split()) for i in dataset.body]) > 100]
    dataset.reset_index(inplace=True,drop=True)
    return dataset


# In[9]:


stopword = set(STOPWORDS)

# Text Cleaning

def rem_specialChar(text):
    # removing html tags
    text = re.sub("<.*?>","",text)
    # removing websites, email addresses or any punctuation
    # changing "*http(s)*", "*www*" or "*@*" with " "
    # \S+ means anything except whitespace char >=1 times
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", "", text)
    # changing anything except alphabets to " "
    text = re.sub("[^a-zA-Z]"," ",text)
    # lower casing
    text= text.lower()
    return text

def stopword_remove(text):
    word_list = nltk.word_tokenize(text)
    # remove all the stopwords in the word_list
    return ' '.join([w for w in word_list if w not in stopword])

# stemmer object -> to apply stemming -> reducing all the words to its root word
stemmer = nltk.stem.porter.PorterStemmer()

def stem_words(text):
    word_list = nltk.word_tokenize(text)
    # removing all the single letters from the word_list
    word_list = [w for w in word_list if len(w) > 1]
    # reduce all family words to parent word
    word_list = [stemmer.stem(w) for w in word_list]
    return ' '.join(word_list)

# # Example, number of character reduction can be seen
# print(len(fake_dataset.body[0]))
# stage0 = rem_specialChar(fake_dataset.body[0])
# stage1 = stopword_remove(stage0)
# stage2 = stem_words(stage1)
# print(len(stage0))
# print(len(stage1))
# print(len(stage2))


# ## WORDCLOUD

# In[11]:


# def filter_body_wordcloud(dataset):
#     # Cleaning only (for wordcloud), 
#     # Stopword removal done in wordcloud (inbuilt) and Stemming not required
#     t1 = time()
#     # cleaning text of special web and email addresses
#     clean_dataset_body_list = [rem_specialChar(i) for i in dataset.body]

#     print("Time taken to clean all the body(s): {0:.2f} min".format((time()-t1)/60))
#     return clean_dataset_body_list

# def wordcloud(clean_dataset_body_list,str_):
#     t1 = time()
#     # creating a WordCloud to see the Type of Content in the text
#     wordcloud = WordCloud(
#                     background_color="white",
#                     stopwords=stopword,
#                     max_words=200,
#                     max_font_size=40,
#                     random_state=42
#     ).generate(str(clean_dataset_body_list))

#     print("Time taken to generate wordcloud: {0:.2f} min".format((time()-t1)/60))
    
#     fig = plt.figure(1)
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()
#     fig.savefig(str_+".png", dpi=900)


# In[12]:


#wordcloud(filter_body_wordcloud(fake_news),"fake_dataset_wordcloud")


# ### Wordcloud shows that the fake news articles have news in the genre of Politics, Business, Global News, US-elections etc.

# In[13]:


real_news = pd.read_csv("Kaggle_dataset/real_67k.csv")


# In[14]:


#wordcloud(filter_body_wordcloud(real_news),"real_dataset_wordcloud")


# In[16]:


real_news = real_news.assign(type = 'real')
real_news.shape


# In[16]:


# removing empty/nan article bodies and articles with less that 100 words
fake_dataset = filter_nan(fake_news)
real_dataset = filter_nan(real_news)


# In[17]:


# the dataset is cleaned of any special characters,
# and word stemming is done to remove any words with almost similar meaning and
# hence will help reduce the tf-idf dimensionality (number of unique words)
def clean_body(dataset):
    t0 = time()

    for i in dataset.index:
        dataset.at[i,'body'] = stem_words(rem_specialChar(dataset.body[i]))

    print("Time taken to cleanUp : %.3f min" % ((time()-t0)/60))


# In[31]:


# # cleaning body of special characters that do not mean anything to the content of the article
# clean_body(fake_dataset)
# clean_body(real_dataset)

# # removing empty/nan article bodies and articles with less that 100 words 
# # (after removing special char and Stemming)
# fake_dataset = filter_nan(fake_dataset)
# real_dataset = filter_nan(real_dataset)

# fake_dataset.to_csv('fake_dataset_clean_body.csv',index=False)
# real_dataset.to_csv('real_dataset_clean_body.csv',index=False)

fake_dataset = pd.read_csv('fake_dataset_clean_body.csv')
real_dataset = pd.read_csv('real_dataset_clean_body.csv')


# In[32]:


fake_dataset.shape


# In[33]:


real_dataset.shape


# In[34]:


# reducing the number of real_dataset datapoints around that of fake_dataset
# keeping the fake to non-fake news ratio close to presence in real world media
real_dataset = real_dataset.sample(n=45000,random_state=42)
real_dataset.reset_index(drop=True, inplace=True)


# In[35]:


dataset = pd.concat([real_dataset,fake_dataset])
dataset.reset_index(drop=True, inplace=True)


# In[36]:


dataset.head(5)


# In[85]:


train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)


# In[86]:


train_dataset.body[0]


# In[87]:


train_dataset.shape


# In[88]:


test_dataset.shape


# ### Data Preprocessing

# In[89]:


# words occuring in only at most 9 document or at least 95% of the documents are REMOVED
vector_tfidf = TfidfVectorizer(stop_words='english', max_df=0.90, min_df=2)
#vector_tf = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)

# tfidf_doc = N_word_doc/N_totalwords_doc * log(N_total_doc/N_totaldocs_word)
tfidf = vector_tfidf.fit_transform(train_dataset.body.astype('U'))   # taking test as type Unicode
#tf = vector_tf.fit_transform(x.body)
tfidf.shape


# In[90]:


tfidf


# In[91]:


docs = tfidf.shape[0]
words = tfidf.shape[1]


# In[92]:


vocab = vector_tfidf.vocabulary_

vocab = sorted(vocab.items(), key=lambda x:x[1])

vocab = [i[0] for i in vocab]


# In[93]:


get_ipython().run_cell_magic('time', '', "# Dimension Reduction of the TF-IDF with Latent Semantic Analysis (LSA)\nn_comp = 1000\n# Contrary to PCA, this estimator does not center the data before computing the singular value decomposition.\n# This means it can work with scipy.sparse matrices efficiently, \n# as centering the data will give non zero values to zero elements.\n\n# setting n-components as the number of docs, as that's the maximum \n# number of features a LSA reduced feature-set can have.\nsvd = TruncatedSVD(n_components=n_comp, n_iter=3, algorithm='randomized')\ntfidf_red = svd.fit_transform(tfidf)\n\nprint(tfidf_red.shape)")


# In[94]:


# plot of variance(percentage) contributed by each feature
plt.plot(np.arange(0,n_comp,1),svd.explained_variance_ratio_*100)
plt.xlabel('features/tokens')
plt.ylabel('Variance Percentage contributed')
plt.savefig('svd_variance.jpeg')


# In[95]:


# commulative contribution of variance by the features
cumm = [(svd.explained_variance_ratio_*100)[:i+1].sum() for i in range(n_comp)]
plt.plot(np.arange(0,n_comp,1),cumm)
plt.xlabel('features/tokens')
plt.ylabel('Cummulative Variance Percentage contributed')
plt.savefig('svd_variance_cummulative.jpeg')


# In[97]:


svd.explained_variance_ratio_.sum()*100


# In[96]:


# plotting the tfidf vector of datapoint 0
plt.plot(np.arange(0,tfidf.shape[1],1),tfidf[0,:].todense().T)
plt.title('DataPoint #0')
plt.xlabel('features/tokens')
plt.savefig('tfidf_vector.jpeg')
plt.show()


# In[98]:


# plotting the reduced tfidf vector of datapoint 0
plt.plot(np.arange(0,tfidf_red.shape[1],1),tfidf_red[0,:])
plt.title('DataPoint #0')
plt.xlabel('features/tokens')
plt.savefig('tfidf_vector_reduced.jpeg')
plt.show()


# In[99]:


# Because our tfidf is not centered, the first feature after LSA reduction
# just contains information on the frequency of the words in the documents.

# Therefore, to really see any difference in features of the two types of articles,
# we will use second and third most variant features plot to see any distinction.

# for data, fak in zip(tfidf_red[:,1:3],target):
#     x, y = data
#     if(fak):
#         fake = plt.scatter(x, y, c='red', alpha=0.5)
#     else:
#         real = plt.scatter(x, y, c='green', alpha=0.5)
c = ['red' if i=='fake' else 'green' for i in train_dataset.type]

plt.figure(figsize=(10,10))
plt.scatter(tfidf_red[:,1], tfidf_red[:,2], c=c, alpha=0.5)
    
#plt.legend(('real','fake'),loc=0)
plt.xlabel('feature index : 1')
plt.ylabel('feature index : 2')
plt.title("All training DataPoints")
plt.savefig('top_two_features_visual.jpeg')
plt.show()


# In[100]:


train_X = tfidf_red
train_y = np.array([1 if i=='fake' else 0 for i in train_dataset.type])

# Using the Vectorizer object fitted with the training
# data to transform the test data with same features.
test_tfidf = vector_tfidf.transform(test_dataset.body.astype('U'))

# Using the LSA's SVD object fitted with the training 
# data to transform the test data with the same features.
test_X = svd.transform(test_tfidf)
test_y = np.array([1 if i=='fake' else 0 for i in test_dataset.type])


# ## MODELS

# ### Logistic Regression

# In[148]:


from sklearn.linear_model import LogisticRegression

def LogisticReg_train(X, y):
    
    clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', multi_class='multinomial')
    model = clf.fit(X, y)
    return model


# In[149]:


get_ipython().run_cell_magic('time', '', 'prf_f = []\nprf_r = []\nacc = []\nfor i in range(10):\n    \n    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)\n    train_dataset.reset_index(drop=True, inplace=True)\n    test_dataset.reset_index(drop=True, inplace=True)\n    \n    vector_tfidf = TfidfVectorizer(stop_words=\'english\', max_df=0.90, min_df=2)\n    tfidf = vector_tfidf.fit_transform(train_dataset.body.astype(\'U\'))\n    \n    svd = TruncatedSVD(n_components=1000, n_iter=3, algorithm=\'randomized\')\n    tfidf_red = svd.fit_transform(tfidf)\n        \n    train_X = tfidf_red\n    train_y = np.array([1 if i==\'fake\' else 0 for i in train_dataset.type])\n    test_tfidf = vector_tfidf.transform(test_dataset.body.astype(\'U\'))\n    test_X = svd.transform(test_tfidf)\n    test_y = np.array([1 if i==\'fake\' else 0 for i in test_dataset.type])\n    \n    logReg_clf = LogisticReg_train(train_X, train_y)\n    test_pred = logReg_clf.predict(test_X)\n    \n    prf_f.append(precision_recall_fscore_support(test_y,test_pred, pos_label=1, average=\'binary\')[:3])\n    prf_r.append(precision_recall_fscore_support(test_y,test_pred, pos_label=0, average=\'binary\')[:3])\n    \n    acc = np.mean(np.equal(test_y,test_pred))\n    print("{} of 10".format(i))\n\nprf_fake = np.mean(prf_f,axis=0)\nprf_real = np.mean(prf_r,axis=0)\naccuracy = np.mean(acc)\n\nprint("LOGISTIC REGRESSION\\n")\nprint("\\nAccuracy : {:.2f}%".format(accuracy*100))\nprint(\'\\nPrediction of Real News (0)\')\nprint("Precision : {:.2f}%".format(prf_real[0]*100))\nprint("Recall : {:.2f}%".format(prf_real[1]*100))\nprint("F1Score : {:.2f}%".format(prf_real[2]*100))\nprint(\'\\nPrediction of Fake News (1)\')\nprint("Precision : {:.2f}%".format(prf_fake[0]*100))\nprint("Recall : {:.2f}%".format(prf_fake[1]*100))\nprint("F1Score : {:.2f}%".format(prf_fake[2]*100))')


# In[150]:


prf_f


# In[151]:


prf_r


# In[152]:


acc


# ### Support Vector Machine

# In[153]:


def SupportVecMachine_train(X, y):
    
    model = svm.SVC(kernel='rbf')
    clf = model.fit(X,y)
    return clf


# In[154]:


get_ipython().run_cell_magic('time', '', 'prf_f = []\nprf_r = []\nacc = []\nfor i in range(10):\n    \n    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)\n    train_dataset.reset_index(drop=True, inplace=True)\n    test_dataset.reset_index(drop=True, inplace=True)\n    \n    vector_tfidf = TfidfVectorizer(stop_words=\'english\', max_df=0.90, min_df=2)\n    tfidf = vector_tfidf.fit_transform(train_dataset.body.astype(\'U\'))\n    \n    svd = TruncatedSVD(n_components=1000, n_iter=3, algorithm=\'randomized\')\n    tfidf_red = svd.fit_transform(tfidf)\n        \n    train_X = tfidf_red\n    train_y = np.array([1 if i==\'fake\' else 0 for i in train_dataset.type])\n    test_tfidf = vector_tfidf.transform(test_dataset.body.astype(\'U\'))\n    test_X = svd.transform(test_tfidf)\n    test_y = np.array([1 if i==\'fake\' else 0 for i in test_dataset.type])\n    \n    svm_clf = SupportVecMachine_train(train_X[:,:50],train_y)\n    test_pred = svm_clf.predict(test_X[:,:50])\n    \n    prf_f.append(precision_recall_fscore_support(test_y,test_pred, pos_label=1, average=\'binary\')[:3])\n    prf_r.append(precision_recall_fscore_support(test_y,test_pred, pos_label=0, average=\'binary\')[:3])\n    \n    acc.append(np.mean(np.equal(test_y,test_pred)))\n    print("{} of 10".format(i))\n\nprf_fake = np.mean(prf_f,axis=0)\nprf_real = np.mean(prf_r,axis=0)\naccuracy = np.mean(acc)\n\nprint("SUPPORT VECTOR MACHINE\\n")\nprint("\\nAccuracy : {:.2f}%".format(accuracy*100))\nprint(\'\\nPrediction of Real News (0)\')\nprint("Precision : {:.2f}%".format(prf_real[0]*100))\nprint("Recall : {:.2f}%".format(prf_real[1]*100))\nprint("F1Score : {:.2f}%".format(prf_real[2]*100))\nprint(\'\\nPrediction of Fake News (1)\')\nprint("Precision : {:.2f}%".format(prf_fake[0]*100))\nprint("Recall : {:.2f}%".format(prf_fake[1]*100))\nprint("F1Score : {:.2f}%".format(prf_fake[2]*100))')


# In[155]:


prf_f


# In[157]:


prf_r


# In[156]:


acc


# ### Random Forest

# In[158]:


from sklearn.ensemble import RandomForestClassifier

def RandomForest_train(X, y):
    
    model = RandomForestClassifier(n_estimators=100, criterion='gini')
    clf = model.fit(X,y)
    return clf


# In[172]:


get_ipython().run_cell_magic('time', '', 'prf_f = []\nprf_r = []\nacc = []\nfor i in range(10):\n    \n    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)\n    train_dataset.reset_index(drop=True, inplace=True)\n    test_dataset.reset_index(drop=True, inplace=True)\n    \n    vector_tfidf = TfidfVectorizer(stop_words=\'english\', max_df=0.90, min_df=2)\n    tfidf = vector_tfidf.fit_transform(train_dataset.body.astype(\'U\'))\n    \n    svd = TruncatedSVD(n_components=1000, n_iter=3, algorithm=\'randomized\')\n    tfidf_red = svd.fit_transform(tfidf)\n        \n    train_X = tfidf_red\n    train_y = np.array([1 if i==\'fake\' else 0 for i in train_dataset.type])\n    test_tfidf = vector_tfidf.transform(test_dataset.body.astype(\'U\'))\n    test_X = svd.transform(test_tfidf)\n    test_y = np.array([1 if i==\'fake\' else 0 for i in test_dataset.type])\n    \n    randforest_clf = RandomForest_train(train_X, train_y)\n    test_pred = randforest_clf.predict(test_X)\n    \n    prf_f.append(precision_recall_fscore_support(test_y,test_pred, pos_label=1, average=\'binary\')[:3])\n    prf_r.append(precision_recall_fscore_support(test_y,test_pred, pos_label=0, average=\'binary\')[:3])\n    \n    acc.append(np.mean(np.equal(test_y,test_pred)))\n    print("{} of 10".format(i))\n\nprf_fake = np.mean(prf_f,axis=0)\nprf_real = np.mean(prf_r,axis=0)\naccuracy = np.mean(acc)\n\nprint("RANDOM FOREST\\n")\nprint("\\nAccuracy : {:.2f}%".format(accuracy*100))\nprint(\'\\nPrediction of Real News (0)\')\nprint("Precision : {:.2f}%".format(prf_real[0]*100))\nprint("Recall : {:.2f}%".format(prf_real[1]*100))\nprint("F1Score : {:.2f}%".format(prf_real[2]*100))\nprint(\'\\nPrediction of Fake News (1)\')\nprint("Precision : {:.2f}%".format(prf_fake[0]*100))\nprint("Recall : {:.2f}%".format(prf_fake[1]*100))\nprint("F1Score : {:.2f}%".format(prf_fake[2]*100))')


# In[173]:


prf_f


# In[174]:


prf_r


# In[175]:


acc


# In[176]:


plt.figure(figsize=(15,10))
plt.bar(np.arange(0,train_X.shape[1],1),randforest_clf.feature_importances_,width=4)
plt.xlabel("features/tokens")
plt.savefig('randomForest_feature_importance.jpeg')
plt.show()


# ### Feed-Forward Neural Network Model

# In[200]:


def nn_train(train_X, train_y, epoch=2):
    
    feats = train_X.shape[1]
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(600, input_shape=(feats,), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(300, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_X, train_y, epochs=epoch,verbose=1)
    return model


# In[202]:


get_ipython().run_cell_magic('time', '', 'prf_f = []\nprf_r = []\nacc = []\nfor i in range(10):\n    \n    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)\n    train_dataset.reset_index(drop=True, inplace=True)\n    test_dataset.reset_index(drop=True, inplace=True)\n    \n    vector_tfidf = TfidfVectorizer(stop_words=\'english\', max_df=0.90, min_df=2)\n    tfidf = vector_tfidf.fit_transform(train_dataset.body.astype(\'U\'))\n    \n    svd = TruncatedSVD(n_components=1000, n_iter=3, algorithm=\'randomized\')\n    tfidf_red = svd.fit_transform(tfidf)\n        \n    train_X = tfidf_red\n    train_y = np.array([1 if i==\'fake\' else 0 for i in train_dataset.type])\n    test_tfidf = vector_tfidf.transform(test_dataset.body.astype(\'U\'))\n    test_X = svd.transform(test_tfidf)\n    test_y = np.array([1 if i==\'fake\' else 0 for i in test_dataset.type])\n    \n    NN_clf = nn_train(train_X, train_y, epoch=2)\n    test_pred = NN_clf.predict(test_X)\n    test_pred = ((test_pred>0.5)*1).flatten()\n    \n    prf_f.append(precision_recall_fscore_support(test_y,test_pred, pos_label=1, average=\'binary\')[:3])\n    prf_r.append(precision_recall_fscore_support(test_y,test_pred, pos_label=0, average=\'binary\')[:3])\n    \n    acc.append(np.mean(np.equal(test_y,test_pred)))\n    print("{} of 10".format(i))\n\nprf_fake = np.mean(prf_f,axis=0)\nprf_real = np.mean(prf_r,axis=0)\naccuracy = np.mean(acc)\n\nprint("Feed-Forward Neural Network Model\\n")\nprint("\\nAccuracy : {:.2f}%".format(accuracy*100))\nprint(\'\\nPrediction of Real News (0)\')\nprint("Precision : {:.2f}%".format(prf_real[0]*100))\nprint("Recall : {:.2f}%".format(prf_real[1]*100))\nprint("F1Score : {:.2f}%".format(prf_real[2]*100))\nprint(\'\\nPrediction of Fake News (1)\')\nprint("Precision : {:.2f}%".format(prf_fake[0]*100))\nprint("Recall : {:.2f}%".format(prf_fake[1]*100))\nprint("F1Score : {:.2f}%".format(prf_fake[2]*100))')


# In[203]:


train_X.shape


# In[204]:


prf_f


# In[205]:


prf_r


# In[206]:


acc


# In[275]:


test_pred = NN_clf.predict(test_X)
test_pred = ((test_pred>0.5)*1).flatten()
### confusion matrix ###
cm = pd.crosstab(test_pred, test_y, rownames=['Predicted'], colnames=['True'], margins = True)

### classification report ###
prf_fake = precision_recall_fscore_support(test_y,test_pred, pos_label=1, average='binary')
prf_real = precision_recall_fscore_support(test_y,test_pred, pos_label=0, average='binary')

acc = np.mean(np.equal(test_y,test_pred))

print("Feed-Forward Neural Network Model\n")
print('--------Confusion Matrix-------\n')
print(cm)
print("\nAccuracy : {:.2f}%".format(acc*100))
print('\nPrediction of Real News (0)')
print("Precision : {:.2f}%".format(prf_real[0]*100))
print("Recall : {:.2f}%".format(prf_real[1]*100))
print("F1Score : {:.2f}%".format(prf_real[2]*100))
print('\nPrediction of Fake News (1)')
print("Precision : {:.2f}%".format(prf_fake[0]*100))
print("Recall : {:.2f}%".format(prf_fake[1]*100))
print("F1Score : {:.2f}%".format(prf_fake[2]*100))


# ### Reccurant Neural Network 

# In[15]:


def filter_text(text):
    # removing html tags
    text = re.sub("<.*?>","",text)
    # removing websites, email addresses or any punctuation
    # changing "*http(s)*", "*www*" or "*@*" to " "
    # \S+ means anything except whitespace char >=1 times
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", "", text)
    # changing anything except alphabets to " "
    text = re.sub("[^a-zA-Z]"," ",text)
    text = text.lower()
    return text


# In[18]:


from nltk.tokenize import word_tokenize
# the dataset is cleaned of any special characters,
# and word stemming is done to remove any words with almost similar meaning and
# hence will help reduce the tf-idf dimensionality (number of unique words)
def clean_text(dataset):
    t0 = time()

    for i in dataset.index:
        # we will crop the first 1000 words for every article under assumption that 
        # we can learn the article's 'fakeness' from first 1000 words
        dataset.at[i,'body'] = ' '.join(filter_text(dataset.body[i]).split()[:1000])

    print("Time taken to cleanUp : %.3f min" % ((time()-t0)/60))


# In[19]:


# removing empty/nan article bodies and articles with less that 100 words
fake_dataset = filter_nan(fake_news)
real_dataset = filter_nan(real_news)


# In[20]:


# reducing the number of real_dataset datapoints around that of fake_dataset
# keeping the fake to non-fake news ratio close to presence in real world media
real_dataset = real_dataset.sample(n=45000,random_state=42)
real_dataset.reset_index(drop=True, inplace=True)


# In[21]:


dataset = pd.concat([real_dataset,fake_dataset])
dataset.reset_index(drop=True, inplace=True)


# In[22]:


clean_text(dataset)


# In[23]:


dataset = filter_nan(dataset)


# #### Data Pre-Processing

# In[24]:


get_ipython().run_cell_magic('time', '', "from tensorflow.keras.preprocessing.text import Tokenizer\nfrom tensorflow.keras.preprocessing.sequence import pad_sequences\n\nlabel = np.array([1 if t=='fake' else 0 for t in dataset.type])\n\n# create a tokenizer object \ntokenizer = Tokenizer()\n# fit on all the list of news articles will\n# create a dictionary of unique word occurances\ntokenizer.fit_on_texts(dataset.body)\n# convert each article to a sequence of integers\nsequences = tokenizer.texts_to_sequences(dataset.body)\n\n# word_index contains the dictionary of unique words\nword_index = tokenizer.word_index\nprint('Found %s unique tokens.' % len(word_index))\n\n# padding each sequence with 0 to make the input constant len of maximum size sequence\ndata = pad_sequences(sequences,padding='post',maxlen=1000)\n\n# shuffling and creating test and train set\nindices = np.arange(data.shape[0])\nnp.random.shuffle(indices)\ndata = data[indices]\nlabel = label[indices]\n\ntest_size = 0.2\n\ntest_samples = int(test_size*data.shape[0])\ntrain_X = data[:-test_samples]\ntrain_y = label[:-test_samples]\ntest_X = data[-test_samples:]\ntest_y = label[-test_samples:]")


# In[25]:


train_X.shape


# In[27]:


embedding_dim = 100
# +1 for the 0th index corresponding <unk> words 
vocab_size = len(word_index) + 1
units = 100
timesteps = train_X.shape[1]


# In[26]:


embeddings_index = {}
f = open('glove.6B.100d.txt', encoding='utf-8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    
f.close()
print('words found :',len(embeddings_index))


# In[28]:


embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    # if word not in embedding_index then the wordvec remains 0s
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[29]:


embedding_matrix.shape


# In[30]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,
                                    embedding_dim,
                                    weights = [embedding_matrix],
                                    input_length = train_X.shape[1],
                                    trainable=False))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv1D(64,4,activation='relu'))
# model.add(tf.keras.layers.MaxPool1D(pool_size=4))
model.add(tf.keras.layers.LSTM(units,
                               input_shape = (timesteps,embedding_dim),
                               recurrent_activation = 'sigmoid',
                               return_sequences = False,             # this means to carry forward the OUTPUT of units
                               recurrent_initializer = 'glorot_uniform',
                               ))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[31]:


model.fit(train_X, train_y, epochs=2)


# In[35]:


test_pred = model.predict(test_X)
test_pred = ((test_pred>0.5)*1).flatten()
### confusion matrix ###
cm = pd.crosstab(test_pred, test_y, rownames=['Predicted'], colnames=['True'], margins = True)

### classification report ###
prf_fake = precision_recall_fscore_support(test_y,test_pred, pos_label=1, average='binary')
prf_real = precision_recall_fscore_support(test_y,test_pred, pos_label=0, average='binary')

acc = np.mean(np.equal(test_y,test_pred))

print("LSTM Network Model\n")
print('--------Confusion Matrix-------\n')
print(cm)
print("\nAccuracy : {:.2f}%".format(acc*100))
print('\nPrediction of Real News (0)')
print("Precision : {:.2f}%".format(prf_real[0]*100))
print("Recall : {:.2f}%".format(prf_real[1]*100))
print("F1Score : {:.2f}%".format(prf_real[2]*100))
print('\nPrediction of Fake News (1)')
print("Precision : {:.2f}%".format(prf_fake[0]*100))
print("Recall : {:.2f}%".format(prf_fake[1]*100))
print("F1Score : {:.2f}%".format(prf_fake[2]*100))

