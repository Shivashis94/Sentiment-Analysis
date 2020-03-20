#!/usr/bin/env python
# coding: utf-8

# In[2906]:


from IPython.display import Image
Image("DataScience.jpg",width=8000,height=4000)


# # Problem Statement

# In this Competition, you are expected to create an end to end NLP framework to
# collect, analyse and perform sentiment analysis using supervised learning on user
# reviews about the latest Hollywood flick - “The Lion King (2019)”

# # Problem Description

# Online reviews are important because they have become a reference point for buyers
# across the globe and because so many people trust them when making purchase
# decisions.
# 
# Reviews are also important for Search Engine Optimization (SEO). Having positive
# reviews is also another way through which you can improve a website’s Search Engine
# visibility. The more that people talk about a brand online, the greater its visibility to
# Search Engines, such as Google, Yahoo and Bing.
# 
# For the audience and booking websites, analysing reviews is significant in
# understanding reviewer opinion about the film. In movie booking websites, 90% of
# people first check out online reviews before purchasing tickets.
# For the production house, analysing negative reviews can be useful for damage control.

# # Import Libraries

# In[1669]:


#Numpy & Pandas
import pandas as pd
import numpy as np

#For Extracting Training Data
import requests
import time

#Visualizations
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[130]:


headers = {
'Referer': 'https://www.rottentomatoes.com/m/the_lion_king_2019/reviews?type=user',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
'X-Requested-With': 'XMLHttpRequest',
}

url = 'https://www.rottentomatoes.com/napi/movie/9057c2cf-7cab-317f-876f-e50b245ca76e/reviews/user'

payload = {
'direction': 'next',
'endCursor': '',
'startCursor': '',
}


# In[131]:


s = requests.Session()


# In[132]:


r = s.get(url, headers=headers, params=payload) # GET Call
data = r.json()


# In[133]:


data


# # Extracting Audience Reviews

# In[135]:


endCursor = ''
startCursor = ''
df_reviews = pd.DataFrame(columns = data["reviews"][0].keys())
for i in range(0,300):
    payload = {
                'direction': 'next',
                'endCursor': endCursor,
                'startCursor': startCursor,
                }
    r = s.get(url, headers=headers, params=payload) # GET Call
    data = r.json()
    time.sleep(5)
    df_new = pd.DataFrame.from_dict(data["reviews"])
    df_reviews = df_reviews.append(df_new)
    endCursor = data["pageInfo"]['endCursor']
    startCursor = data["pageInfo"]['startCursor']


# In[2966]:


df_reviews.head()


# In[767]:


df_reviews.shape


# In[768]:


#Dumping Dataframe to a csv file for backup purpose
df_reviews.to_csv('TrainData.csv')


# In[1096]:


df_reviews_bk = df_reviews.copy(deep=True)


# In[2967]:


df_reviews_bk.head()


# In[1098]:


list_user = df_reviews_bk["user"].tolist()
list_user


# In[1100]:


list_RI = []


# In[1101]:


for i in range(0,len(list_user)):
    list_RI.append(list_user[i].get('userId'))


# In[1102]:


df_reviews_bk['ReviewID'] = list_RI


# In[775]:


df_reviews_bk.rename(columns={'displayName': 'Reviewer Name', 'review': 'Review','rating': 'Rating','updateDate': 'Date-of-Review'}, inplace=True)


# In[777]:


df_reviews_bk.head()


# In[778]:


#Initial Columns to Drop
cols_to_drop = ['createDate','displayImageUrl','timeFromCreation','user']


# In[779]:


df_reviews_bk.drop(cols_to_drop,axis=1,inplace=True)


# In[2968]:


# 1. The dataframe df_reviews_bk has the following features:-
# a. ReviewID
# b. Reviewer Name
# c. Review
# d. Rating
# e. Date-of-Review

# along with some other additional features like hasProfanity, hasSpoilers, isSuperReviewer, isVerified

#There are 3000 reviews in this dataframe

df_reviews_bk.head()


# # Label the Review Sentiment

# In[781]:


df_reviews_bk.dtypes


# In[782]:


#if ‘Rating’ > 3 then positive review else negative review – Create target attribute with
#the name: “sentiment” (binary class)

df_reviews_bk['sentiment'] = [0 if x >3.0 else 1 for x in df_reviews_bk['score']]                             
          


# In[2973]:


df_reviews_bk.head()


# In[788]:


# Drop the Rating attribute once the Target is derived. It should not be part of
# model building as independent attributes.

df_reviews_bk.drop(['Rating','score'],axis=1,inplace=True)


# In[2969]:


df_reviews_bk.head()


# In[791]:


df_reviews_bk.to_csv('CheckLabelData.csv')


# # Exploratory Data Analysis And Visualization

# In[425]:


df_reviews_mod = df_reviews_bk.copy(deep=True)


# In[426]:


df_reviews_mod.shape


# In[427]:


df_reviews_mod.drop(['Reviewer Name'],axis=1,inplace=True)


# In[428]:


df_reviews_mod.shape


# In[429]:


# To Check for NA Values
df_reviews_mod.isna().sum()


# In[430]:


from datetime import datetime
df_reviews_mod['Date-of-Review'] = pd.to_datetime(df_reviews_mod['Date-of-Review'])
df_reviews_mod['Date-of-Review'].head()


# In[431]:


df_reviews_mod['DayOfReview'] = df_reviews_mod['Date-of-Review'].dt.strftime("%d")


# In[432]:


df_reviews_mod.head()


# In[433]:


#Drop Date-of-Review and ReviewDate
df_reviews_mod.drop(['Date-of-Review'],axis=1,inplace=True)


# In[2970]:


df_reviews_mod.head()


# In[435]:


#Convert Approprate Columns to numeric
for item in ['DayOfReview']:
    df_reviews_mod[item] = df_reviews_mod[item].astype('int64')


# In[436]:


df_reviews_mod.dtypes


# In[437]:


df_reviews_mod.shape


# In[438]:


#Import Text Libraries
import nltk
import re
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import RegexpTokenizer as regextoken


# In[2975]:


df = df_reviews_mod.reset_index()
del df['index']


# In[1081]:


def process_text(text):
    sent = text.lower()
    word = word_tokenize(sent)
    clwrds = [w for w in word if not w in stop]
    lmtzr = WordNetLemmatizer()
    lmtzrs = [lmtzr.lemmatize(i) for i in clwrds]
    ln = len(lmtzrs)
    rt = [ln, " ".join(lmtzrs)]
    return(rt)


# In[1676]:


preprocessed_text = []


# In[1677]:


#Perform Text PreProcessing on the Review Column values which has text's values.
for i in range(len(df)):
    text=df['Review'][i]
    preprocessed_text.append(process_text(text))
        


# In[1678]:


preprocessed_text


# In[572]:


df['ReviewTextPreprocessed'] = 0
df['NumWordsInReview'] = 0


# In[580]:


#regex = regextoken('\w+')
for i in range(0,3000):
    x = preprocessed_text[i][1]
    y = preprocessed_text[i][0]
    df['ReviewTextPreprocessed'][i] = x
    df['NumWordsInReview'][i] = y


# In[794]:


df.to_csv('TextPreprocessedData.csv')


# In[644]:


#remove punctuations and special chracters
df['ReviewTextPreprocessed'] = df['ReviewTextPreprocessed'].str.replace('[^\w\s]','')


# In[2976]:


df.head()


# In[905]:


df_reviews_lk = df.copy(deep=True)   


# In[2946]:


# Distribution of postive (i.e 0) and negative (i.e 1) sentiments
sns.countplot(x='sentiment', data=df_reviews_lk)
plt.show()


# In[647]:


def plot_bar(col_name):
    # create a table with value counts
    temp = df_reviews_lk[col_name].value_counts()
    # creating a Bar chart object of plotly
    data = [go.Bar(
            x=temp.index.astype(str), # x axis values
            y=np.round(temp.values.astype(float)/temp.values.sum(),4)*100, # y axis values
            text = ['{}%'.format(i) for i in np.round(temp.values.astype(float)/temp.values.sum(),4)*100],
        # text to be displayed on the bar, we are doing this to display the '%' symbol along with the number on the bar
            textposition = 'auto', # specify at which position on the bar the text should appear
        marker = dict(color = '#0047AB'),)] # change color of the bar
    # color used here Cobalt Blue
     
    layout_bar = generate_layout_bar(col_name=col_name)

    fig = go.Figure(data=data, layout=layout_bar)
    return iplot(fig)


# In[648]:


def generate_layout_bar(col_name):
    layout_bar = go.Layout(
        autosize=False, # auto size the graph? use False if you are specifying the height and width
        width=800, # height of the figure in pixels
        height=600, # height of the figure in pixels
        title = "Distribution of {} column".format(col_name), # title of the figure
        # more granular control on the title font 
        titlefont=dict( 
            family='Courier New, monospace', # font family
            size=14, # size of the font
            color='black' # color of the font
        ),
        # granular control on the axes objects 
        xaxis=dict( 
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the x axis
            color='black'  # color of the font
            )
        ),
        yaxis=dict(
            title='Percentage',
            titlefont=dict(
                size=14,
                color='black'
            ),
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the y axis
            color='black' # color of the font
            )
        ),
        font = dict(
            family='Courier New, monospace', # font family
            color = "white",# color of the font
            size = 12 # size of the font displayed on the bar
                )  
        )
    return layout_bar


# In[649]:


#Distribution of target attribute - sentiment (Majority sentiment is positive i.e. 0 in the train data)
plot_bar('sentiment')


# In[650]:


#Distribution of DayofReview column - Maximum Reviews are given by the users on the 5th Day which is a post weekend Review
plot_bar('DayOfReview')


# In[2952]:


#Number a times a Review of a particular count occurs

plt.figure()
df_reviews_lk['NumWordsInReview'].value_counts().plot(kind='bar',
                                  figsize=(25,75),
                                  color="blue",
                                  alpha = 0.7,
                                  fontsize=20)
plt.title('Review of a Particular count')
plt.ylabel('Count',fontsize = 14)
plt.xlabel('Number a times a Review of a particular count occurs',fontsize = 5)
plt.grid()
plt.show()


# In[653]:


reviews = df_reviews_lk.Review.str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(reviews)
stop_words = set(stopwords.words('english'))
stopwordrmvd = [w for w in tokens if not w in stop_words]
lmt = WordNetLemmatizer()
WordLemmatized = [lmt.lemmatize(i) for i in stopwordrmvd]
frequency_dist = nltk.FreqDist(WordLemmatized)


# In[654]:


#WordCloud helps to create word clouds by placing words on a canvas randomly, with sizes proportional to their frequency in the text.
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[655]:


def ngrams_process(input_sentence_tokens):
    ngram_list = []
    for sentence in input_sentence_tokens:
        ngram_sent = nltk.ngrams(sentence, 2)
        ngram_list = ngram_list + list(ngram_sent)
    return ngram_list

def text_process(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    sentence_tokens = [word_tokenize(sentence) for sentence in sentences] #list of lists
    tokens = [] # initialising variable
    token_lamt=[]
    tokens = [[word.lower() for word in sent if word not in stop] for sent in sentence_tokens]
    lmt = WordNetLemmatizer()
    token_lamt = [[lmt.lemmatize(word) for word in sent] for sent in tokens]
    return token_lamt


# In[656]:


string_ngram_list = []
sent_token_list = []


# In[657]:


for k in range(0,3000):
    sent_tokn = text_process(df_reviews_lk['Review'][k])
    sent_token_list.append(sent_tokn)   


# In[658]:


sent_token_list


# In[659]:


for i in range(0,len(sent_token_list)):
    string_ngrams = ngrams_process(sent_token_list[i])
    string_ngram_list.append(string_ngrams)   


# In[660]:


string_ngram_list


# In[663]:


# List to store bigrams and their frequency
ngram_freq_list = []


# In[664]:


for j in range(0,3000):
    ngram_freq = nltk.FreqDist()
    for ngram in string_ngram_list[j]:
        ngram_freq[ngram] += 1
    ngram_freq_list.append(ngram_freq.most_common())


# In[665]:


ngram_freq_list

#Frequency of bigrams for each review


# In[666]:


df_reviews_lk


# In[1941]:


df_reviews_lk['hasProfanity'] = [0 if x == True else 1 for x in df_reviews_lk['hasProfanity']]
df_reviews_lk['hasSpoilers'] = [0 if x == True else 1 for x in df_reviews_lk['hasSpoilers']] 
df_reviews_lk['isSuperReviewer'] = [0 if x == True else 1 for x in df_reviews_lk['isSuperReviewer']] 
df_reviews_lk['isVerified'] = [0 if x == True else 1 for x in df_reviews_lk['isVerified']] 


# In[1942]:


df_reviews_lk


# In[1943]:


df_reviews_lk.dtypes


# In[1944]:


#corelation matrix
sns.heatmap(df_reviews_lk.corr(),annot=True,fmt="0.2f",cmap="coolwarm")
plt.show()

## DayOfReview, hasProfanity,hasSpoilers,isSuperReviewer is -vely correlated with the target - "sentiment"
## NumWordsInReview, isVerified is +vely correlated with the target - "sentiment"


# In[1945]:


data =  df_reviews_lk.copy(deep=True)


# In[1946]:


data


# In[1947]:


data['sentiment'].value_counts()


# In[1948]:


data.dtypes


# # Train-Validation Split

# In[2574]:


data.head()


# In[2916]:


X = data['ReviewTextPreprocessed']
Y = data['sentiment']


# In[2917]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,Y, test_size=0.3, random_state=646)


# In[2351]:


X_train.shape


# In[2352]:


X_val.shape


# In[2353]:


y_train.shape


# In[2354]:


y_val.shape


# In[2602]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import SpatialDropout1D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from xgboost import XGBClassifier
import tensorflow as tf


# # Model Building

# ## Naive Bayes 

# In[2220]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2096]:


clf = MultinomialNB().fit(train_vectors,y_train)


# In[2097]:


predicted_val = clf.predict(validation_vectors)


# In[2098]:


predicted_train = clf.predict(train_vectors)


# In[2099]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2100]:


print(classification_report(y_val, predicted_val))
print("\nValdation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # Logistic

# In[2268]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2269]:


logreg = LogisticRegression().fit(train_vectors, y_train)


# In[2270]:


predicted_val = logreg.predict(validation_vectors)


# In[2271]:


predicted_train = logreg.predict(train_vectors)


# In[2272]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2273]:


print(classification_report(y_val, predicted_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # Decision Tree

# In[2107]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2108]:


dtc = DecisionTreeClassifier()

param_grid = {"max_depth" : [14, 16, 18, 20, 22, 24],
              "min_samples_leaf" : [5, 10, 15, 20, 25],
              "class_weight" : ['balanced'],
              "min_samples_split": [2, 15, 30],
              "criterion": ['entropy', 'gini']}
 
dtc_cv_grid = RandomizedSearchCV(estimator = dtc, param_distributions = param_grid, cv = 3, n_iter=7)


# In[2109]:


dtc_cv_grid.estimator


# In[2110]:


dtreg = dtc_cv_grid.fit(train_vectors,y_train)


# In[2111]:


predicted_val = dtreg.predict(validation_vectors)


# In[2112]:


predicted_train = dtreg.predict(train_vectors)


# In[2113]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2114]:


print(classification_report(y_val, predicted_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # Random Forest

# In[2918]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2953]:


rfe = RandomForestClassifier(n_estimators=500,max_features=1000)

#kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=123)


# In[2939]:


#dt_param_grid = {"n_estimators" : [500, 800, 1000],
#                 "max_features" : [1000, 1200,1400]}

#rfe_grid = GridSearchCV(estimator = rfe, param_grid=dt_param_grid, cv=kfold)


# In[2954]:


rfereg = rfe.fit(train_vectors,y_train)


# In[2955]:


predicted_val = rfereg.predict(validation_vectors)


# In[2956]:


predicted_train = rfereg.predict(train_vectors)


# In[2957]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2958]:


print(classification_report(y_val, predicted_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # SVM

# In[2139]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2140]:


## Create an SVC object and print it to see the default arguments
svc = SVC(kernel='linear',random_state=123)
svc


# In[2141]:


svmreg = svc.fit(train_vectors,y_train)


# In[2142]:


predicted_val = svmreg.predict(validation_vectors)


# In[2143]:


predicted_train = svmreg.predict(train_vectors)


# In[2144]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2145]:


print(classification_report(y_val, predicted_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # XGBoost

# In[2203]:


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2205]:


XGB = XGBClassifier(n_jobs=-1)
 
# Use a grid over parameters of interest
param_grid = {
     'colsample_bytree': np.linspace(0.5, 0.9, 5),
     'n_estimators':[100],
     'max_depth': [10, 15, 20],
    'class_weight':['balanced']
}

 
CV_XGB = GridSearchCV(estimator=XGB, param_grid=param_grid, cv= 4)


# In[2206]:


xgreg = CV_XGB.fit(train_vectors,y_train)


# In[2207]:


predicted_val = xgreg.predict(validation_vectors)


# In[2208]:


predicted_train = xgreg.predict(train_vectors)


# In[2209]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2210]:


print(classification_report(y_val, predicted_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # MLP

# In[2288]:


vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
validation_vectors = vectorizer.transform(X_val)
print(train_vectors.shape, validation_vectors.shape)


# In[2289]:


mlp = MLPClassifier(hidden_layer_sizes=(100,),activation='relu',verbose=False)


# In[2290]:


mlpreg = mlp.fit(train_vectors, y_train)


# In[2291]:


predicted_val = mlpreg.predict(validation_vectors)


# In[2292]:


predicted_train = mlpreg.predict(train_vectors)


# In[2293]:


print(classification_report(y_train, predicted_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predicted_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predicted_train,pos_label=1))


# In[2294]:


print(classification_report(y_val, predicted_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predicted_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predicted_val,pos_label=1))


# # Stochastic Gradient Descent

# In[2185]:


#grid search result
vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2), max_features=50000,max_df=0.5,use_idf=True, norm='l2') 
counts = vectorizer.fit_transform(X_train)
classifier = SGDClassifier(alpha=1e-05,max_iter=50,penalty='elasticnet')
targets = y_train
classifier = classifier.fit(counts, targets)
predictions_val = classifier.predict(vectorizer.transform(X_val))
predictions_train = classifier.predict(vectorizer.transform(X_train))


# In[2186]:


print(classification_report(y_train, predictions_train))
print("\nTraining data f1-score for sentiment '0'",f1_score(y_train,predictions_train,pos_label=0))
print("\nTraining data f1-score for sentiment '1'",f1_score(y_train,predictions_train,pos_label=1))


# In[2187]:


print(classification_report(y_val, predictions_val))
print("\nValidation data f1-score for sentiment '0'",f1_score(y_val,predictions_val,pos_label=0))
print("\nValidation data f1-score for sentiment '1'",f1_score(y_val,predictions_val,pos_label=1))


# # LSTM

# In[2827]:


max_length = 500
max_features = 7000


# In[2828]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['ReviewTextPreprocessed'],)
X = tokenizer.texts_to_sequences(data['ReviewTextPreprocessed'])
X = pad_sequences(X,maxlen=max_length)


# In[2829]:


X.shape


# In[2830]:


Y = data['sentiment'].values
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_val.shape,Y_val.shape)


# In[2833]:


from keras.optimizers import Adam


# In[2834]:


model = Sequential()
model.add(Embedding(max_features,100,mask_zero=True))
model.add(LSTM(64,dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()


# In[2835]:


epochs = 5
batch_size = 32


# In[2836]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)


# In[2837]:


score,acc = model.evaluate(X_val, Y_val, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[2838]:


pred = model.predict_classes(X_val)


# In[2839]:


print(classification_report(Y_val, pred))


# # CNN

# In[2877]:


tokenizer = Tokenizer(num_words=5000)


# In[2878]:


tokenizer.fit_on_texts(X_train)


# In[2879]:


X_train_tok = tokenizer.texts_to_sequences(X_train)
X_val_tok = tokenizer.texts_to_sequences(X_val)


# In[2880]:


vocab_size = len(tokenizer.word_index) + 1
vocab_size


# In[2881]:


maxlen = 500


# In[2882]:


X_train_pad = pad_sequences(X_train_tok, padding='post', maxlen=maxlen)
X_val_pad = pad_sequences(X_val_tok, padding='post', maxlen=maxlen)


# In[2883]:


embedding_dim = 100
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[2884]:


history = model.fit(X_train_pad, y_train,epochs=7,verbose=False,validation_data=(X_val_pad, y_val),batch_size=32)


# In[2885]:


score,acc = model.evaluate(X_val_pad, y_val, verbose=2,batch_size=10)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[2886]:


pred = model.predict_classes(X_val_pad)


# In[2887]:


print(classification_report(y_val, pred))


# # Test Data Preprocessing

# In[1671]:


testdata = pd.read_csv("test-1566619745327.csv")


# In[1672]:


testdata.head()


# In[1679]:


preprocessed_text_test = []


# In[1680]:


#Perform Text PreProcessing on the Review Column values which has text's values.
for i in range(len(testdata)):
    texttestdata=testdata['review'][i]
    preprocessed_text_test.append(process_text(texttestdata))


# In[1681]:


preprocessed_text_test


# In[1685]:


testdata['reviewPreprocessed'] = 0


# In[1692]:


#regex = regextoken('\w+')
for i in range(0,1200):
    x = preprocessed_text_test[i][1]
    testdata['reviewPreprocessed'][i] = x    


# In[1693]:


#Back up file
testdata.to_csv("TestDataPreprocessed.csv")


# In[1694]:


testdata.head()


# In[1695]:


#remove punctuations and special chracters
testdata['reviewPreprocessed'] = testdata['reviewPreprocessed'].str.replace('[^\w\s]','')


# In[1696]:


testdata.head()


# In[1697]:


#Backup
testdata.to_csv("TestReviewFinal.csv")


# # Prediction on Test Data

# # Applying SVM

# In[2172]:


M = testdata['reviewPreprocessed']


# In[2173]:


M.shape


# In[2174]:


#Applying the vectorizer object if TF-IDF for transforming test data
test_vectors = vectorizer.transform(M)


# In[2175]:


test_vectors.shape


# In[2176]:


#Applying SVM on test data
predictions_test = svmreg.predict(test_vectors)


# In[2177]:


predictions_test


# In[2178]:


#Reading the original test file
#testdf = pd.read_csv("test-1566619745327.csv")
testdf = testdata.copy(deep = True)


# In[2179]:


testdf['sentiment'] = predictions_test


# In[2180]:


testdf.head()


# In[2182]:


testdf.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[2183]:


testdf.head()


# In[2184]:


testdf.to_csv('submission1-SVM_1.csv',index = False)


# # Applying SDG

# In[2190]:


M = testdata['reviewPreprocessed']


# In[2191]:


M.shape


# In[2192]:


#Applying the vectorizer object if TF-IDF for transforming test data
test_vectors = vectorizer.transform(M)


# In[2193]:


test_vectors.shape


# In[2194]:


# Applying SDG classfier object on test data
predictions_test = classifier.predict(test_vectors)


# In[2195]:


predictions_test


# In[2196]:


testdf1 = testdata.copy(deep = True)


# In[2197]:


testdf1['sentiment'] = predictions_test


# In[2198]:


testdf1.head()


# In[2199]:


#testdf1.to_csv('sub2.csv')


# In[2200]:


testdf1.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[2201]:


testdf1.head()


# In[2202]:


testdf1.to_csv('submission2-SDG.csv',index = False)


# # Applying Decision Tree

# In[1887]:


M = testdata['reviewPreprocessed']


# In[1888]:


M.shape


# In[1889]:


#Applying the vectorizer object if TF-IDF for transforming test data
test_vectors = vectorizer.transform(M)


# In[1890]:


test_vectors.shape


# In[1891]:


# Applying Decsion Tree classifier object on test data
predictions_test = dtreg.predict(test_vectors)


# In[1892]:


predictions_test


# In[1876]:


testdf2 = testdata.copy(deep = True)


# In[1893]:


testdf2['sentiment'] = predictions_test


# In[1894]:


#testdf2.to_csv('submission3.csv',index=False)


# In[1896]:


testdf2.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[1897]:


testdf2.head()


# In[1898]:


testdf2.to_csv('submission3-DT.csv',index=False)


# # Apply MLP

# In[2295]:


M = testdata['reviewPreprocessed']


# In[2296]:


M.shape


# In[2297]:


test_vectors = vectorizer.transform(M)


# In[2298]:


test_vectors.shape


# In[2299]:


predictions_test = mlpreg.predict(test_vectors)


# In[2300]:


predictions_test


# In[2301]:


testdf3 = testdata.copy(deep = True)


# In[2302]:


testdf3['sentiment'] = predictions_test


# In[2304]:


testdf3.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[2306]:


testdf3.to_csv('submission-MLP_1.csv',index=False)


# # Applying Logistic Regression

# In[2274]:


M = testdata['reviewPreprocessed']


# In[2275]:


M.shape


# In[2277]:


test_vectors = vectorizer.transform(M)


# In[2278]:


test_vectors.shape


# In[2279]:


# Applying Logistic Regression classifier object on test data
predictions_test = logreg.predict(test_vectors)


# In[2280]:


predictions_test


# In[2281]:


testdf4 = testdata.copy(deep = True)


# In[2282]:


testdf4['sentiment'] = predictions_test


# In[2283]:


testdf4.head()


# In[2284]:


testdf4.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[2285]:


testdf4.to_csv('submission-LR.csv',index=False)


# # Apply LSTM 

# In[2752]:


M = testdata['reviewPreprocessed']


# In[2753]:


M.shape


# In[2754]:


X_test = tokenizer.texts_to_sequences(M)


# In[2755]:


X_test = pad_sequences(X_test,maxlen=max_length)


# In[2756]:


pred_test = model.predict_classes(X_test)


# In[2760]:


testdf5 = testdata.copy(deep = True)


# In[2761]:


testdf5['sentiment'] = predictions_test


# In[2762]:


testdf5.head()


# In[2763]:


testdf5.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[2764]:


testdf5.to_csv('submission-LSTM.csv',index=False)


# # Apply CNN

# In[2888]:


M = testdata['reviewPreprocessed']


# In[2889]:


M.shape


# In[2890]:


X_test = tokenizer.texts_to_sequences(M)


# In[2891]:


X_test = pad_sequences(X_test,maxlen=max_length)


# In[2892]:


pred_test = model.predict_classes(X_test)


# In[2893]:


testdf6 = testdata.copy(deep = True)


# In[2896]:


testdf6['sentiment'] = pred_test


# In[2897]:


testdf6.head()


# In[2899]:


testdf6.drop(['review','reviewPreprocessed'],axis = 1,inplace=True)


# In[2900]:


testdf6.to_csv('submission-CNN.csv',index=False)


# # Clustering

# In[2533]:


data_cluster = df_reviews_lk.copy(deep=True)


# In[2534]:


data_cluster.head()


# In[2535]:


ReviewForClustering = data_cluster['Review'].values


# In[2559]:


ReviewForClustering.shape


# In[2537]:


from sklearn.cluster import KMeans


# In[2538]:


vectorizer_cluster = TfidfVectorizer(stop_words='english')
X = vectorizer_cluster.fit_transform(ReviewForClustering)


# In[2539]:


X.shape


# In[2540]:


NumOfClusters = 2
model = KMeans(n_clusters=NumOfClusters, init='k-means++', max_iter=100, n_init=1)


# In[2541]:


km = model.fit(X)


# In[2915]:


km.labels_


# In[2542]:


c1 = km.labels_.tolist()


# In[2543]:


data_cluster['ReviewClusterNumber'] = c1


# In[2546]:


cols = ['hasProfanity','hasSpoilers','isSuperReviewer','isVerified','ReviewID','NumWordsInReview','DayofReview']


# In[2558]:


data_cluster.drop(cols,axis=1, inplace=True)


# In[2550]:


data_cluster.head()


# In[2551]:


data_cluster.to_csv('ClusterComparision.csv',index=False)


# In[2567]:


#Comparing Cluster labels with train data target attribute - 'sentiment'

# Positive Reviews :- Target attribute - sentiment mapped to '0'
# Negative Reviews :- Target attribute - sentiment mapped to '1'

# There are 2173 Positive Reviews and 827 Negative Reviews in the training data

# k-means clustering algorithm applied to 'reviews' attribute of training data and lablleled into two clusters namely 0 and 1

# The cluster label to a which a particular review of the training data belongs is populated to a new attribute ReviewClusterNumber

# Among the Positive Reviews, 1775 reviews belong to ReviewClusterNumber '0' and 398 reveiws belong to ReviewClusterNumber '1'

# Among the Negative Reviews, 823 reviews belong to ReviewClusterNumber '0' and only 4 reviews belong to ReviewClusterNumber '1'

# Percentage wise comparsion

# Positive Reviews - 81.684 % of the reviews belong to ReviewClusterNumber '0' and 18.315% of them belong to ReviewClusterNumber '1'

# Negative Reviews - 99.516 % of the reviews belong to ReviewClusterNumber '0' and 0.0048% of them belong to ReviewClusterNumber '1'


# In[2568]:


### Reversing the list so that index of max element is in 0th index
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]


# In[2571]:


# terms = vectorizer_cluster.get_feature_names()


# In[2963]:


# for i in range(NumOfClusters):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :20]:
#         print('%s' % terms[ind])


# # Conclusion

# In[2962]:


# Among the traditonal ML algorithms,SVM and SDG gave a decent F1 score on the provided test data

# F1 score with SVM on test data - 0.68
# F1 score with SDG on test data - 0.69

# Among the Deep Learning algorithms,LSTM and CNN gave a decent F1 score on the provided test data

# F1 score with LSTM on test data - 0.62
# F1 score with CNN on test data - 0.59


# In[2965]:


Image("ThankYou.jpg")

