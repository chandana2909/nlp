#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install emot')


# In[2]:


text1="What are you saying😂. I am the boss 😎,and why are you so 😒" 


# In[3]:


import re 
from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO


# In[4]:


def converting_emojis(text):
    for emot in UNICODE_EMOJI:
        text=text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","")\
                                        .replace(":","").split()))
converting_emojis(text1)


# In[5]:


def emoji_removal(string):
    emoji_unicodes=re.compile("[]")


# In[9]:


import pandas as pd
dataset=pd.read_csv(r"C:\Users\chinn\OneDrive\Documents\hate_speech.csv")
dataset.head()


# In[10]:


dataset.shape


# In[11]:


dataset.label.value_counts()


# In[12]:


for index,tweet in enumerate(dataset["tweet"][10:5]):
    print(index+1,"-",tweet)


# In[13]:


import re
def clean_text(text):
    text=re.sub(r'[^a-zA-Z\']', ' ',text)
    text=re.sub(r'[^\x00-\x7F]+', ' ',text)
    text=text.lower()
    return text


# In[14]:


dataset['clean_text']=dataset.tweet.apply(lambda x: clean_text(x))


# In[15]:


dataset.head(10)


# In[16]:


from nltk.corpus import stopwords
len(stopwords.words('english'))


# In[17]:


stop=stopwords.words('english')


# In[18]:


def gen_freq(text):
    word_list=[]
    for tw_words in text.split():
        word_list.extend(tw_words)
    word_freq=pd.Series(word_list).value_counts()
    word_freq=word_freq.drop(stop,errors='ignore')
    return word_freq


# In[19]:


def any_neg(words):
    for word in words:
        if word in ['n','no','non','not'] or re.search(r"\wn't",word):
            return 1
        else:
            return 0


# In[32]:


def any_rare(words,rare_100):
    for word in words:
        if word in rare_100:
            return 1
        else:
            return 0


# In[33]:


def is_question(words):
    for word in words:
        if word in ['when','what','how','why','who','where']:
            return 1
        else:
            return 0


# In[34]:


word_freq = gen_freq(dataset.clean_text.str)
rare_100 = word_freq[-100:] 
dataset['word_count'] = dataset.clean_text.str.split().apply(lambda x: len(x))
dataset['any_neg'] = dataset.clean_text.str.split().apply(lambda x: any_neg(x))
dataset['is_question'] = dataset.clean_text.str.split().apply(lambda x: is_question(x))
dataset['any_rare'] = dataset.clean_text.str.split().apply(lambda x: any_rare(x, rare_100))
dataset['char_count'] = dataset.clean_text.apply(lambda x: len(x))


# In[35]:


dataset.head(10)


# In[38]:


from sklearn.model_selection import train_test_split
X=dataset[['word_count','any_neg','any_rare','char_count','is_question']]
y=dataset.label
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[39]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model=model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[40]:


from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,pred)*100,"%")


# In[44]:


from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier()
clf_rf.fit(X_train,y_train)
rf_pred=clf_rf.predict(X_test).astype(int)


# In[46]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test,rf_pred))
print("Accuracy:",accuracy_score(y_test,rf_pred))


# In[49]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(class_weight='balanced')
logreg.fit(X_train,y_train)


# In[50]:


y_pred=logreg.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




