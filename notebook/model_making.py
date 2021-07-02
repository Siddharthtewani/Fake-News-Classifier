#%%
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

print("Done")
#%%
df=pd.read_csv(r"C:\Users\siddh\Desktop\Projects\Fake News Classifier\datasets\fake-news\train.csv")
df.head()
#%%
df.shape
# %%
df.info()
# %%
df.isnull().sum()
#%%
df=df.drop(['author'],axis=1)
#%%
df.isnull().sum()
#%%
df.head()
# %%
df=df.set_index("id")
#%%
sns.countplot(df["label"])
# %%
df["label"].value_counts()
# %%
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps= PorterStemmer()
# %%

def preproccessing_work(dataframe):
    print(dataframe.isnull().sum())
    dataframe["text"].fillna("",inplace=True)
    dataframe["title"].fillna("",inplace=True)
    text_lst=[]
    for i in range(len(dataframe["text"])):
        print(i)
        text=re.sub('[^a-zA-Z]',' ',dataframe["text"][i])
        text=text.lower()
        text=text.split()
        for words in text :
            if not words in set(stopwords.words('English')):
                text= ps.stem(words)

        text="".join(text)

        text_lst.append(text)

    title_lst=[]
    for i in range(len(dataframe["title"])):
        print(i)
        second_title=re.sub('[^a-zA-Z]',' ',dataframe["title"][i])
        second_title=second_title.lower()
        second_title=second_title.split()
        for words in second_title :
            if not words in set(stopwords.words('English')):
                second_title= ps.stem(words)

        second_title="".join(second_title)

        title_lst.append(second_title)

    tfidf=TfidfVectorizer()
    X=tfidf.fit_transform(text_lst).toarray()
    Y=tfidf.fit_transform(title_lst).toarray()

    x1=pd.DataFrame(X)
    x_x=pd.DataFrame(Y)
    for i in range(x_x.shape[1]):
        print(i)
        x=x_x.rename(columns={i:"{}_title".format(i)})
    
    final=pd.concat([x1,x],axis=1)
    return final

# %%
# lem.lemmatize(word) for word in text if word not in set(stopwords.words("english"))
y=df["label"]
# %%
final=preproccessing_work(df)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(final,y,test_size=0.2,random_state=102)
# %%
print("X Train--->",X_train.shape)
print("Y Train",Y_train.shape)
print("X test--->",X_test.shape)
print("Y test",Y_test.shape)
# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import itertools

# %%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#%%
for al in np.arange(0,1,0.1):
    classifier=MultinomialNB(alpha=al)
    classifier.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    score = metrics.accuracy_score(Y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(Y_test, pred)
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
# %%
testing=pd.read_csv(r"C:\Users\siddh\Desktop\Projects\Fake News Classifier\datasets\fake-news\test.csv")
# %%
testing.head()
# %%
testing.shape
# %%
test_data=preproccessing_work(testing)
#%%
pred_test=classifier.predict(test_data)
# %%
