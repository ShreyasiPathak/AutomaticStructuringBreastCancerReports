import re
import csv
import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def get_text_length(x):
    ar=np.array([np.log10(len(t)) for t in x]).reshape(-1, 1)
    #maxAr=np.amax(ar)
    #return ar/float(maxAr)
    return ar

def get_endofline_symbol(x):
    lastLetDic={}
    count=0
    ar=np.array([t[-1] for t in x]).reshape(-1,1)
    #ar1=np.zeros([len(ar),1])
    for i in range(len(ar)):
        #if lastLetDic.has_key(ar[i][0]):
        #    ar1[i][0]=lastLetDic[ar[i][0]]
        #else:
        #    count=count+1
        #    lastLetDic[ar[i][0]]=count
        #    ar1[i][0]=lastLetDic[ar[i][0]]
        if ar[i][0]==':':
            ar[i][0]=1
        elif ar[i][0]==',':
            ar[i][0]=2
        else:
            ar[i][0]=0
    ar=ar.astype(int)
    #print ar1
    return ar

def naivebayesfunc(trainData, trainLabel):
    '''classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])'''
    classifier = Pipeline([
        ('features',FeatureUnion([
            ('text',Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),
            ('count', FunctionTransformer(get_text_length, validate=False)),
            ('last symbol', FunctionTransformer(get_endofline_symbol, validate=False))
        ])),
        ('clf', MultinomialNB())
    ])
    predicted = cross_val_predict(classifier, trainData, trainLabel, cv=5)
    #transValue=classifier.fit_transform(trainData)
    #clf=MultinomialNB.fit(trainData,trainLabel)
    #predicted=clf.predict(trainData)
    #parameter=classifier.named_steps['features']._iter()#text.named_steps['vectorizer'].get_feature_names()
    #featureList=list(parameter)#[1].named_steps['vectorizer'].get_feature_names()
    return predicted

def svmfunc(trainData, trainLabel):
    '''classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))
    ])'''
    classifier = Pipeline([
        ('features',FeatureUnion([
            ('text',Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),
            ('count', FunctionTransformer(get_text_length, validate=False)),
            ('last symbol', FunctionTransformer(get_endofline_symbol, validate=False))
        ])),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))
    ])
    predicted = cross_val_predict(classifier, trainData, trainLabel, cv=5)
    return predicted

def randomforest(trainData, trainLabel):
    classifier = Pipeline([
        ('features',FeatureUnion([
            ('text',Pipeline([
                ('vectorizer', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),
            ('count', FunctionTransformer(get_text_length, validate=False)),
            ('last symbol', FunctionTransformer(get_endofline_symbol, validate=False))
        ])),
        ('clf', RandomForestClassifier(max_depth=10, random_state=0))
    ])
    predicted = cross_val_predict(classifier, trainData, trainLabel, cv=5)
    return predicted

def results(predictTestLabel, trueTestLabel):
    accuracy=np.mean(predictTestLabel == trueTestLabel) 
    print "Accuracy:",metrics.accuracy_score(trueTestLabel, predictTestLabel), accuracy
    print "F1_micro", f1_score(trueTestLabel, predictTestLabel, average="micro")
    print "Test result:",metrics.classification_report(trueTestLabel, predictTestLabel)
    print "Confusion matrix:",metrics.confusion_matrix(trueTestLabel, predictTestLabel)

sentence=[]
label=[]
ct=0

f=open('Sentence180_H_NH.csv','r')
fRead=csv.reader(f)
next(fRead)
'''f1Write=csv.writer(f1)
f1Write.writerow(['Sentence','Heading'])'''

for row in fRead:
    ct=ct+1
    sentence.append(row[0])
    if row[1]=='':
        row[1]='Not Heading'
        #print row[1]
    label.append(row[1])

'''for line in EachLine:
    for li in line:
        #print li,'\n'
        if li!='':
            sent=[li]
            f1Write.writerow(sent)'''

predictTestLabelNB=naivebayesfunc(sentence, label)
print "Naive Bayes"
results(predictTestLabelNB, label)
predictTestLabelSVM=svmfunc(sentence, label)
print "SVM"
results(predictTestLabelSVM, label)
predictTestLabelRF=randomforest(sentence, label)
print "RF"
results(predictTestLabelRF, label)