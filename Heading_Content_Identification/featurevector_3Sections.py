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
import matplotlib
import matplotlib.pyplot as plt

import confusionmatrix_heatmap

def get_text_length(x):
    #ar=np.array([np.log10(len(t)) for t in x]).reshape(-1, 1)
    ar=np.array([len(t) for t in x]).reshape(-1, 1)
    #maxAr=np.amax(ar)
    #return ar/float(maxAr)
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
                ('vectorizer', CountVectorizer(max_df=0.6)),
                #('tfidf', TfidfTransformer())
            ])),
            #('length', Pipeline([
            ('count', FunctionTransformer(get_text_length, validate=False))
            #]))
        ])),
        ('clf', MultinomialNB())
    ])
    predicted = cross_val_predict(classifier, trainData, trainLabel, cv=5)
    print predicted
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
                ('vectorizer', CountVectorizer(max_df=0.6)),
                #('tfidf', TfidfTransformer())
            ])),
            ('count', FunctionTransformer(get_text_length, validate=False))
        ])),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None))
    ])
    #transValue=classifier.fit_transform(trainData)
    #parameter=classifier.named_steps['features']._iter()#text.named_steps['vectorizer'].get_feature_names()
    #featureList=list(parameter)#[1].named_steps['vectorizer'].get_feature_names()
    predicted = cross_val_predict(classifier, trainData, trainLabel, cv=5)
    #print transValue 
    return predicted

def randomforest(trainData, trainLabel):
    classifier = Pipeline([
        ('vectorizer', CountVectorizer(max_df=0.6)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(max_depth=10, random_state=0))
    ])
    predicted = cross_val_predict(classifier, trainData, trainLabel, cv=5)
    return predicted

def results(predictTestLabel, trueTestLabel):
    accuracy=np.mean(predictTestLabel == trueTestLabel) 
    print "Accuracy:",metrics.accuracy_score(trueTestLabel, predictTestLabel), accuracy
    print "F1_micro", f1_score(trueTestLabel, predictTestLabel, average="micro")
    class_names=['Conclusion','Clinical Data','Title','Names','Findings']
    print "Test result:",metrics.classification_report(trueTestLabel, predictTestLabel)
    #conf_mat=metrics.confusion_matrix(trueTestLabel, predictTestLabel)
    #print conf_mat
    #fig=confusionmatrix_heatmap.print_confusion_matrix(conf_mat,class_names)
    #fig.savefig('confMat_3sections_SVM.pdf',bbox_inches="tight")

sentence=[]
label=[]
ct=0

f=open('Sentence180_3Sections.csv','r')
fRead=csv.reader(f)
next(fRead)
'''f1Write=csv.writer(f1)
f1Write.writerow(['Sentence','Heading'])'''

for row in fRead:
    #ct=ct+1
    sentence.append(row[0])
    label.append(row[1])
    #if ct>416:
    #    break

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
#predictTestLabelRF=randomforest(sentence, label)
#print "random forest"
#results(predictTestLabelRF, label)
