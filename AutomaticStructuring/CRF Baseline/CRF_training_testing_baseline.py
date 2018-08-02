import re
import sys
import nltk
import string
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import classification_report
from nltk.tokenize import RegexpTokenizer
#from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.utils import flatten
import CRF_features_baseline
import CRF_measures_baseline
import xml.etree.cElementTree as Elem
import xml
from xml.etree.ElementTree import Element, ElementTree

import parsingReportsXML_baseline


def token_LabelCreation(reports):
    docs=[]
    labelList=[]
    for report in reports:
    #print report
        report=report.strip('[(').strip('\n').strip(')]')
        lines=report.split('), (')
        EachReport=[]
        for line in lines:
            #line=re.sub(r'\d+[,-]*\d*[?]*\s*[mc]m',"#NUM",line)
            #print line
            line=line.strip("'").strip('"')
            #print line
            #line=re.sub(r"\\","",line)
            ar=re.split(r"', '|', \"|', u'",line)
            #print ar
            if len(ar)!=2:
                #ar=line.split("', \"")
                print line
                print ar
            #print ar
            #ar[0]=re.sub(r"'","",ar[0])
            #print ar[0]
            #ar[1]=re.sub(r"'|\"","",ar[1])
            #print ar[1]
            #print ar[0]
            label=("/").join(ar[0].split('/')[2:])
            labelList.append(label)
            #print label
            tokens=ar[1].strip()
            #print tokens
            tovPat=re.compile(r't,o,v',re.IGNORECASE)
            tokens=tovPat.sub('tov',tokens)
            #date=re.search(r'\d\d-\d\d-\d\d',tokens)
            #if date!=None:
            #    print line
            #    date_new=re.sub(r'-','\\',date.group())
            #    tokens=re.sub(r'\d\d-\d\d-\d\d',date_new,tokens)
            tokens=re.sub(r'\d',"#NUM",tokens)
            #tokens=re.sub(r'#NUM#NUM-#NUM#NUM-#NUM#NUM#NUM#NUM',"#NUM#NUM/#NUM#NUM/#NUM#NUM#NUM#NUM",tokens)
            tokens=re.split(r'([,\(\).?:-]*)\s*',tokens)
            tokens=filter(lambda a: a!='', tokens)
            #tokens=re.findall(r"[\w']+|[.,?!;]",tokens)
            #print tokens
            for i in range(len(tokens)):
                if label=='O':
                    EachReport.append((tokens[i],label))
                else:
                    if i==0:
                        tag='B-'+label
                        EachReport.append((tokens[i],tag))
                    else:
                        tag='I-'+label
                        EachReport.append((tokens[i],tag))
        docs.append(EachReport)
    #print np.unique(labelList)
    return docs

def CRF_featureCreation(docs):
    tokenList,data=CRF_features_baseline.posTagAdding(docs)
    X = [CRF_features_baseline.sent2features(doc) for doc in data]
    y = [CRF_features_baseline.sent2labels(doc) for doc in data]
    print len(X)
    print len(y)
    return X,y,tokenList

def CRF_trainer(X,y,X_Te):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    #kf=KFold(n_splits=5,shuffle=True)
    #predicted=cross_val_predict(crf, X, y, cv=kf)
    crf.fit(X,y)
    predicted=crf.predict(X_Te)
    return predicted

tree_all = Elem.parse('./../labeling/new_data.xml')
list_tree=tree_all.findall('report')
k=len(list_tree)/4
label_dic_all={}
label_dic_2={}
out=open('CRF_baseline_file.txt','a')
out1=open('CRF_baseline_featurefile.txt','a')
out2=open('CRF_baseline_predictedvsTrue.txt','a')
for i in range(0,4):
    if i==0:
        list_tree_test=list_tree[:k]
        list_tree_train=list_tree[k:]
    elif i==len(list_tree)-1:
        list_tree_test=list_tree[3*k:]
        list_tree_train=list_tree[:3*k]
    else:
        list_tree_train=list_tree[:i*k]+list_tree[(i+1)*k:]
        list_tree_test=list_tree[i*k:(i+1)*k]
        
    root=Element('radiology_reports')
    root1=Element('radiology_reports')
    for list_tree_elem in list_tree_train:
        root.append(list_tree_elem)
    for list_tree_elem in list_tree_test:
        root1.append(list_tree_elem)
    tree=ElementTree(root)
    roott=tree.getroot()
    tree1=ElementTree(root1)
    roott1=tree1.getroot()
    out3=open('file_train_'+str(i),'w')
    out4=open('file_test_'+str(i),'w')
    EachReport1=[]
    EachReport2=[]
    parsingReportsXML_baseline.print_path_of_elems(out3,EachReport1, roott, roott.tag)
    parsingReportsXML_baseline.print_path_of_elems(out4,EachReport2, roott1, roott1.tag)
    total_measure_dic={}
    out3.close()
    out4.close()
    f_train=open('file_train_'+str(i),'r')
    f_test=open('file_test_'+str(i),'r')
    reports_tr=f_train.readlines()
    reports_te=f_test.readlines()
    
    docs1_tr=token_LabelCreation(reports_tr)
    docs1_te=token_LabelCreation(reports_te)
    for i,doc in enumerate(docs1_te):
        for line in doc:
            out.write(str(i)+"\t"+line[0]+"\t"+line[1]+"\n")
    
    X_tr,Y_tr,tokenList_tr_=CRF_featureCreation(docs1_tr)
    X_te,Y_te,tokenList_te=CRF_featureCreation(docs1_te)
    for doc in X_te:
            for line in doc:
                out1.write(str(line)+"\n")
    
    predicted1=CRF_trainer(X_tr,Y_tr,X_te)
    predicted2=[]
    Y1=[]
    for data1,true1,pre1 in zip(tokenList_te,Y_te,predicted1):
        preSub=[]
        YSub=[]
        for i in range(0,len(data1)):
            preSub.append(pre1[i])
            YSub.append(true1[i])
            if data1[i] not in string.punctuation:
                out2.write(str(data1[i])+"\t"+str(true1[i])+"\t"+str(pre1[i])+"\n")
                #print data1[i],"\t",pre1[i],"\t",true1[i]
        predicted2.append(preSub)
        Y1.append(YSub)
        
    CRF_measures_baseline.tokenLevel_measures(predicted2,Y1,tokenList_te,label_dic_all)
    #partial_phrase_dic,complete_phrase_dic=CRF_measures_baseline.partialPhraseLevel_measures(tokenList,predicted1,Y)
    #print "Label\tTokenPrecision,recall,fmeasure,support\tPartialPhraseAccuracy\tCompletePhraseAccuracy"
    '''for key in token_dic.iterkeys():
        total_measure_dic[key]=[token_dic[key],partial_phrase_dic[key],complete_phrase_dic[key]]
        print key,"\t",token_dic[key],"\t", partial_phrase_dic[key],"\t",complete_phrase_dic[key]'''
label_dic2_fscore={}
label_dic2_support={}
for key in label_dic_all.iterkeys():
    label_dic2_fscore[key]=float(sum(label_dic_all[key][0]))/len(label_dic_all[key][0])
    label_dic2_support[key]=label_dic_all[key][1][0]
print label_dic2_fscore
print label_dic2_support