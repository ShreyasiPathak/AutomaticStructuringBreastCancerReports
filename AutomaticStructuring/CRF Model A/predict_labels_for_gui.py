import re
import nltk
import string
import numpy as np
import sys
import sklearn
import pickle
from nltk.tokenize import RegexpTokenizer

import xml
import xml.etree.cElementTree as Elem
from xml.etree.ElementTree import Element, ElementTree

import CRF_features_cascadedCRF
import CRF_measures_cascadedCRF
import labeling_to_xml

reload(sys)
sys.setdefaultencoding('utf8')

def dataPreprocessing(allReports):
    docs=[]
    docs_1=[]
    for report in allReports:
        txt=report.strip()
        txt1=re.sub(r'\d',"#NUM",txt)
        tovPat=re.compile(r't,o,v',re.IGNORECASE)
        txt1=tovPat.sub('tov',txt1)
        tokens=re.split(r'([,\(\).?:-]*)\s*',txt1)
        tokens=filter(lambda a: a!='', tokens)
        docs.append(tokens)
        
        txt2=tovPat.sub('tov',txt)
        tokens2=re.split(r'([,\(\).?:-]*)\s*',txt2)
        if 'tov' in tokens2:
            ind=tokens2.index('tov')
            tokens2[ind]='t,o,v'
        tokens2=filter(lambda a: a!='', tokens2)
        docs_1.append(tokens2)
    return docs,docs_1

def CRF_featureCreationTest(docs):
    tokenList,data=CRF_features_cascadedCRF.posTagAddingTest(docs)
    X = CRF_features_cascadedCRF.sent2features(data)
    return X,tokenList

def CRF_predict(crf,xTest):
    #cv=ShuffleSplit(n_split=2,test_size=0.5,random_state=0)
    #crf.predict(X)
    xTest=[xTest]
    predicted2=crf.predict(xTest)
    #print predicted2
    return predicted2

def train_test_onTrue(currentNode):
    reportNodes=tree.findall("./"+currentNode)
    allReports=[]
    for report in reportNodes:
        allReports.append(report.text)
    
    global docs_ori
    docs1,docs_ori=dataPreprocessing(allReports)
    if currentNode=='report':
        global tokenListAll
        tokenListAll=docs1        

def test_onPredicted(beg,end,phrase,preLabels,currentNode):
    #print currentNode
    phrase2=phrase[beg:end]
    #print phrase1
    X1,tokenList1=CRF_featureCreationTest(phrase2)
    #print X1
    predicted1=CRF_predict(crfDic[currentNode],X1)
    predicted1=predicted1[0]
    #print predicted1
    if type(predicted1) is not list:
        predicted1=predicted1.tolist()
        #if currentNode=='report/negative_finding':
        #    print zip(tokenList1,y1,predicted1.tolist())
    preTokenList=zip(predicted1,tokenList1)
    #print preTokenList
    j=0
    for i in range(beg,end):
        pre1=preTokenList[j][0]
        data1=preTokenList[j][1]
        pre1List=pre1.split('-')
        labelStartPres=pre1List[0]
        labelEndPres=pre1List[len(pre1List)-1]
        #print preLabels
        preLabels[i][0]=preLabels[i][0]+"/"+labelEndPres
        preLabels[i][1]=preLabels[i][1]+"/"+pre1
        if j!=len(preTokenList)-1:
            labelStartNext=preTokenList[j+1][0].split('-')[0]
        else:
            labelStartNext=None
        if labelStartPres=='B':
            beg=i
            if labelStartNext=='B' or labelStartNext=='O' or j==len(preTokenList)-1:
                end=i
                child=str(currentNode)+"/"+labelEndPres
                if crfDic.has_key(child):
                    test_onPredicted(beg,end,phrase,preLabels,child)
        elif labelStartPres=='I':
            if labelStartNext=='B' or labelStartNext=='O' or j==len(preTokenList)-1:
                end=i
                child=str(currentNode)+"/"+labelEndPres
                if crfDic.has_key(child):
                    test_onPredicted(beg,end+1,phrase,preLabels,child)
        j=j+1
            

def mainFunc(path_to_prediction_report):
    #tree_all = Elem.parse('./../labeling/testSample_predict.xml')
    #path_to_prediction report is the xml file containing reports to be labeled
    tree_all = Elem.parse(path_to_prediction_report)
    list_tree=tree_all.findall('report')
    
    root=Element('radiology_reports')
    for list_tree_elem in list_tree:
        root.append(list_tree_elem)
    print "No.of reports:",len(root)
    global tree
    tree=ElementTree(root)
    
    
    dicListTrue={}
    dicListPre={}
    global crfDic
    
    #load models
    models_path="CRFmodel1_models.pkl"
    models_file=open(models_path,'rb')
    crfDic=pickle.load(models_file)
    
    train_test_onTrue("report")
    
    dicKeyTrueCount={}
    cascadedOnPre=[]
    for i in range(len(tokenListAll)):
        cascadedOnPre.append([])
        for j in range(len(tokenListAll[i])):
            cascadedOnPre[i].append(['',''])
    
    
    for i in range(len(tokenListAll)):
        preLabels=list(cascadedOnPre[i])
        phrase1=list(tokenListAll[i])
        #print phrase1
        test_onPredicted(0,len(tokenListAll[i]),phrase1,preLabels,"report")
        for j in range(len(cascadedOnPre[i])):
            cascadedOnPre[i][j]=preLabels[j]
            #print cascadedOnPre[i]
    
    
    cascPreList=[]
    cascPreList1=[]
    
    #level2
    predictPredictList1=[]
    tokenFor2levelList1=[]
    
    #level3
    predictpredictPredictList1=[]
    tokenFor3levelList1=[]
    
    for i in range(len(tokenListAll)):
        cascPreVal=[]
        cascPreVal1=[]
        
        #level2
        predictPredict1=[]
        tokenFor2level=[]
        
        #level3
        predictpredictPredict1=[]
        tokenFor3level=[]
        
        for j in range(len(tokenListAll[i])):
            cascadedOnPre[i][j][0]=cascadedOnPre[i][j][0].strip('/')
            cascadedOnPre[i][j][1]=cascadedOnPre[i][j][1].strip('/')
            cascPre=cascadedOnPre[i][j][0].split('/')
            if  len(cascPre)>=2:
                predictPredict="/".join(cascPre[:2])
                tokenFor2level.append(tokenListAll[i][j])
                predictPredict1.append(predictPredict)
            
            if len(cascPre)==3:
                predictpredictPredict="/".join(cascPre[0:3])
                tokenFor3level.append(tokenListAll[i][j])
                predictpredictPredict1.append(predictpredictPredict)
            cascPreVal.append(cascadedOnPre[i][j][0])
            cascPreVal1.append(cascadedOnPre[i][j][1])
            
        cascPreList.append(cascPreVal)
        cascPreList1.append(cascPreVal1)
        
        predictPredictList1.append(predictPredict1)
        tokenFor2levelList1.append(tokenFor2level)
        
        predictpredictPredictList1.append(predictpredictPredict1)
        tokenFor3levelList1.append(tokenFor3level)
    
    labeling_to_xml.mainFunc(docs_ori,cascPreList1)

path='./../labeling/testSample_predict.xml'
mainFunc(path)