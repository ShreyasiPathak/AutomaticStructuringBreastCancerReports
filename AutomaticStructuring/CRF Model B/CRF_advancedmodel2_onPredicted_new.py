import xml.etree.cElementTree as Elem

import re
import nltk
import string
import sys
import itertools
from random import shuffle
import operator
import numpy as np

import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from nltk.tokenize import RegexpTokenizer
import xml
from xml.etree.ElementTree import Element, ElementTree
#from lxml import etree
#from sklearn.grid_search import RandomizedSearchCV

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from sklearn.metrics import classification_report
import CRF_features_cascadedCRF
import CRF_measures_cascadedCRF
import labeling_to_xml

reload(sys)
sys.setdefaultencoding('utf8')

def token_LabelCreation(allReports):
    docs=[]
    for report in allReports:
        #print report
        EachReport=[]
        for line in report:
            #print line
            label=line[1]
            #print label
            txt=line[0].strip()
            txt=re.sub(r'\d',"#NUM",txt)
            tovPat=re.compile(r't,o,v',re.IGNORECASE)
            txt=tovPat.sub('tov',txt)
            tokens=re.split(r'([,\(\).?:-]*)\s*',txt)
            tokens=filter(lambda a: a!='', tokens)
            #tokens=line[1].strip().split()
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
    #print docs
    return docs

def token_LabelCreation_Test(allReports):
    docs1=[]
    for report in allReports:
        #print report
        EachReport=[]
        for line in report:
            #print line
            label=line[1]
            #print label
            txt=line[0].strip()
            tovPat=re.compile(r't,o,v',re.IGNORECASE)
            txt=tovPat.sub('tov',txt)
            tokens=re.split(r'([,\(\).?:-]*)\s*',txt)
            tokens=filter(lambda a: a!='', tokens)
            #tokens=line[1].strip().split()
            #print tokens
            for i in range(len(tokens)):
                if label=='O':
                    EachReport.append((tokens[i]))
                else:
                    if i==0:
                        tag='B-'+label
                        EachReport.append((tokens[i]))
                    else:
                        tag='I-'+label
                        EachReport.append((tokens[i]))
        docs1.append(EachReport)
    #print docs
    return docs1

def CRF_featureCreation(docs):
    tokenList,data=CRF_features_cascadedCRF.posTagAdding(docs)
    X = [CRF_features_cascadedCRF.sent2features(doc) for doc in data]
    y = [CRF_features_cascadedCRF.sent2labels(doc) for doc in data]
    #print len(X)
    #print len(y)
    return X,y,tokenList

def CRF_featureCreationTest(docs):
    tokenList,data=CRF_features_cascadedCRF.posTagAddingTest(docs)
    #print data
    X = CRF_features_cascadedCRF.sent2features(data)
    #print len(X)
    #print len(y)
    return X,tokenList

def CRF_trainer(xTrain,yTrain,xTest):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    #cv=ShuffleSplit(n_split=2,test_size=0.5,random_state=0)
    crf.fit(xTrain,yTrain)
    predicted=crf.predict(xTest)
    pred_prob=crf.predict_marginals(xTest)
    label_prob_all=[]
    for pred_prob1 in pred_prob:
        label_prob=[]
        for dic_label in pred_prob1:
            #print dic_label
            label_prob.append(list(max(dic_label.iteritems(), key=operator.itemgetter(1))))
        label_prob_all.append(label_prob)
    return predicted,crf,label_prob_all

def CRF_predict(crf,xTest):
    #cv=ShuffleSplit(n_split=2,test_size=0.5,random_state=0)
    #crf.predict(X)
    #xTest=[xTest]
    label_prob=[]
    predicted2=crf.predict_single(xTest)
    pred_prob=crf.predict_marginals_single(xTest)
    for dic_label in pred_prob:
        #print dic_label
        label_prob.append(list(max(dic_label.iteritems(), key=operator.itemgetter(1))))
    #print label_prob
    #print predicted2
    return predicted2,label_prob

def textExtraction(currentNode,tree_used):
    #reportNodes=tree.findall("./report/negative_finding")
    allReports=[]
    reportNodes=tree_used.findall("./"+currentNode)
    for report in reportNodes:
        #print report.tag
        oneReport=[]
        if re.search(r'\S',report.text):
            reprtText=re.sub(r'\t|\n','',report.text)
            #print reprtText
            oneReport.append((reprtText.strip(),'O'))
        for node in report:
            #print node.tag
            nodeText=[]
            #print node
            #if node.tag not in nodeAr:
            #    nodeAr.append(node.tag)
            for text in node.itertext():
                text=re.sub(r'\t|\n','',text)
                nodeText.append(text.strip())
                #nodeText.append(text.strip().strip('\t').strip('\n').strip())
            if report.tag=='report':
                oneReport.append((" ".join(nodeText),node.tag))
            if report.tag=='positive_finding':
                oneReport.append((" ".join(nodeText),node.tag))
            if report.tag=='negative_finding' and node.tag!='location':
                oneReport.append((" ".join(nodeText),node.tag))
            elif report.tag=='negative_finding' and node.tag=='location':
                oneReport.append((" ".join(nodeText),'O'))
            if re.search(r'\S',node.tail):
                nodeTail=re.sub(r'\t|\n','',node.tail)
                #print nodeTail
                oneReport.append((nodeTail.strip(),'O'))
        allReports.append(oneReport)
    return allReports

def CRF4(allReport,EachReport,elem):
    #elem=tree.findall("./"+currentNode)
    for child in elem:
        if child.tag!="breast_composition" and child.tag!="O":
            #print child.tag
            #print EachReport
            if not child.getchildren() and child.text:
                if child.tag=="location":
                    # leaf node with text => print
                    #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                    text=re.sub(r'\t|\n','',child.text)
                    EachReport.append((text,str(child.tag)))
                else:
                    text=re.sub(r'\t|\n','',child.text)
                    EachReport.append((text,'O'))
                if re.search(r'\S',child.tail):
                    #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                    text1=re.sub(r'\t|\n','',child.tail)
                    EachReport.append((text1,'O'))
            else:
                if re.search(r'\S',child.text):
                    #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                    text=re.sub(r'\t|\n','',child.text)
                    EachReport.append((text,'O'))
                # node with child elements => recurse
                #print child.tag
                CRF4(allReport,EachReport, child)
                #print child.tag
                #print EachReport
                if re.search(r'\S',child.tail):
                    #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                    text1=re.sub(r'\t|\n','',child.tail)
                    EachReport.append((text1,'O'))
            if child.tag=="positive_finding" or child.tag=="negative_finding":
                #print EachReport
                if EachReport!=[]:
                    allReport.append(EachReport)
                    EachReport=[]
                #print EachReport
            #print EachReport
    #return allReports

def CRF5(allReport,EachReport,elem):
    for child in elem:
        if not child.getchildren() and child.text:
            if child.tag=="margin":
                # leaf node with text => print
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,str(child.tag)))
            else:
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        else:
            if re.search(r'\S',child.text):
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            # node with child elements => recurse
            CRF5(allReport,EachReport, child)
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        if child.tag=="mass":
            if EachReport!=[]:
                allReport.append(EachReport)
                EachReport=[]

def CRF6(allReport,EachReport,elem):
    for child in elem:
        if not child.getchildren() and child.text:
            if child.tag=="size":
                # leaf node with text => print
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,str(child.tag)))
            else:
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        else:
            if re.search(r'\S',child.text):
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            # node with child elements => recurse
            CRF6(allReport,EachReport, child)
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        if child.tag=="mass" or child.tag=="calcification" or child.tag=="asymmetry":
            if EachReport!=[]:
                allReport.append(EachReport)
                EachReport=[]

def CRF7(allReport,EachReport,elem):
    for child in elem:
        if not child.getchildren() and child.text:
            if child.tag=="density" or child.tag=="shape":
                # leaf node with text => print
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,str(child.tag)))
            else:
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        else:
            if re.search(r'\S',child.text):
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            # node with child elements => recurse
            CRF7(allReport,EachReport, child)
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        if child.tag=="mass":
            if EachReport!=[]:
                allReport.append(EachReport)
                EachReport=[]

def CRF8(allReport,EachReport,elem):
    for child in elem:
        if not child.getchildren() and child.text:
            if child.tag=="morphology" or child.tag=="distribution":
                # leaf node with text => print
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,str(child.tag)))
            else:
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        else:
            if re.search(r'\S',child.text):
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            # node with child elements => recurse
            CRF8(allReport,EachReport, child)
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        if child.tag=="calcification":
            if EachReport!=[]:
                allReport.append(EachReport)
                EachReport=[]

def CRF9(allReport,EachReport,elem):
    for child in elem:
        if not child.getchildren() and child.text:
            if child.tag=="associated_features":
                # leaf node with text => print
                #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,str(child.tag)))
            else:
                text=re.sub(r'\t|\n','',child.text)
                EachReport.append((text,'O'))
            if re.search(r'\S',child.tail):
                #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                text1=re.sub(r'\t|\n','',child.tail)
                EachReport.append((text1,'O'))
        else:
            if child.tag=="associated_features":
                    #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                    for text in child.itertext():
                        text1=re.sub(r'\t|\n','',text)
                        EachReport.append((text1,child.tag))
            else:
                if re.search(r'\S',child.text):
                    #print "%s/%s, %s" % (elem_path, child.tag, child.text.strip('\t').strip().strip('\n'))
                    text=re.sub(r'\t|\n','',child.text)
                    EachReport.append((text,'O'))
                # node with child elements => recurse
                CRF9(allReport,EachReport,child)
                if re.search(r'\S',child.tail):
                    #print "%s, %s" % (elem_path, child.tail.strip('\t').strip().strip('\n'))
                    text1=re.sub(r'\t|\n','',child.tail)
                    EachReport.append((text1,'O'))
        if child.tag=="mass" or child.tag=="calcification" or child.tag=="asymmetry" or child.tag=="architectural_distortion":
            if EachReport!=[]:
                allReport.append(EachReport)
                EachReport=[]

def mergingResults(i,m,n,labelStartTrue,phrase,dicKeyTrueCount1):
    currentNodeInstanceTrue=dicListTrue[labelStartTrue][i]
    #print "label start true:",labelStartTrue,i
    #print currentNodeInstanceTrue
    currentNodeInstancePre=dicListPre[labelStartTrue][i]
    k=0
    for j in range(m,n):
        labelTrue=currentNodeInstanceTrue[k].split('-')
        labelStartTagPres=labelTrue[0]
        dicKeyTrue=labelStartTrue+'/'+labelTrue[len(labelTrue)-1]
        #print "dicKeyTrue:",dicKeyTrue
        #print currentNodeInstancePre
        if type(currentNodeInstancePre[k]) is list:
            pre1List=currentNodeInstancePre[k][0]
            pre1List=pre1List.split('-')
            phrase[j][1]=phrase[j][1]+"/"+pre1List[len(pre1List)-1]+":"+str(currentNodeInstancePre[k][1])
            
        else:
            pre1List=currentNodeInstancePre[k].split('-')
            phrase[j][1]=phrase[j][1]+"/"+pre1List[len(pre1List)-1]
        #print phrase[j]
        phrase[j][0]=phrase[j][0]+"/"+labelTrue[len(labelTrue)-1]
        #print j
        #print phrase[j]
        if k!=len(currentNodeInstanceTrue)-1:
            labelStartTagNext=currentNodeInstanceTrue[k+1].split('-')[0]
        else:
            labelStartTagNext=None
        if labelStartTagPres=='B':
            beg=j
            if labelStartTagNext=='B' or labelStartTagNext=='O' or k==len(currentNodeInstanceTrue)-1:
                end=j
                if dicKeyTrueCount1.has_key(dicKeyTrue):
                    dicKeyTrueCount1[dicKeyTrue]=dicKeyTrueCount1.get(dicKeyTrue)+1
                else:
                    dicKeyTrueCount1[dicKeyTrue]=1
                if dicListTrue.has_key(dicKeyTrue):
                    #print dicKeyTrue,beg,end
                    mergingResults(dicKeyTrueCount1[dicKeyTrue]-1,beg,end+1,dicKeyTrue,phrase,dicKeyTrueCount1)
                #else:
                #    TokenTruePre.append([tokenListAll[i][j],dicKeyTrue,dicKeyPre])
                
        elif labelStartTagPres=='I':
            if labelStartTagNext=='B' or labelStartTagNext=='O' or k==len(currentNodeInstanceTrue)-1:
                end=j
                if dicKeyTrueCount1.has_key(dicKeyTrue):
                    dicKeyTrueCount1[dicKeyTrue]=dicKeyTrueCount1.get(dicKeyTrue)+1
                else:
                    dicKeyTrueCount1[dicKeyTrue]=1
                if dicListTrue.has_key(dicKeyTrue):
                    #print dicKeyTrue, beg,end
                    mergingResults(dicKeyTrueCount1[dicKeyTrue]-1,beg,end+1,dicKeyTrue,phrase,dicKeyTrueCount1)
                #else:
                #    TokenTruePre.append([tokenListAll[i][j],dicKeyTrue,dicKeyPre])
        k=k+1            

def max_label_cal(prob_All1,preLabels1,beg1,end1,preLabelLoc):
    z=0
    #print preLabels1
    locLab=preLabelLoc[beg1:end1+1]
    #print prob_All1
    #prob_All2=zip(*(zip(*prob_All1)+[locLab]))
    #print prob_All2
    for i1 in range(beg1,end1+1):
        List_NE=filter(lambda a:a[0]!='O',prob_All1[z])
        if List_NE!=():
            #print List_NE
            label_now1=max(List_NE, key=operator.itemgetter(1))
            #print label_now1
            if locLab[z][0]!='O':
                if label_now1[1]>locLab[z][1]:
                    label_now=label_now1[0]
                else:
                    label_now=locLab[z][0]
            else:
                label_now=label_now1[0]
            #print label_now
            preLabels1[i1][0]=preLabels1[i1][0]+"/"+label_now.split('-')[1]
            preLabels1[i1][1]=preLabels1[i1][1]+"/"+label_now
        else:
            preLabels1[i1][0]=preLabels1[i1][0]+"/"+locLab[z][0].split('-')[len(locLab[z][0].split('-'))-1]
            preLabels1[i1][1]=preLabels1[i1][1]+"/"+locLab[z][0]
        z=z+1
    return preLabels1

def test_onPredicted(beg,end,phrase,preLabels,crfDicKey,currentNode,preLabelLoc):
    #print currentNode
    phrase2=phrase[beg:end]
    #print phrase1
    X1,tokenList1=CRF_featureCreationTest(phrase2)
    #print X1
    predicted1,prob=CRF_predict(crfDic[crfDicKey],X1)
    #print predicted1
    #predicted1=predicted1[0]
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
        #if labelEndPres!='O' or preLabels[i][0]=='':
        preLabels[i][0]=preLabels[i][0]+"/"+labelEndPres
        preLabels[i][1]=preLabels[i][1]+"/"+pre1
        if j!=len(preTokenList)-1:
            labelStartNext=preTokenList[j+1][0].split('-')[0]
        else:
            labelStartNext=None
        if preLabels[i][0]=="/negative_finding/O":
            preLabels[i][0]='/'.join(preLabels[i][0].split('/')[:len(preLabels[i][0].split('/'))-1])+"/"+preLabelLoc[i][0].split('-')[len(preLabelLoc[i][0].split('-'))-1]
            #print phrase[i]
            #print preLabelLoc[i][0]
            preLabels[i][1]='/'.join(preLabels[i][1].split('/')[:len(preLabels[i][1].split('/'))-1])+"/"+preLabelLoc[i][0]
        if labelStartPres=='B':
            beg=i
            if labelStartNext=='B' or labelStartNext=='O' or j==len(preTokenList)-1:
                end=i
                child=str(currentNode)+"/"+labelEndPres
                #print child
                if child=="report/positive_finding":
                    phrase3=phrase[beg:end+1]
                    X1_ch3,tokenList_ch3=CRF_featureCreationTest(phrase3)
                    predicted1_ch3_loc,prob_ch3_loc=CRF_predict(crfDic['pfnf'],X1_ch3)
                    z1=0
                    for i1 in range(beg,end+1):
                        preLabelLoc[i1]=prob_ch3_loc[z1]
                        z1=z1+1
                    test_onPredicted(beg,end+1,phrase,preLabels,'positive_finding',child,preLabelLoc)
                    #print "pf:",preLabels
                if child=="report/negative_finding":
                    phrase10=phrase[beg:end+1]
                    X1_ch10,tokenList_ch10=CRF_featureCreationTest(phrase10)
                    predicted1_ch10_loc,prob_ch10_loc=CRF_predict(crfDic['pfnf'],X1_ch10)
                    z2=0
                    for i1 in range(beg,end+1):
                        preLabelLoc[i1]=prob_ch10_loc[z2]
                        z2=z2+1
                    test_onPredicted(beg,end+1,phrase,preLabels,'negative_finding',child,preLabelLoc)
                    #print "nf:",preLabels
                    
                if child=="report/positive_finding/mass":
                    phrase5=phrase[beg:end+1]
                    X1_ch5,tokenList_ch5=CRF_featureCreationTest(phrase5)
                    predicted1_ch5_ShDen,prob_ch5_ShDen=CRF_predict(crfDic['pf/mass'],X1_ch5)
                    predicted1_ch5_Si,prob_ch5_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch5)
                    predicted1_ch5_Mar,prob_ch5_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch5)
                    predicted1_ch5_asso,prob_ch5_asso=CRF_predict(crfDic['pf/asso'],X1_ch5)
                    #predicted1_ch5_loc,prob_ch5_loc=CRF_predict(crfDic['pfnf'],X1_ch5)
                    prob_All=zip(prob_ch5_ShDen,prob_ch5_Si,prob_ch5_Mar,prob_ch5_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/calcification":
                    phrase6=phrase[beg:end+1]
                    X1_ch6,tokenList_ch6=CRF_featureCreationTest(phrase6)
                    predicted1_ch6_Si,prob_ch6_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch6)
                    predicted1_ch6_MoDi,prob_ch6_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch6)
                    predicted1_ch6_asso,prob_ch6_asso=CRF_predict(crfDic['pf/asso'],X1_ch6)
                    #predicted1_ch6_loc,prob_ch6_loc=CRF_predict(crfDic['pfnf'],X1_ch6)
                    prob_All_6=zip(prob_ch6_Si,prob_ch6_MoDi,prob_ch6_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_6,preLabels,beg,end,preLabelLoc)
                
                if child=="report/positive_finding/architectural_distortion":
                    phrase7=phrase[beg:end+1]
                    X1_ch7,tokenList_ch7=CRF_featureCreationTest(phrase7)
                    predicted1_ch7_asso,prob_ch7_asso=CRF_predict(crfDic['pf/asso'],X1_ch7)
                    #predicted1_ch7_loc,prob_ch7_loc=CRF_predict(crfDic['pfnf'],X1_ch7)
                    prob_All_7=zip(prob_ch7_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_7,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/asymmetry":
                    phrase8=phrase[beg:end+1]
                    X1_ch8,tokenList_ch8=CRF_featureCreationTest(phrase8)
                    predicted1_ch8_asso,prob_ch8_asso=CRF_predict(crfDic['pf/asso'],X1_ch8)
                    #predicted1_ch8_loc,prob_ch8_loc=CRF_predict(crfDic['pfnf'],X1_ch8)
                    predicted1_ch8_Si,prob_ch8_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch8)
                    prob_All_8=zip(prob_ch8_asso,prob_ch8_Si,preLabelLoc[beg:end+1])
                    print prob_All_8
                    preLabels=max_label_cal(prob_All_8,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/mass":
                    phrase9=phrase[beg:end+1]
                    X1_ch9,tokenList_ch9=CRF_featureCreationTest(phrase9)
                    predicted1_ch9_Mar,prob_ch9_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch9)
                    #predicted1_ch9_loc,prob_ch9_loc=CRF_predict(crfDic['pfnf'],X1_ch9)
                    prob_All_9=zip(prob_ch9_Mar,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_9,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/calcification":
                    phrase4=phrase[beg:end+1]
                    X1_ch4,tokenList_ch4=CRF_featureCreationTest(phrase4)
                    predicted1_ch4_MoDi,prob_ch4_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch4)
                    #predicted1_ch4_loc,prob_ch4_loc=CRF_predict(crfDic['pfnf'],X1_ch4)
                    prob_All_4=zip(prob_ch4_MoDi,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_4,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/O":
                    for i1 in range(beg,end+1):
                        preLabels[i1][0]='/'.join(preLabels[i1][0].split('/')[:len(preLabels[i1][0].split('/'))-1])+"/"+preLabelLoc[i1][0].split('-')[len(preLabelLoc[i1][0].split('-'))-1]
                        print phrase[i1]
                        print preLabelLoc[i1][0]
                        preLabels[i1][1]='/'.join(preLabels[i1][0].split('/')[:len(preLabels[i1][0].split('/'))-1])+preLabels[i1][1]+"/"+preLabelLoc[i1][0]
                    
                if child=="report/negative_finding/associated_features" or child=="report/negative_finding/architectural_distortion" \
                         or child=="report/negative_finding/asymmetry" or child=="report/positive_finding/associated_features":
                    for i1 in range(beg,end+1):
                        preLabels[i1][0]=preLabels[i1][0]+"/"+preLabelLoc[i1][0].split('-')[len(preLabelLoc[i1][0].split('-'))-1]
                        preLabels[i1][1]=preLabels[i1][1]+"/"+preLabelLoc[i1][0]
                    
        elif labelStartPres=='I':
            if labelStartNext=='B' or labelStartNext=='O' or j==len(preTokenList)-1:
                end=i
                child=str(currentNode)+"/"+labelEndPres
                if child=="report/positive_finding":
                    phrase3=phrase[beg:end+1]
                    X1_ch3,tokenList_ch3=CRF_featureCreationTest(phrase3)
                    predicted1_ch3_loc,prob_ch3_loc=CRF_predict(crfDic['pfnf'],X1_ch3)
                    z1=0
                    for i1 in range(beg,end+1):
                        preLabelLoc[i1]=prob_ch3_loc[z1]
                        z1=z1+1
                    test_onPredicted(beg,end+1,phrase,preLabels,'positive_finding',child,preLabelLoc)
                    #print "pf:",preLabels
                if child=="report/negative_finding":
                    phrase10=phrase[beg:end+1]
                    X1_ch10,tokenList_ch10=CRF_featureCreationTest(phrase10)
                    predicted1_ch10_loc,prob_ch10_loc=CRF_predict(crfDic['pfnf'],X1_ch10)
                    z2=0
                    for i1 in range(beg,end+1):
                        preLabelLoc[i1]=prob_ch10_loc[z2]
                        z2=z2+1
                    test_onPredicted(beg,end+1,phrase,preLabels,'negative_finding',child,preLabelLoc)
                    #print "nf:",preLabels
                    
                if child=="report/positive_finding/mass":
                    phrase5=phrase[beg:end+1]
                    X1_ch5,tokenList_ch5=CRF_featureCreationTest(phrase5)
                    predicted1_ch5_ShDen,prob_ch5_ShDen=CRF_predict(crfDic['pf/mass'],X1_ch5)
                    predicted1_ch5_Si,prob_ch5_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch5)
                    predicted1_ch5_Mar,prob_ch5_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch5)
                    predicted1_ch5_asso,prob_ch5_asso=CRF_predict(crfDic['pf/asso'],X1_ch5)
                    #predicted1_ch5_loc,prob_ch5_loc=CRF_predict(crfDic['pfnf'],X1_ch5)
                    prob_All=zip(prob_ch5_ShDen,prob_ch5_Si,prob_ch5_Mar,prob_ch5_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/calcification":
                    phrase6=phrase[beg:end+1]
                    X1_ch6,tokenList_ch6=CRF_featureCreationTest(phrase6)
                    predicted1_ch6_Si,prob_ch6_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch6)
                    predicted1_ch6_MoDi,prob_ch6_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch6)
                    predicted1_ch6_asso,prob_ch6_asso=CRF_predict(crfDic['pf/asso'],X1_ch6)
                    #predicted1_ch6_loc,prob_ch6_loc=CRF_predict(crfDic['pfnf'],X1_ch6)
                    prob_All_6=zip(prob_ch6_Si,prob_ch6_MoDi,prob_ch6_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_6,preLabels,beg,end,preLabelLoc)
                
                if child=="report/positive_finding/architectural_distortion":
                    phrase7=phrase[beg:end+1]
                    X1_ch7,tokenList_ch7=CRF_featureCreationTest(phrase7)
                    predicted1_ch7_asso,prob_ch7_asso=CRF_predict(crfDic['pf/asso'],X1_ch7)
                    #predicted1_ch7_loc,prob_ch7_loc=CRF_predict(crfDic['pfnf'],X1_ch7)
                    prob_All_7=zip(prob_ch7_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_7,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/asymmetry":
                    phrase8=phrase[beg:end+1]
                    X1_ch8,tokenList_ch8=CRF_featureCreationTest(phrase8)
                    predicted1_ch8_asso,prob_ch8_asso=CRF_predict(crfDic['pf/asso'],X1_ch8)
                    #predicted1_ch8_loc,prob_ch8_loc=CRF_predict(crfDic['pfnf'],X1_ch8)
                    predicted1_ch8_Si,prob_ch8_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch8)
                    prob_All_8=zip(prob_ch8_asso,prob_ch8_Si,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_8,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/mass":
                    phrase9=phrase[beg:end+1]
                    X1_ch9,tokenList_ch9=CRF_featureCreationTest(phrase9)
                    predicted1_ch9_Mar,prob_ch9_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch9)
                    #predicted1_ch9_loc,prob_ch9_loc=CRF_predict(crfDic['pfnf'],X1_ch9)
                    prob_All_9=zip(prob_ch9_Mar,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_9,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/calcification":
                    phrase4=phrase[beg:end+1]
                    X1_ch4,tokenList_ch4=CRF_featureCreationTest(phrase4)
                    predicted1_ch4_MoDi,prob_ch4_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch4)
                    #predicted1_ch4_loc,prob_ch4_loc=CRF_predict(crfDic['pfnf'],X1_ch4)
                    prob_All_4=zip(prob_ch4_MoDi,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_4,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/O":
                    for i1 in range(beg,end+1):
                        preLabels[i1][0]='/'.join(preLabels[i1][0].split('/')[:len(preLabels[i1][0].split('/'))-1])+"/"+preLabelLoc[i1][0].split('-')[len(preLabelLoc[i1][0].split('-'))-1]
                        print phrase[i1]
                        print preLabelLoc[i1][0]
                        preLabels[i1][1]='/'.join(preLabels[i1][0].split('/')[:len(preLabels[i1][0].split('/'))-1])+preLabels[i1][1]+"/"+preLabelLoc[i1][0]
                    
                if child=="report/negative_finding/associated_features" or child=="report/negative_finding/architectural_distortion" \
                         or child=="report/negative_finding/asymmetry" or child=="report/positive_finding/associated_features":
                    for i1 in range(beg,end+1):
                        preLabels[i1][0]=preLabels[i1][0]+"/"+preLabelLoc[i1][0].split('-')[len(preLabelLoc[i1][0].split('-'))-1]
                        preLabels[i1][1]=preLabels[i1][1]+"/"+preLabelLoc[i1][0]
                    
        j=j+1
        
#tree = Elem.parse('./../labeling/Shreyasi_80reports_labeled.xml')
#tree = Elem.parse('./../labeling/train_shuffled_70_30.xml')
#tree1 = Elem.parse('./../labeling/test_shuffled_70_30.xml')
'''split=int(0.70*len(list_tree))
list_tree_train=list_tree[:split]
list_tree_test=list_tree[split:]
root=Element('radiology_reports')
root1=Element('radiology_reports')
for list_tree_elem in list_tree_train:
    root.append(list_tree_elem)
for list_tree_elem in list_tree_test:
    root1.append(list_tree_elem)
tree=ElementTree(root)#.write('train_shuffled.xml',encoding='utf-8',xml_declaration=True)
tree1=ElementTree(root1)
tree.write('./../labeling/train_shuffled.xml')
tree1.write('./../labeling/test_shuffled.xml')'''

tree_all = Elem.parse('./../labeling/new_data.xml')
list_tree=tree_all.findall('report')
'''shuffle(list_tree)
root2=Element('radiology_reports')
for list_tree_elem in list_tree:
    root2.append(list_tree_elem)
tree2=ElementTree(root2)
tree2.write('new_data.xml')'''
k=len(list_tree)/4
label_dic_all={}
label_dic_2={}
label_dic_2_true={}
label_dic_2_pre={}
label_dic_3_true={}
label_dic_3_pre={}
label_dic_C1={}
conf_mat_agg=np.zeros((34,34))
out3=open('calcDist_Other_model2.txt','a')
out2=open('location_Other_model2.txt','a')
out=open('CRF_advancedmodel2.txt','a')
for cvf in range(0,4):
    if cvf==0:
        list_tree_test=list_tree[:k]
        list_tree_train=list_tree[k:]
    elif cvf==len(list_tree)-1:
        list_tree_test=list_tree[3*k:]
        list_tree_train=list_tree[:3*k]
    else:
        list_tree_train=list_tree[:cvf*k]+list_tree[(cvf+1)*k:]
        list_tree_test=list_tree[cvf*k:(cvf+1)*k]
        
    root=Element('radiology_reports')
    root1=Element('radiology_reports')
    for list_tree_elem in list_tree_train:
        root.append(list_tree_elem)
    for list_tree_elem in list_tree_test:
        root1.append(list_tree_elem)
    tree=ElementTree(root)
    tree1=ElementTree(root1)
    tree.write('./../labeling/train_shuffled'+str(cvf)+'.xml')
    tree1.write('./../labeling/test_shuffled'+str(cvf)+'.xml')

    #root=tree.getroot()
    dicListPre={}
    dicListTrue={}
    crfDic={}
    
    #CRF1=>breast composition, positive_finding, negative_finding and Other classifier, input: all reports
    allReports1=textExtraction("report",tree)
    allReports1_Te=textExtraction("report",tree1)
    #print allReports1
    docs1=token_LabelCreation(allReports1)
    docs1_Te=token_LabelCreation(allReports1_Te)
    docs1_Te_ori=token_LabelCreation_Test(allReports1_Te)
    X1,y1,tokenList1=CRF_featureCreation(docs1)
    X1_Te,y1_Te,tokenList1_Te=CRF_featureCreation(docs1_Te)
    predicted1_Te,crf1,prob_all_1=CRF_trainer(X1,y1,X1_Te)
    crfDic['report']=crf1
    if type(predicted1_Te) is not list:
        predicted1_Te=predicted1_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted1_Te,y1_Te,tokenList1_Te,label_dic_C1)
    dicListPre['report']=predicted1_Te
    dicListTrue['report']=y1_Te
    
    #CRF2=>mass,calcification,asymmetry,architectural distortion, input: all positive finding
    allReports2=textExtraction("report/positive_finding",tree)
    allReports2_Te=textExtraction("report/positive_finding",tree1)
    docs2=token_LabelCreation(allReports2)
    docs2_Te=token_LabelCreation(allReports2_Te)
    X2,y2,tokenList2=CRF_featureCreation(docs2)
    X2_Te,y2_Te,tokenList2_Te=CRF_featureCreation(docs2_Te)
    predicted2_Te,crf1,prob_all_2=CRF_trainer(X2,y2,X2_Te)
    crfDic['positive_finding']=crf1
    if type(predicted2_Te) is not list:
        predicted2_Te=predicted2_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted2_Te,y2_Te,tokenList2_Te,label_dic_C1)
    dicListPre['report/positive_finding']=predicted2_Te
    dicListTrue['report/positive_finding']=y2_Te
    
    #CRF3=>mass,calcification,asymmetry,architectural distortion, associated features, input: all negative finding
    allReports3=textExtraction("report/negative_finding",tree)
    allReports3_Te=textExtraction("report/negative_finding",tree1)
    docs3=token_LabelCreation(allReports3)
    docs3_Te=token_LabelCreation(allReports3_Te)
    X3,y3,tokenList3=CRF_featureCreation(docs3)
    X3_Te,y3_Te,tokenList3_Te=CRF_featureCreation(docs3_Te)
    predicted3_Te,crf1,prob_all_3=CRF_trainer(X3,y3,X3_Te)
    crfDic['negative_finding']=crf1
    if type(predicted3_Te) is not list:
        predicted3_Te=predicted3_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted3_Te,y3_Te,tokenList3_Te,label_dic_C1)
    dicListPre['report/negative_finding']=predicted3_Te
    dicListTrue['report/negative_finding']=y3_Te
    #print zip(tokenList3,predicted3)
    
    '''allReports1=textExtraction("report/positive_finding/mass")
    docs1=token_LabelCreation(allReports1)
    X1,y1,tokenList1=CRF_featureCreation(docs1)
    predicted1=CRF_trainer(X1,y1)
    CRF_measures_cascadedCRF.tokenLevel_measures(predicted1,y1)
    
    allReports1=textExtraction("report/positive_finding/calcification")
    docs1=token_LabelCreation(allReports1)
    X1,y1,tokenList1=CRF_featureCreation(docs1)
    predicted1=CRF_trainer(X1,y1)
    CRF_measures_cascadedCRF.tokenLevel_measures(predicted1,y1)'''
    
    
    '''for i,doc in enumerate(docs):
        for line in doc:
            out.write(str(i)+"\t"+line[0]+"\t"+line[1]+"\n")'''
    
    #CRF4=>location and other classifier, input: all positive finding and negative finding
    print "CRF4"
    allReports4=[]
    EachReport1=[]
    allReports4_Te=[]
    EachReport1_Te=[]
    pfNodes=tree.findall("./report/positive_finding")
    pfNodes_Te=tree1.findall("./report/positive_finding")
    CRF4(allReports4,EachReport1,pfNodes)
    CRF4(allReports4_Te,EachReport1_Te,pfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    nfNodes=tree.findall("./report/negative_finding")
    nfNodes_Te=tree1.findall("./report/negative_finding")
    CRF4(allReports4,EachReport1,nfNodes)
    CRF4(allReports4_Te,EachReport1_Te,nfNodes_Te)
    #print allReports4
    docs4=token_LabelCreation(allReports4)
    docs4_Te=token_LabelCreation(allReports4_Te)
    X4,y4,tokenList4=CRF_featureCreation(docs4)
    X4_Te,y4_Te,tokenList4_Te=CRF_featureCreation(docs4_Te)
    predicted4_Te,crf1,prob_all_4=CRF_trainer(X4,y4,X4_Te)
    crfDic['pfnf']=crf1
    if type(predicted4_Te) is not list:
        predicted4_Te=predicted4_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted4_Te,y4_Te,tokenList4_Te,label_dic_C1)
    #out2=open('location_Other_model2.txt','a')
    for data1,true1,pre1 in zip(tokenList4_Te,y4_Te,predicted4_Te):
        for i in range(0,len(data1)):
            if data1[i] not in string.punctuation:
                out2.write(str(data1[i])+"\t"+str(true1[i])+"\t"+str(pre1[i])+"\n")
    
    #CRF5=>margin and other classifier, input: all mass
    print "CRF5"
    allReports5=[]
    EachReport1=[]
    allReports5_Te=[]
    EachReport1_Te=[]
    massPfNodes=tree.findall("./report/positive_finding/mass")
    massPfNodes_Te=tree1.findall("./report/positive_finding/mass")
    CRF5(allReports5,EachReport1,massPfNodes)
    CRF5(allReports5_Te,EachReport1_Te,massPfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    massNfNodes=tree.findall("./report/negative_finding/mass")
    massNfNodes_Te=tree1.findall("./report/negative_finding/mass")
    CRF5(allReports5,EachReport1,massNfNodes)
    CRF5(allReports5_Te,EachReport1_Te,massNfNodes_Te)
    docs5=token_LabelCreation(allReports5)
    docs5_Te=token_LabelCreation(allReports5_Te)
    X5,y5,tokenList5=CRF_featureCreation(docs5)
    X5_Te,y5_Te,tokenList5_Te=CRF_featureCreation(docs5_Te)
    predicted5_Te,crf1,prob_all_5=CRF_trainer(X5,y5,X5_Te)
    crfDic['pfnf/mass']=crf1
    if type(predicted5_Te) is not list:
        predicted5_Te=predicted5_Te.tolist()
    #print predicted5
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted5_Te,y5_Te,tokenList5_Te,label_dic_C1)
    
    #CRF6=>size and other classifier, input: all positive_finding -> mass, calcification & asymmetry
    print "CRF6"
    allReports6=[]
    EachReport1=[]
    allReports6_Te=[]
    EachReport1_Te=[]
    massPfNodes=tree.findall("./report/positive_finding/mass")
    massPfNodes_Te=tree1.findall("./report/positive_finding/mass")
    CRF6(allReports6,EachReport1,massPfNodes)
    CRF6(allReports6_Te,EachReport1_Te,massPfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    calcPfNodes=tree.findall("./report/positive_finding/calcification")
    calcPfNodes_Te=tree1.findall("./report/positive_finding/calcification")
    CRF6(allReports6,EachReport1,calcPfNodes)
    CRF6(allReports6_Te,EachReport1_Te,calcPfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    asymPfNodes=tree.findall("./report/positive_finding/asymmetry")
    asymPfNodes_Te=tree1.findall("./report/positive_finding/asymmetry")
    CRF6(allReports6,EachReport1,asymPfNodes)
    CRF6(allReports6_Te,EachReport1_Te,asymPfNodes_Te)
    docs6=token_LabelCreation(allReports6)
    docs6_Te=token_LabelCreation(allReports6_Te)
    X6,y6,tokenList6=CRF_featureCreation(docs6)
    X6_Te,y6_Te,tokenList6_Te=CRF_featureCreation(docs6_Te)
    predicted6_Te,crf1,prob_all_6=CRF_trainer(X6,y6,X6_Te)
    crfDic['pf/mass_calc_asym']=crf1
    if type(predicted6_Te) is not list:
        predicted6_Te=predicted6_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted6_Te,y6_Te,tokenList6_Te,label_dic_C1)
    #out3=open('size_Other_model2.txt','w')'''
    '''for data1,true1,pre1 in zip(tokenList3,y3,predicted3):
        for i in range(0,len(data1)):
            if data1[i] not in string.punctuation:
                out3.write(str(data1[i])+"\t"+str(true1[i])+"\t"+str(pre1[i])+"\n")'''
    
    #CRF7=>density, shape and other classifier, input: all positive_finding -> mass
    print "CRF7"
    allReports7=[]
    EachReport1=[]
    allReports7_Te=[]
    EachReport1_Te=[]
    massPfNodes=tree.findall("./report/positive_finding/mass")
    massPfNodes_Te=tree1.findall("./report/positive_finding/mass")
    CRF7(allReports7,EachReport1,massPfNodes)
    CRF7(allReports7_Te,EachReport1_Te,massPfNodes_Te)
    docs7=token_LabelCreation(allReports7)
    docs7_Te=token_LabelCreation(allReports7_Te)
    X7,y7,tokenList7=CRF_featureCreation(docs7)
    X7_Te,y7_Te,tokenList7_Te=CRF_featureCreation(docs7_Te)
    predicted7_Te,crf1,prob_all_7=CRF_trainer(X7,y7,X7_Te)
    crfDic['pf/mass']=crf1
    if type(predicted7_Te) is not list:
        predicted7_Te=predicted7_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted7_Te,y7_Te,tokenList7_Te,label_dic_C1)
    
    #CRF8=>morphology, distribution and other classifier, input: all positive_finding, negative finding -> calcification
    print "CRF8"
    allReports8=[]
    EachReport1=[]
    allReports8_Te=[]
    EachReport1_Te=[]
    calcPfNodes=tree.findall("./report/positive_finding/calcification")
    calcPfNodes_Te=tree1.findall("./report/positive_finding/calcification")
    CRF8(allReports8,EachReport1,calcPfNodes)
    CRF8(allReports8_Te,EachReport1_Te,calcPfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    calcNfNodes=tree.findall("./report/negative_finding/calcification")
    calcNfNodes_Te=tree1.findall("./report/negative_finding/calcification")
    CRF8(allReports8,EachReport1,calcNfNodes)
    CRF8(allReports8_Te,EachReport1_Te,calcNfNodes_Te)
    docs8=token_LabelCreation(allReports8)
    docs8_Te=token_LabelCreation(allReports8_Te)
    X8,y8,tokenList8=CRF_featureCreation(docs8)
    X8_Te,y8_Te,tokenList8_Te=CRF_featureCreation(docs8_Te)
    predicted8_Te,crf1,prob_all_8=CRF_trainer(X8,y8,X8_Te)
    crfDic['pfnf/calc']=crf1
    if type(predicted8_Te) is not list:
        predicted8_Te=predicted8_Te.tolist()
    for data1,true1,pre1 in zip(tokenList8_Te,y8_Te,predicted8_Te):
        for i in range(0,len(data1)):
            if data1[i] not in string.punctuation:
                out3.write(str(data1[i])+"\t"+str(true1[i])+"\t"+str(pre1[i])+"\n")
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted8_Te,y8_Te,tokenList8_Te,label_dic_C1)
    
    #CRF9=>associated features and other classifier, input: all positive_finding
    print "CRF9"
    allReports9=[]
    EachReport1=[]
    allReports9_Te=[]
    EachReport1_Te=[]
    mass_pfNodes=tree.findall("./report/positive_finding/mass")
    mass_pfNodes_Te=tree1.findall("./report/positive_finding/mass")
    CRF9(allReports9,EachReport1,mass_pfNodes)
    CRF9(allReports9_Te,EachReport1_Te,mass_pfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    calc_pfNodes=tree.findall("./report/positive_finding/calcification")
    calc_pfNodes_Te=tree1.findall("./report/positive_finding/calcification")
    CRF9(allReports9,EachReport1,calc_pfNodes)
    CRF9(allReports9_Te,EachReport1_Te,calc_pfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    asym_pfNodes=tree.findall("./report/positive_finding/asymmetry")
    asym_pfNodes_Te=tree1.findall("./report/positive_finding/asymmetry")
    CRF9(allReports9,EachReport1,asym_pfNodes)
    CRF9(allReports9_Te,EachReport1_Te,asym_pfNodes_Te)
    EachReport1=[]
    EachReport1_Te=[]
    arcDis_pfNodes=tree.findall("./report/positive_finding/architectural_distortion")
    arcDis_pfNodes_Te=tree1.findall("./report/positive_finding/architectural_distortion")
    CRF9(allReports9,EachReport1,arcDis_pfNodes)
    CRF9(allReports9_Te,EachReport1_Te,arcDis_pfNodes_Te)
    docs9=token_LabelCreation(allReports9)
    docs9_Te=token_LabelCreation(allReports9_Te)
    X9,y9,tokenList9=CRF_featureCreation(docs9)
    X9_Te,y9_Te,tokenList9_Te=CRF_featureCreation(docs9_Te)
    predicted9_Te,crf1,prob_all_9=CRF_trainer(X9,y9,X9_Te)
    crfDic['pf/asso']=crf1
    if type(predicted9_Te) is not list:
        predicted9_Te=predicted9_Te.tolist()
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted9_Te,y9_Te,tokenList9_Te,label_dic_C1)
    
    masspfNodes_count=len(tree1.findall("./report/positive_finding/mass"))
    massPfMargin=prob_all_5[:masspfNodes_count]
    massPfSize=prob_all_6[:masspfNodes_count]
    massPfDenShape=prob_all_7[:masspfNodes_count]
    massPfAsso=prob_all_9[:masspfNodes_count]
    massPfMarginTrue=y5_Te[:masspfNodes_count]
    massPfSizeTrue=y6_Te[:masspfNodes_count]
    massPfDenShapeTrue=y7_Te[:masspfNodes_count]
    massPfAssoTrue=y9_Te[:masspfNodes_count]
    massPf=[]
    massPfTrue=[]
    for i in range(masspfNodes_count):
        massPf.append([])
        massPfTrue.append([])
        for j in range(len(massPfMargin[i])):
            label_NE1=filter(lambda a: a[0]!='O', [massPfMargin[i][j],massPfSize[i][j],massPfDenShape[i][j],massPfAsso[i][j]])
            if label_NE1!=[]:
                massPf[i].append(max(label_NE1,key=operator.itemgetter(1)))
            else:
                massPf[i].append('O')
            if massPfMarginTrue[i][j]!='O' or massPfSizeTrue[i][j]!='O' or massPfDenShapeTrue[i][j]!='O' or massPfAssoTrue[i][j]!='O':
                massPfTrue[i].append(filter(lambda x: x!='O', [massPfMarginTrue[i][j],massPfSizeTrue[i][j],massPfDenShapeTrue[i][j],massPfAssoTrue[i][j]])[0])
            else:
                massPfTrue[i].append('O')
    dicListPre['report/positive_finding/mass']=massPf
    dicListTrue['report/positive_finding/mass']=massPfTrue
    #print massPf
    
    calcpfNodes_count=len(tree1.findall("./report/positive_finding/calcification"))
    calcPfSize=prob_all_6[masspfNodes_count:(masspfNodes_count+calcpfNodes_count)]
    calcPfAsso=prob_all_9[masspfNodes_count:(masspfNodes_count+calcpfNodes_count)]
    calcPfMorfDist=prob_all_8[:calcpfNodes_count]
    calcPfSizeTrue=y6_Te[masspfNodes_count:(masspfNodes_count+calcpfNodes_count)]
    calcPfAssoTrue=y9_Te[masspfNodes_count:(masspfNodes_count+calcpfNodes_count)]
    calcPfMorfDistTrue=y8_Te[:calcpfNodes_count]
    calcPf=[]
    calcPfTrue=[]
    for i in range(calcpfNodes_count):
        calcPf.append([])
        calcPfTrue.append([])
        for j in range(len(calcPfSize[i])):
            label_NE2=filter(lambda a: a[0]!='O', [calcPfSize[i][j],calcPfAsso[i][j],calcPfMorfDist[i][j]])
            if label_NE2!=[]:
                #print i
                #print label_NE2
                calcPf[i].append(max(label_NE2,key=operator.itemgetter(1)))
            else:
                calcPf[i].append('O')
            if calcPfSizeTrue[i][j]!='O' or calcPfAssoTrue[i][j]!='O' or calcPfMorfDistTrue[i][j]!='O':
                calcPfTrue[i].append(filter(lambda x: x!='O', [calcPfSizeTrue[i][j],calcPfAssoTrue[i][j],calcPfMorfDistTrue[i][j]])[0])
            else:
                calcPfTrue[i].append('O')
    dicListPre['report/positive_finding/calcification']=calcPf
    dicListTrue['report/positive_finding/calcification']=calcPfTrue
    
    asympfNodes_count=len(tree1.findall("./report/positive_finding/asymmetry"))
    asymPfSize=prob_all_6[(masspfNodes_count+calcpfNodes_count):(masspfNodes_count+calcpfNodes_count+asympfNodes_count)]
    asymPfAsso=prob_all_9[(masspfNodes_count+calcpfNodes_count):(masspfNodes_count+calcpfNodes_count+asympfNodes_count)]
    asymPfSizeTrue=y6_Te[(masspfNodes_count+calcpfNodes_count):(masspfNodes_count+calcpfNodes_count+asympfNodes_count)]
    asymPfAssoTrue=y9_Te[(masspfNodes_count+calcpfNodes_count):(masspfNodes_count+calcpfNodes_count+asympfNodes_count)]
    asymPf=[]
    asymPfTrue=[]
    for i in range(asympfNodes_count):
        asymPf.append([])
        asymPfTrue.append([])
        for j in range(len(asymPfSize[i])):
            label_NE3=filter(lambda a: a[0]!='O', [asymPfSize[i][j],asymPfAsso[i][j]])
            if label_NE3!=[]:
                asymPf[i].append(max(label_NE3,key=operator.itemgetter(1)))
            else:
                asymPf[i].append('O')
            if asymPfSizeTrue[i][j]!='O' or asymPfAssoTrue[i][j]!='O':
                asymPfTrue[i].append(filter(lambda x: x!='O', [asymPfSizeTrue[i][j],asymPfAssoTrue[i][j]])[0])
            else:
                asymPfTrue[i].append('O')
            
    dicListPre['report/positive_finding/asymmetry']=asymPf
    dicListTrue['report/positive_finding/asymmetry']=asymPfTrue
    
    massnfNodes_count=len(tree1.findall("./report/negative_finding/mass"))
    massNfMargin=prob_all_5[masspfNodes_count:]
    massNfMarginTrue=y5_Te[masspfNodes_count:]
    massNf=[]
    massNfTrue=[]
    for i in range(massnfNodes_count):
        massNf.append([])
        massNfTrue.append([])
        for j in range(len(massNfMargin[i])):
            if massNfMargin[i][j][0]!='O':
                massNf[i].append(massNfMargin[i][j])
            else:
                massNf[i].append(massNfMargin[i][j][0])
            massNfTrue[i].append(massNfMarginTrue[i][j])
    dicListPre['report/negative_finding/mass']=massNf
    dicListTrue['report/negative_finding/mass']=massNfTrue
    
    calcnfNodes_count=len(tree1.findall("./report/negative_finding/calcification"))
    calcNfMorfDist=prob_all_8[calcpfNodes_count:]
    calcNfMorfDistTrue=y8_Te[calcpfNodes_count:]
    calcNf=[]
    calcNfTrue=[]
    for i in range(calcnfNodes_count):
        calcNf.append([])
        calcNfTrue.append([])
        for j in range(len(calcNfMorfDist[i])):
            if calcNfMorfDist[i][j][0]!='O':
                calcNf[i].append(calcNfMorfDist[i][j])
            else:
                calcNf[i].append(calcNfMorfDist[i][j][0])
            calcNfTrue[i].append(calcNfMorfDistTrue[i][j])
    dicListPre['report/negative_finding/calcification']=calcNf
    dicListTrue['report/negative_finding/calcification']=calcNfTrue
    
    
    
    #print dicListPre
    dicKeyTrueCount={}
    TruePreLabels=[]
    cascadedOnPre=[]
    locLists=[]
    for i in range(len(tokenList1_Te)):
        TruePreLabels.append([])
        cascadedOnPre.append([])
        locLists.append([])
        for j in range(len(tokenList1_Te[i])):
            TruePreLabels[i].append(['',''])
            cascadedOnPre[i].append(['',''])
            locLists[i].append('')

    for i in range(len(tokenList1_Te)):
        phrase=list(TruePreLabels[i])
        mergingResults(i,0,len(tokenList1_Te[i]),"report",phrase,dicKeyTrueCount)
        #print tokenListAll[i]
        for j in range(len(TruePreLabels[i])):
            TruePreLabels[i][j]=phrase[j]
        #print TruePreLabels[i],"\n\n"
    #print TruePreLabels
    '''k=0
    predicted9Flat=list(itertools.chain(*predicted9_Te))
    y9Flat=list(itertools.chain(*y9_Te))
    for i in range(len(TruePreLabels)):
        for j in range(len(TruePreLabels[i])):
            if TruePreLabels[i][j][0].strip('/').split('/')[0]=='positive_finding':
                if predicted9Flat[k]!='O':
                    pre1=TruePreLabels[i][j][1].split('/')
                    if pre1[len(pre1)-1]=='O':
                        TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)-1])+"/"+predicted9Flat[k].split('-')[len(predicted9Flat[k].split('-'))-1]
                    else:
                        TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)])+"/"+predicted9Flat[k].split('-')[len(predicted9Flat[k].split('-'))-1]
                if y9Flat[k]!='O':
                    true1=TruePreLabels[i][j][0].split('/')
                    if true1[len(true1)-1]=='O':
                        TruePreLabels[i][j][0]="/".join(true1[:len(true1)-1])+"/"+y9Flat[k].split('-')[len(y9Flat[k].split('-'))-1]
                    else:
                        TruePreLabels[i][j][0]="/".join(true1[:len(true1)])+"/"+y9Flat[k].split('-')[len(y9Flat[k].split('-'))-1]
                k=k+1'''
    
    l=0
    m=0
    pfNodes_count=len(pfNodes_Te)
    pfLocation=prob_all_4[:pfNodes_count]
    pfLocationTrue=y4_Te[:pfNodes_count]
    nfLocation=prob_all_4[pfNodes_count:]
    nfLocationTrue=y4_Te[pfNodes_count:]
    pfLocationPreFlat=list(itertools.chain(*pfLocation))
    pfLocationTrueFlat=list(itertools.chain(*pfLocationTrue))
    nfLocationPreFlat=list(itertools.chain(*nfLocation))
    nfLocationTrueFlat=list(itertools.chain(*nfLocationTrue))
    for i in range(len(TruePreLabels)):
        for j in range(len(TruePreLabels[i])):
            if TruePreLabels[i][j][0].strip('/').split('/')[0]=='positive_finding':
                pre1=TruePreLabels[i][j][1].split('/')
                true1=TruePreLabels[i][j][0].split('/')
                if pfLocationPreFlat[l][0]!='O':
                    if pre1[len(pre1)-1]=='O':
                        TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)-1])+"/"+pfLocationPreFlat[l][0].split('-')[len(pfLocationPreFlat[l][0].split('-'))-1]
                    else:
                        if len(TruePreLabels[i][j][0].strip('/').split('/'))==3:
                            pre1_lab_prob=pre1[len(pre1)-1].split(':')
                            if len(pre1_lab_prob)==2:
                                label_max=max([pre1_lab_prob,pfLocationPreFlat[l]], key=operator.itemgetter(1))[0]
                                TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)-1])+"/"+label_max.split('-')[len(label_max.split('-'))-1]
                        elif len(TruePreLabels[i][j][0].strip('/').split('/'))==2:
                            TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)])+"/"+pfLocationPreFlat[l][0].split('-')[len(pfLocationPreFlat[l][0].split('-'))-1]
                else:
                    #print TruePreLabels[i][j]
                    pre1_lab_prob=pre1[len(pre1)-1].split(':')
                    if len(pre1_lab_prob)==2:
                        TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)-1])+"/"+pre1_lab_prob[0]
                    elif len(TruePreLabels[i][j][0].strip('/').split('/'))==2:
                        TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)])+"/"+pfLocationPreFlat[l][0]
                    
                if pfLocationTrueFlat[l]!='O':
                    true1=TruePreLabels[i][j][0].split('/')
                    if true1[len(true1)-1]=='O':
                        TruePreLabels[i][j][0]="/".join(true1[:len(true1)-1])+"/"+pfLocationTrueFlat[l].split('-')[len(pfLocationTrueFlat[l].split('-'))-1]
                    else:
                        TruePreLabels[i][j][0]="/".join(true1[:len(true1)])+"/"+pfLocationTrueFlat[l].split('-')[len(pfLocationTrueFlat[l].split('-'))-1]
                elif pfLocationTrueFlat[l]=='O':
                    if true1[len(true1)-1]!='O' and len(TruePreLabels[i][j][0].strip('/').split('/'))==2:
                        TruePreLabels[i][j][0]="/".join(true1[:len(true1)])+"/"+'O'
                l=l+1
            if TruePreLabels[i][j][0].strip('/').split('/')[0]=='negative_finding':
                pre2=TruePreLabels[i][j][1].split('/')
                true2=TruePreLabels[i][j][0].split('/')
                if nfLocationPreFlat[m][0]!='O':
                    if true2[len(true2)-1]=='O':
                        TruePreLabels[i][j][1]="/".join(pre2[:len(pre2)-1])+"/"+nfLocationPreFlat[m][0].split('-')[len(nfLocationPreFlat[m][0].split('-'))-1]
                    else:
                        if len(TruePreLabels[i][j][0].strip('/').split('/'))==3:
                            pre2_lab_prob=pre2[len(pre2)-1].split(':')
                            if len(pre2_lab_prob)==2:
                                label_max1=max([pre2_lab_prob,nfLocationPreFlat[m]], key=operator.itemgetter(1))[0]
                                TruePreLabels[i][j][1]="/".join(pre2[:len(pre2)-1])+"/"+label_max1.split('-')[len(label_max1.split('-'))-1]
                        elif len(TruePreLabels[i][j][0].strip('/').split('/'))==2:
                            TruePreLabels[i][j][1]="/".join(pre2[:len(pre2)])+"/"+nfLocationPreFlat[m][0].split('-')[len(nfLocationPreFlat[m][0].split('-'))-1]
                else:
                    pre2_lab_prob=pre2[len(pre2)-1].split(':')
                    if len(pre2_lab_prob)==2:
                        TruePreLabels[i][j][1]="/".join(pre2[:len(pre2)-1])+"/"+pre2_lab_prob[0]
                    elif len(TruePreLabels[i][j][0].strip('/').split('/'))==2:
                        if true2[len(true2)-1]!='O':
                            TruePreLabels[i][j][1]="/".join(pre2[:len(pre2)])+"/"+nfLocationPreFlat[m][0]
                        else:
                            TruePreLabels[i][j][1]="/".join(pre2[:len(pre2)-1])+"/"+nfLocationPreFlat[m][0]
                if nfLocationTrueFlat[m]!='O':
                    true2=TruePreLabels[i][j][0].split('/')
                    if true2[len(true2)-1]=='O':
                        TruePreLabels[i][j][0]="/".join(true2[:len(true2)-1])+"/"+nfLocationTrueFlat[m].split('-')[len(nfLocationTrueFlat[m].split('-'))-1]
                    else:
                        TruePreLabels[i][j][0]="/".join(true2[:len(true2)])+"/"+nfLocationTrueFlat[m].split('-')[len(nfLocationTrueFlat[m].split('-'))-1]
                elif nfLocationTrueFlat[m]=='O':
                    if true2[len(true2)-1]!='O' and len(TruePreLabels[i][j][0].strip('/').split('/'))==2:
                        TruePreLabels[i][j][0]="/".join(true2[:len(true2)])+"/"+'O'
                
                m=m+1
            #print TruePreLabels[i][j]    
    
    for i in range(len(tokenList1_Te)):
        preLabels=list(cascadedOnPre[i])
        phrase1=list(tokenList1_Te[i])
        locList=list(locLists[i])
        #print phrase1
        test_onPredicted(0,len(tokenList1_Te[i]),phrase1,preLabels,"report","report",locList)
        for j in range(len(cascadedOnPre[i])):
            cascadedOnPre[i][j]=preLabels[j]
    #print cascadedOnPre[0]
    
    trueList=[]
    preList=[]
    cascPreList=[]
    cascPreList1=[]
    lastLevelPreList=[]
    #level2
    truePredictedList1=[]
    predictPredictList1=[]
    tokenFor2levelList1=[]
    trueLabel_2labelsList1=[]
    #level3
    truetruePredictedList1=[]
    predictpredictPredictList1=[]
    tokenFor3levelList1=[]
    trueLabel_3labelsList1=[]
    print len(TruePreLabels)
    count=0
    for i in range(len(TruePreLabels)):
        trueVal=[]
        preVal=[]
        cascPreVal=[]
        cascPreVal1=[]
        lastLevelPre=[]
        #level2
        truePredicted1=[]
        predictPredict1=[]
        tokenFor2level=[]
        trueLabel_2labels1=[]
        #level3
        truetruePredicted1=[]
        predictpredictPredict1=[]
        tokenFor3level=[]
        trueLabel_3labels1=[]
        #if i==20 or i==29 or i==33 or i==34:
        #    print i, "\t", len(tokenList1[i])
        for j in range(len(TruePreLabels[i])):
            TruePreLabels[i][j][0]=TruePreLabels[i][j][0].strip('/')
            TruePreLabels[i][j][1]=TruePreLabels[i][j][1].strip('/')
            cascadedOnPre[i][j][0]=cascadedOnPre[i][j][0].strip('/')
            cascadedOnPre[i][j][1]=cascadedOnPre[i][j][1].strip('/')
            trueLabel=TruePreLabels[i][j][0].split('/')
            truePredict=TruePreLabels[i][j][1].split('/')
            cascPre=cascadedOnPre[i][j][0].split('/')
            firstSecond_trueLabel=trueLabel[:len(trueLabel)-1]
            if firstSecond_trueLabel!=[]:
                lastLevelPreLabel='/'.join(TruePreLabels[i][j][0].split('/')[:len(TruePreLabels[i][j][0].split('/'))-1])+"/"+str(TruePreLabels[i][j][1].split('/')[len(TruePreLabels[i][j][1].split('/'))-1])
            else:
                lastLevelPreLabel=str(TruePreLabels[i][j][1].split('/')[len(TruePreLabels[i][j][1].split('/'))-1])
            if len(trueLabel)>=2:
                truePredicted=trueLabel[0]+"/"+truePredict[1]
                truePredicted1.append(truePredicted)
            if len(trueLabel)>=2 or len(cascPre)>=2:
                predictPredict="/".join(cascPre[:2])
                trueLabel_2labels="/".join(trueLabel[:2])
                #if trueLabel_2labels=="negative_finding/O":
                #    print predictPredict
                #    print truePredicted
                tokenFor2level.append(tokenList1_Te[i][j])
                predictPredict1.append(predictPredict)
                trueLabel_2labels1.append(trueLabel_2labels)
                
            if len(trueLabel)==3:
                #print trueLabel
                truetruePredicted=trueLabel[0]+"/"+trueLabel[1]+"/"+truePredict[2]
                predictpredictPredict="/".join(cascPre[0:3])
                trueLabel_3labels="/".join(trueLabel[:3])
                #print truetruePredicted, truepredictPredict, trueLabel_3labels
                #if trueLabel_2labels=="negative_finding/O":
                #    print predictPredict
                #    print truePredicted
                tokenFor3level.append(tokenList1_Te[i][j])
                truetruePredicted1.append(truetruePredicted)
                predictpredictPredict1.append(predictpredictPredict)
                trueLabel_3labels1.append(trueLabel_3labels)
            '''if TruePreLabels[i][j][0]!='O':
                true1=TruePreLabels[i][j][0].split('/')
                if true1[len(true1)-1]=='O':
                    TruePreLabels[i][j][0]="/".join(true1[:len(true1)-1])
            if TruePreLabels[i][j][1]!='O':
                pre1=TruePreLabels[i][j][1].split('/')
                if pre1[len(pre1)-1]=='O':
                    TruePreLabels[i][j][1]="/".join(pre1[:len(pre1)-1])
            if cascadedOnPre[i][j][0]!='O':
                cascPre1=cascadedOnPre[i][j][0].split('/')
                cascPre2=cascadedOnPre[i][j][1].split('/')
                if cascPre1[len(cascPre1)-1]=='O':
                    cascadedOnPre[i][j][0]="/".join(cascPre1[:len(cascPre1)-1])
                    cascadedOnPre[i][j][1]="/".join(cascPre2[:len(cascPre2)-1])'''
            trueVal.append(TruePreLabels[i][j][0])
            preVal.append(TruePreLabels[i][j][1])
            cascPreVal.append(cascadedOnPre[i][j][0])
            cascPreVal1.append(cascadedOnPre[i][j][1])
            lastLevelPre.append(lastLevelPreLabel)
            #print tokenListAll[i][j]
            if tokenList1_Te[i][j] not in string.punctuation:
                count=count+1
                out.write(tokenList1_Te[i][j].decode('utf-8')+"\t"+str(TruePreLabels[i][j][0])+"\t"+str(TruePreLabels[i][j][1])+"\t"+str(cascadedOnPre[i][j][0])+"\n")
        trueList.append(trueVal)
        preList.append(preVal)
        cascPreList.append(cascPreVal)
        cascPreList1.append(cascPreVal1)
        
        lastLevelPreList.append(lastLevelPre)
        truePredictedList1.append(truePredicted1)
        predictPredictList1.append(predictPredict1)
        trueLabel_2labelsList1.append(trueLabel_2labels1)
        tokenFor2levelList1.append(tokenFor2level)
        
        truetruePredictedList1.append(truetruePredicted1)
        predictpredictPredictList1.append(predictpredictPredict1)
        trueLabel_3labelsList1.append(trueLabel_3labels1)
        tokenFor3levelList1.append(tokenFor3level)
    #CRF_measures_cascadedCRF.tokenLevel_measures(preList,trueList,tokenList1_Te)
    #print count
    #CRF_measures_cascadedCRF.tokenLevel_measures(cascPreList,trueList,tokenList1_Te,label_dic_all)
    #global labels predicted on true values (comparison between true/true/predicted & cascaded predicted/predicted/predicted)
    #CRF_measures_cascadedCRF.tokenLevel_measures(lastLevelPreList,trueList,tokenList1_Te)
    #Level_2 on true (true/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(truePredictedList1,trueLabel_2labelsList1,tokenFor2levelList1,label_dic_2_true)
    #level_2 on predict (predict/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(predictPredictList1,trueLabel_2labelsList1,tokenFor2levelList1,label_dic_2_pre)
    #level_3 on true (true/true/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(truetruePredictedList1,trueLabel_3labelsList1,tokenFor3levelList1,label_dic_3_true)
    #level_3 on predict (predict/predict/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(predictpredictPredictList1,trueLabel_3labelsList1,tokenFor3levelList1,label_dic_3_pre)
    #print docs1_Te_ori
    if cvf==3:
        print "hi"
        labeling_to_xml.mainFunc(docs1_Te_ori,cascPreList1)

'''label_dic_abb={'O':'O','breast_composition':'BC','positive_finding/mass/location':'PF/MS/L','positive_finding/mass/size':'PF/MS/SI','positive_finding/mass/margin':'PF/MS/MA','positive_finding/mass/density':'PF/MS/DE','positive_finding/mass/associated_features':'PF/MS/AF','positive_finding/mass/shape':'PF/MS/SH','positive_finding/mass/O':'PF/MS/O','positive_finding/calcification/location':'PF/C/L',\
               'positive_finding/calcification/size':'PF/C/SI','positive_finding/calcification/morphology':'PF/C/MO','positive_finding/calcification/distribution':'PF/C/DI','positive_finding/calcification/associated_features':'PF/C/AF','positive_finding/calcification/O':'PF/C/O','positive_finding/architectural_distortion/location':'PF/AD/L','positive_finding/architectural_distortion/associated_features':'PF/AD/AF',\
               'positive_finding/architectural_distortion/O':'PF/AD/O','positive_finding/associated_features/location':'PF/AF/L','positive_finding/associated_features/O':'PF/AF/O','positive_finding/asymmetry/location':'PF/AS/L','positive_finding/asymmetry/size':'PF/AS/SI','positive_finding/asymmetry/associated_features':'PF/AS/AF','positive_finding/asymmetry/O':'PF/AS/O','negative_finding/mass/location':'NF/MS/L',\
               'negative_finding/mass/margin':'NF/MS/MA','negative_finding/mass/O':'NF/MS/O','negative_finding/calcification/location':'NF/C/L','negative_finding/calcification/morphology':'NF/C/MO','negative_finding/calcification/distribution':'NF/C/DI','negative_finding/calcification/O':'NF/C/O','negative_finding/architectural_distortion/location':'NF/AD/L','negative_finding/architectural_distortion/O':'NF/AD/O',\
               'negative_finding/associated_features/location':'NF/AF/L','negative_finding/associated_features/O':'NF/AF/O','negative_finding/asymmetry/location':'NF/AS/L','negative_finding/asymmetry/O':'NF/AS/O','negative_finding/location':'NF/L','negative_finding/O':'NF/O'}
label_dic1={}
for key in label_dic_all.iterkeys():
    label_dic1[label_dic_abb[key]]=label_dic_all[key][2][0]

axis_labels=sorted(label_dic1,key=label_dic1.__getitem__)
conf_mat_agg=conf_mat_agg.astype(int)
#print conf_mat_agg
conf_mat_agg_norm=(np.zeros((34,34))).astype('float')
for i in range(len(conf_mat_agg)):
    s=np.sum(conf_mat_agg[i,:])
    for j in range(len(conf_mat_agg[i])):
        conf_mat_agg_norm[i,j]=float(conf_mat_agg[i,j])/s
sns.set()
f=plt.figure(figsize=(8,5))
sns.heatmap(
        yticklabels=axis_labels,
        xticklabels=axis_labels,
        data=conf_mat_agg_norm,
        cmap='YlGnBu',
        #annot=True,
        #fmt="d",
        linewidths=0.75)
#plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#plt.show()
f.savefig("ConfusionMatrixHeatmap_modelB.pdf",bbox_inches='tight')'''
'''label_dic2_fscore={}
label_dic2_support={}
for key in label_dic_2_pre.iterkeys():
    label_dic2_fscore[key]=float(sum(label_dic_2_pre[key][0]))/len(label_dic_2_pre[key][0])
    label_dic2_support[key]=label_dic_2_pre[key][1][0]
print label_dic2_fscore
print label_dic2_support'''

'''label_dic_2_fscore={}
label_dic_2_support={}
for key in label_dic_all.iterkeys():
    label_dic_2_fscore[key]=float(sum(label_dic_all[key][0]))/len(label_dic_all[key][0])
    label_dic_2_support[key]=label_dic_all[key][1][0]
print label_dic_2_fscore
print label_dic_2_support'''