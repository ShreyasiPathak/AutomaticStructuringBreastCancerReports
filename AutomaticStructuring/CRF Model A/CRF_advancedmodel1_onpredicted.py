import xml.etree.cElementTree as Elem

import re
import nltk
import string
import numpy as np
import sys
import sklearn
from sklearn.model_selection import cross_val_predict, ShuffleSplit, KFold
from nltk.tokenize import RegexpTokenizer
#from sklearn.grid_search import RandomizedSearchCV

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import xml
from xml.etree.ElementTree import Element, ElementTree

from sklearn.metrics import classification_report
import CRF_features_cascadedCRF
import CRF_measures_cascadedCRF

reload(sys)
sys.setdefaultencoding('utf8')

def dataPreprocessing(allReports):
    for report in allReports:
        #print report
        EachReport=[]
        for line in report:
            #print line
            label=line[1]
            #print label
            txt=line[0].strip()
            #date=re.search(r'\d\d-\d\d-\d\d\d\d',txt)
            #if date!=None:
            #    print line
            #    date_new=re.sub(r'-','/',date.group())
            #    txt=re.sub(r'\d\d-\d\d-\d\d\d\d',date_new,txt)
            txt=re.sub(r'\d',"#NUM",txt)
            tovPat=re.compile(r't,o,v',re.IGNORECASE)
            txt=tovPat.sub('tov',txt)
            #txt=re.sub(r'#NUM#NUM-#NUM#NUM-#NUM#NUM#NUM#NUM',"#NUM#NUM/#NUM#NUM/#NUM#NUM#NUM#NUM",txt)
            tokens=re.split(r'([,\(\).?:-]*)\s*',txt)
            tokens=filter(lambda a: a!='', tokens)

def token_LabelCreation(allReports):
    docs=[]
    data=[]
    for report in allReports:
        #print report
        EachReport=[]
        for line in report:
            #print line
            label=line[1]
            #print label
            txt=line[0].strip()
            #date=re.search(r'\d\d-\d\d-\d\d\d\d',txt)
            #if date!=None:
            #    print line
            #    date_new=re.sub(r'-','/',date.group())
            #    txt=re.sub(r'\d\d-\d\d-\d\d\d\d',date_new,txt)
            txt=re.sub(r'\d',"#NUM",txt)
            tovPat=re.compile(r't,o,v',re.IGNORECASE)
            txt=tovPat.sub('tov',txt)
            #txt=re.sub(r'#NUM#NUM-#NUM#NUM-#NUM#NUM#NUM#NUM',"#NUM#NUM/#NUM#NUM/#NUM#NUM#NUM#NUM",txt)
            tokens=re.split(r'([,\(\).?:-]*)\s*',txt)
            tokens=filter(lambda a: a!='', tokens)
            #tokens=txt.split()
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
    return docs

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
    #crf.predict(X)
    crf.fit(xTrain,yTrain)
    #print xTest
    predicted=crf.predict(xTest)
    #print predicted
    return crf,predicted

def CRF_predict(crf,xTest):
    #cv=ShuffleSplit(n_split=2,test_size=0.5,random_state=0)
    #crf.predict(X)
    xTest=[xTest]
    predicted2=crf.predict(xTest)
    #print predicted2
    return predicted2

def textExtraction(reportNodes):
    allReports=[]
    nodeAr=[]
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
            if node.tag not in nodeAr:
                nodeAr.append(node.tag)
            for text in node.itertext():
                text=re.sub(r'\t|\n','',text)
                nodeText.append(text.strip())
                #print nodeText
                #nodeText.append(text.strip().strip('\t').strip('\n').strip())
            oneReport.append((" ".join(nodeText),node.tag))
            if re.search(r'\S',node.tail):
                nodeTail=re.sub(r'\t|\n','',node.tail)
                #print nodeTail
                oneReport.append((nodeTail.strip(),'O'))
        allReports.append(oneReport)
    return allReports,nodeAr

def train_test_onTrue(currentNode):
    reportNodes1=tree.findall("./"+currentNode)
    reportNodes2=tree1.findall("./"+currentNode)
    #print currentNode
    allReports1,nodeAr1=textExtraction(reportNodes1)
    allReports2,nodeAr2=textExtraction(reportNodes2)
    #print allReports1
    #np.random.shuffle(allReports1)
    docs1=token_LabelCreation(allReports1)
    X1,y1,tokenList1=CRF_featureCreation(docs1)
    docs2=token_LabelCreation(allReports2)
    X2,y2,tokenList2=CRF_featureCreation(docs2)
    #print tokenList1
    #print ReportList1
    crf1,predicted2=CRF_trainer(X1,y1,X2)
    #print predicted2[0]
    if currentNode=='report':
        global tokenListAll
        tokenListAll=tokenList2
        #print predicted2[0]
    crfDic[currentNode]=crf1
    if type(predicted2) is list:
        dicListPre[currentNode]=predicted2
        #if currentNode=='report/negative_finding':
        #    print zip(tokenList1,y1,predicted1)
    else:
        dicListPre[currentNode]=predicted2.tolist()
        #if currentNode=='report/negative_finding':
        #    print zip(tokenList1,y1,predicted1.tolist())
    dicListTrue[currentNode]=y2
    #if currentNode=='report/negative_finding':
        #print allReports1
        #print docs1
        #print predicted1
    #individual performance of each classifier, comparing the true and predicted(for prediction, trained on true values)
    if currentNode=="report/negative_finding/asymmetry" and y2!=[]:    
        #CRF_measures_cascadedCRF.partialPhraseLevel_measures(tokenList2,predicted2,y2)
        CRF_measures_cascadedCRF.tokenLevel_measures(predicted2,y2,tokenList2,label_dic_2_pre)
    for node in nodeAr1:
        child=str(currentNode)+"/"+str(node)
        if node=='positive_finding' or node=='negative_finding' \
        or node=='mass' or node=='calcification' \
        or node=='asymmetry' or node=='architectural_distortion' \
        or child=='report/positive_finding/associated_features' \
        or child=='report/negative_finding/associated_features': 
            train_test_onTrue(child)

def mergingResults(i,m,n,labelStartTrue,phrase,dicKeyTrueCount1):
    #print labelStartTrue
    #print dicListTrue[labelStartTrue]
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
        pre1List=currentNodeInstancePre[k].split('-')
        #print phrase[j]
        phrase[j][0]=phrase[j][0]+"/"+labelTrue[len(labelTrue)-1]
        phrase[j][1]=phrase[j][1]+"/"+currentNodeInstanceTrue[k]
        #print j
        phrase[j][2]=phrase[j][2]+"/"+pre1List[len(pre1List)-1]
        phrase[j][3]=phrase[j][3]+"/"+currentNodeInstancePre[k]
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
    #if currentNode=='report/negative_finding':
        #print allReports1
        #print docs1
        #print predicted1
    #CRF_measures_cascadedCRF.tokenLevel_measures(predicted1,y1,tokenList1)
            
def indivClassi_cascPredPerf(classifierName,trueList1,cascPreList):
    yTrueAll=[]
    yTrue=[]
    yPreAll=[]
    yPre=[]
    tokenAll=[]
    tokenHere=[]
    for i in range(len(tokenListAll)):
        for j in range(len(tokenListAll[i])):
            clNameLen=len(classifierName.split('/'))
            trueLabelList=trueList1[i][j].split('/')
            labelWoBITrue=[]
            for item in trueLabelList:
                itemList=item.split('-')
                labelWoBITrue.append(str(itemList[len(itemList)-1]))
            trueLabel="/".join(labelWoBITrue[:clNameLen])
            #print trueLabel
            if len(labelWoBITrue)>=clNameLen+1:
                trueLabelChild="/".join(labelWoBITrue[:clNameLen+1])
                #print trueLabelChild
                if trueLabel==classifierName:
                    #print trueLabel, "\t", classifierName
                    #print trueLabelList
                    trueLabelList1=trueLabelList[clNameLen-1].split('-')
                    #print trueLabelList1
                    labelStartPres=trueLabelList1[0]
                    labelEndPres=trueLabelList1[len(trueLabelList1)-1]
                    #print preLabels
                    if j!=len(tokenListAll[i])-1:
                        if len(trueList1[i][j+1].split('/'))>=clNameLen:
                            labelStartNext=trueList1[i][j+1].split('/')[clNameLen-1].split('-')[0]
                        else:
                            labelStartNext='O'
                    else:
                        labelStartNext=None
                    if labelStartPres=='B':
                        tokenHere.append(tokenListAll[i][j])
                        yTrue.append(trueLabelChild)
                        preLabelChild="/".join(cascPreList[i][j].split('/')[:clNameLen+1])
                        #print trueLabelChild
                        #print preLabelChild
                        yPre.append(preLabelChild)
                        if labelStartNext=='B' or labelStartNext=='O' or j==len(tokenListAll[i])-1:
                            tokenAll.append(tokenHere)
                            tokenHere=[]
                            #print yTrue
                            yTrueAll.append(yTrue)
                            yTrue=[]
                            #print yPre
                            yPreAll.append(yPre)
                            yPre=[]
                    elif labelStartPres=='I':
                        tokenHere.append(tokenListAll[i][j])
                        yTrue.append(trueLabelChild)
                        preLabelChild="/".join(cascPreList[i][j].split('/')[:clNameLen+1])
                        yPre.append(preLabelChild)
                        if labelStartNext=='B' or labelStartNext=='O' or j==len(tokenListAll[i])-1:
                            tokenAll.append(tokenHere)
                            tokenHere=[]
                            #print yTrue
                            yTrueAll.append(yTrue)
                            yTrue=[]
                            #print yPre
                            yPreAll.append(yPre)
                            yPre=[]
    #print yPreAll
    out2=open('positive_finding_classifier.txt','w')
    CRF_measures_cascadedCRF.tokenLevel_measures(yPreAll,yTrueAll,tokenAll)
    for pre1,true1,token1 in zip(yPreAll,yTrueAll,tokenAll):
        for i in range(len(pre1)):
            if token1[i] not in string.punctuation:
                out2.write(str(token1[i])+"\t"+str(true1[i])+"\t"+str(pre1[i])+"\n")

#tree = Elem.parse('./../labeling/train_shuffled_70_30.xml')
#root=tree.getroot()
#tree1 = Elem.parse('./../labeling/test_shuffled_70_30.xml')
#out=open('CRF_level1_file.txt','w')
tree_all = Elem.parse('./../labeling/new_data.xml')
list_tree=tree_all.findall('report')
out1=open('Token_True_Predicted_Labels',"a")
k=len(list_tree)/4
label_dic_all={}
label_dic_2={}
label_dic_2_true={}
label_dic_2_pre={}
label_dic_3_true={}
label_dic_3_pre={}
conf_mat_agg=np.zeros((34,34))
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
    print "length:",len(root)
    for list_tree_elem in list_tree_test:
        root1.append(list_tree_elem)
    print "length:",len(root1)
    tree=ElementTree(root)
    tree1=ElementTree(root1)
    
    dicListTrue={}
    dicListPre={}
    crfDic={}
    train_test_onTrue("report")
    #print crfDic
    #print dicListPre
    
    #for key in dicListPre.keys():
    #    print key,"\n",dicListPre[key]
    
    dicKeyTrueCount={}
    TruePreLabels=[]
    cascadedOnPre=[]
    for i in range(len(tokenListAll)):
        TruePreLabels.append([])
        cascadedOnPre.append([])
        for j in range(len(tokenListAll[i])):
            TruePreLabels[i].append(['','','',''])
            cascadedOnPre[i].append(['',''])
    
    #print len(tokenListAll[0])
    #print TruePreLabels[0]
    
    for i in range(len(tokenListAll)):
        phrase=list(TruePreLabels[i])
        mergingResults(i,0,len(tokenListAll[i]),"report",phrase,dicKeyTrueCount)
        #print tokenListAll[i]
        for j in range(len(TruePreLabels[i])):
            TruePreLabels[i][j]=phrase[j]
        #print TruePreLabels[i],"\n\n"
    
    for i in range(len(tokenListAll)):
        preLabels=list(cascadedOnPre[i])
        phrase1=list(tokenListAll[i])
        #print phrase1
        test_onPredicted(0,len(tokenListAll[i]),phrase1,preLabels,"report")
        for j in range(len(cascadedOnPre[i])):
            cascadedOnPre[i][j]=preLabels[j]
            #print cascadedOnPre[i]
    
    trueList=[]
    trueList1=[]
    preList=[]
    preList1=[]
    lastLevelPreList=[]
    cascPreList=[]
    cascPreList1=[]
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
    
    for i in range(len(TruePreLabels)):
        trueVal=[]
        trueVal1=[]
        preVal=[]
        preVal1=[]
        lastLevelPre=[]
        cascPreVal=[]
        cascPreVal1=[]
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
        #    print i, "\n", tokenListAll[i]
        for j in range(len(TruePreLabels[i])):
            TruePreLabels[i][j][0]=TruePreLabels[i][j][0].strip('/')
            TruePreLabels[i][j][1]=TruePreLabels[i][j][1].strip('/')
            TruePreLabels[i][j][2]=TruePreLabels[i][j][2].strip('/')
            TruePreLabels[i][j][3]=TruePreLabels[i][j][3].strip('/')
            cascadedOnPre[i][j][0]=cascadedOnPre[i][j][0].strip('/')
            cascadedOnPre[i][j][1]=cascadedOnPre[i][j][1].strip('/')
            trueLabel=TruePreLabels[i][j][0].split('/')
            truePredict=TruePreLabels[i][j][2].split('/')
            cascPre=cascadedOnPre[i][j][0].split('/')
            firstSecond_trueLabel=trueLabel[:len(trueLabel)-1]
            if firstSecond_trueLabel!=[]:
                lastLevelPreLabel='/'.join(TruePreLabels[i][j][0].split('/')[:len(TruePreLabels[i][j][0].split('/'))-1])+"/"+str(TruePreLabels[i][j][2].split('/')[len(TruePreLabels[i][j][2].split('/'))-1])
            else:
                lastLevelPreLabel=str(TruePreLabels[i][j][2].split('/')[len(TruePreLabels[i][j][2].split('/'))-1])
            if len(trueLabel)>=2:
                truePredicted=trueLabel[0]+"/"+truePredict[1]
                truePredicted1.append(truePredicted)
            if len(trueLabel)>=2 or len(cascPre)>=2:
                predictPredict="/".join(cascPre[:2])
                trueLabel_2labels="/".join(trueLabel[:2])
                #if trueLabel_2labels=="negative_finding/O":
                #    print predictPredict
                #    print truePredicted
                tokenFor2level.append(tokenListAll[i][j])
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
                tokenFor3level.append(tokenListAll[i][j])
                truetruePredicted1.append(truetruePredicted)
                predictpredictPredict1.append(predictpredictPredict)
                trueLabel_3labels1.append(trueLabel_3labels)
            #print lastLevelPreLabel,TruePreLabels[i][j][0]
            '''if TruePreLabels[i][j][0]!='O':
                true1=TruePreLabels[i][j][0].split('/')
                true2=TruePreLabels[i][j][1].split('/')
                #if true1[len(true1)-1]=='O':
                TruePreLabels[i][j][0]="/".join(true1[:len(true1)])
                TruePreLabels[i][j][1]="/".join(true2[:len(true2)])
            if TruePreLabels[i][j][2]!='O':
                pre1=TruePreLabels[i][j][2].split('/')
                pre2=TruePreLabels[i][j][3].split('/')
                #if pre1[len(pre1)-1]=='O':
                TruePreLabels[i][j][2]="/".join(pre1[:len(pre1)])
                TruePreLabels[i][j][3]="/".join(pre2[:len(pre2)])
            if cascadedOnPre[i][j][0]!='O':
                cascPre1=cascadedOnPre[i][j][0].split('/')
                cascPre2=cascadedOnPre[i][j][1].split('/')
                #if cascPre1[len(cascPre1)-1]=='O':
                cascadedOnPre[i][j][0]="/".join(cascPre1[:len(cascPre1)])
                cascadedOnPre[i][j][1]="/".join(cascPre2[:len(cascPre2)])'''
            trueVal.append(TruePreLabels[i][j][0])
            trueVal1.append(TruePreLabels[i][j][1])
            preVal.append(TruePreLabels[i][j][2])
            preVal1.append(TruePreLabels[i][j][3])
            lastLevelPre.append(lastLevelPreLabel)
            cascPreVal.append(cascadedOnPre[i][j][0])
            cascPreVal1.append(cascadedOnPre[i][j][1])
            
            #print tokenListAll[i][j]
            if tokenListAll[i][j] not in string.punctuation:
                out1.write(tokenListAll[i][j].decode('utf-8')+"\t"+str(TruePreLabels[i][j][1])+"\t"+str(TruePreLabels[i][j][0])+"\t"+str(cascadedOnPre[i][j][0])+"\n")
        trueList.append(trueVal)
        trueList1.append(trueVal1)
        preList.append(preVal)
        preList1.append(preVal1)
        lastLevelPreList.append(lastLevelPre)
        cascPreList.append(cascPreVal)
        cascPreList1.append(cascPreVal1)
        
        truePredictedList1.append(truePredicted1)
        predictPredictList1.append(predictPredict1)
        trueLabel_2labelsList1.append(trueLabel_2labels1)
        tokenFor2levelList1.append(tokenFor2level)
        
        truetruePredictedList1.append(truetruePredicted1)
        predictpredictPredictList1.append(predictpredictPredict1)
        trueLabel_3labelsList1.append(trueLabel_3labels1)
        tokenFor3levelList1.append(tokenFor3level)
    #global labels predicted on true values
    #CRF_measures_cascadedCRF.tokenLevel_measures(preList,trueList,tokenListAll)
    #global labels predicted on predicted values--->cascaded prediction used
    CRF_measures_cascadedCRF.tokenLevel_measures(cascPreList,trueList,tokenListAll,label_dic_all,conf_mat_agg)
    #global labels predicted on true values (comparison between true/true/predicted & cascaded predicted/predicted/predicted)
    #CRF_measures_cascadedCRF.tokenLevel_measures(lastLevelPreList,trueList,tokenListAll)
    #Level_2 on true (true/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(truePredictedList1,trueLabel_2labelsList1,tokenFor2levelList1,label_dic_2_true)
    #level_2 on predict (predict/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(predictPredictList1,trueLabel_2labelsList1,tokenFor2levelList1,label_dic_2_pre)
    #level_3 on true (true/true/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(truetruePredictedList1,trueLabel_3labelsList1,tokenFor3levelList1,label_dic_3_true)
    #level_3 on predict (predict/predict/predict)
    #CRF_measures_cascadedCRF.tokenLevel_measures(predictpredictPredictList1,trueLabel_3labelsList1,tokenFor3levelList1,label_dic_3_pre)
    #print cascPreList1
    #indivClassi_cascPredPerf("negative_finding",trueList1,cascPreList)
    #X_1,tokenList_1=CRF_featureCreationTest(['Kleine','verkalking','ongewijzigd','rechts','caudal','Mediaal', 'caudaal', 'rechts','kleine','lage', 'radiopake','massa', 'met', 'een', 'doorsnede', 'van' '9','?','mm',','])
    #X_1,tokenList_1=CRF_featureCreationTest(['Kleine','verkalking','ongewijzigd','rechts','caudal'])
    #predicted_1=crfDic['report/positive_finding'].predict([X_1])
    #print crfDic['report/positive_finding'].predict_marginals([X_1])
    #print predicted_1

#np.set_printoptions(threshold=np.nan, suppress=True,linewidth=100)
#print conf_mat_agg
#label_dic1={}
#for key in label_dic_all.iterkeys():
#    label_dic1[key]=conf_mat_agg[label_dic_all[key][2][0]]
#print label_dic1
label_dic_abb={'O':'O','breast_composition':'BC','positive_finding/mass/location':'PF/MS/L','positive_finding/mass/size':'PF/MS/SI','positive_finding/mass/margin':'PF/MS/MA','positive_finding/mass/density':'PF/MS/DE','positive_finding/mass/associated_features':'PF/MS/AF','positive_finding/mass/shape':'PF/MS/SH','positive_finding/mass/O':'PF/MS/O','positive_finding/calcification/location':'PF/C/L',\
               'positive_finding/calcification/size':'PF/C/SI','positive_finding/calcification/morphology':'PF/C/MO','positive_finding/calcification/distribution':'PF/C/DI','positive_finding/calcification/associated_features':'PF/C/AF','positive_finding/calcification/O':'PF/C/O','positive_finding/architectural_distortion/location':'PF/AD/L','positive_finding/architectural_distortion/associated_features':'PF/AD/AF',\
               'positive_finding/architectural_distortion':'PF/AD/O','positive_finding/associated_features/location':'PF/AF/L','positive_finding/associated_features/O':'PF/AF/O','positive_finding/asymmetry/location':'PF/AS/L','positive_finding/asymmetry/size':'PF/AS/SI','positive_finding/asymmetry/associated_features':'PF/AS/AF','positive_finding/asymmetry/O':'PF/AS/O','negative_finding/mass/location':'NF/MS/L',\
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
f.savefig("ConfusionMatrixHeatmap_modelA.pdf",bbox_inches='tight')
#plt.show()'''
'''label_dic2_fscore={}
label_dic2_support={}
for key in label_dic1.iterkeys():
    label_dic2_fscore[key]=float(sum(label_dic1[key][0]))/len(label_dic1[key][0])
    label_dic2_support[key]=label_dic1[key][1][0]
print label_dic2_fscore
print label_dic2_support'''

'''label_dic_2_fscore={}
label_dic_2_support={}
for key in label_dic_2_pre.iterkeys():
    label_dic_2_fscore[key]=float(sum(label_dic_2_pre[key][0]))/len(label_dic_2_pre[key][0])
    label_dic_2_support[key]=label_dic_2_pre[key][1][0]
print label_dic_2_fscore
print label_dic_2_support'''