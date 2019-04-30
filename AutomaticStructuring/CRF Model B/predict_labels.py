import re
import nltk
import string
import numpy as np
import sys
import sklearn
import pickle
import operator
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
    label_prob=[]
    predicted2=crf.predict_single(xTest)
    pred_prob=crf.predict_marginals_single(xTest)
    for dic_label in pred_prob:
        label_prob.append(list(max(dic_label.iteritems(), key=operator.itemgetter(1))))
    return predicted2,label_prob

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

def max_label_cal(prob_All1,preLabels1,beg1,end1,preLabelLoc):
    z=0
    locLab=preLabelLoc[beg1:end1+1]
    for i1 in range(beg1,end1+1):
        List_NE=filter(lambda a:a[0]!='O',prob_All1[z])
        if List_NE!=():
            label_now1=max(List_NE, key=operator.itemgetter(1))
            if locLab[z][0]!='O':
                if label_now1[1]>locLab[z][1]:
                    label_now=label_now1[0]
                else:
                    label_now=locLab[z][0]
            else:
                label_now=label_now1[0]
            preLabels1[i1][0]=preLabels1[i1][0]+"/"+label_now.split('-')[1]
            preLabels1[i1][1]=preLabels1[i1][1]+"/"+label_now
        else:
            preLabels1[i1][0]=preLabels1[i1][0]+"/"+locLab[z][0].split('-')[len(locLab[z][0].split('-'))-1]
            preLabels1[i1][1]=preLabels1[i1][1]+"/"+locLab[z][0]
        z=z+1
    return preLabels1

def test_onPredicted(beg,end,phrase,preLabels,crfDicKey,currentNode,preLabelLoc):
    phrase2=phrase[beg:end]
    X1,tokenList1=CRF_featureCreationTest(phrase2)
    predicted1,prob=CRF_predict(crfDic[crfDicKey],X1)
    if type(predicted1) is not list:
        predicted1=predicted1.tolist()
    preTokenList=zip(predicted1,tokenList1)
    j=0
    for i in range(beg,end):
        pre1=preTokenList[j][0]
        data1=preTokenList[j][1]
        pre1List=pre1.split('-')
        labelStartPres=pre1List[0]
        labelEndPres=pre1List[len(pre1List)-1]
        preLabels[i][0]=preLabels[i][0]+"/"+labelEndPres
        preLabels[i][1]=preLabels[i][1]+"/"+pre1
        if j!=len(preTokenList)-1:
            labelStartNext=preTokenList[j+1][0].split('-')[0]
        else:
            labelStartNext=None
        if preLabels[i][0]=="/negative_finding/O":
            preLabels[i][0]='/'.join(preLabels[i][0].split('/')[:len(preLabels[i][0].split('/'))-1])+"/"+preLabelLoc[i][0].split('-')[len(preLabelLoc[i][0].split('-'))-1]
            preLabels[i][1]='/'.join(preLabels[i][1].split('/')[:len(preLabels[i][1].split('/'))-1])+"/"+preLabelLoc[i][0]
        if labelStartPres=='B':
            beg=i
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
                if child=="report/negative_finding":
                    phrase10=phrase[beg:end+1]
                    X1_ch10,tokenList_ch10=CRF_featureCreationTest(phrase10)
                    predicted1_ch10_loc,prob_ch10_loc=CRF_predict(crfDic['pfnf'],X1_ch10)
                    z2=0
                    for i1 in range(beg,end+1):
                        preLabelLoc[i1]=prob_ch10_loc[z2]
                        z2=z2+1
                    test_onPredicted(beg,end+1,phrase,preLabels,'negative_finding',child,preLabelLoc)
                    
                if child=="report/positive_finding/mass":
                    phrase5=phrase[beg:end+1]
                    X1_ch5,tokenList_ch5=CRF_featureCreationTest(phrase5)
                    predicted1_ch5_ShDen,prob_ch5_ShDen=CRF_predict(crfDic['pf/mass'],X1_ch5)
                    predicted1_ch5_Si,prob_ch5_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch5)
                    predicted1_ch5_Mar,prob_ch5_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch5)
                    predicted1_ch5_asso,prob_ch5_asso=CRF_predict(crfDic['pf/asso'],X1_ch5)
                    prob_All=zip(prob_ch5_ShDen,prob_ch5_Si,prob_ch5_Mar,prob_ch5_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/calcification":
                    phrase6=phrase[beg:end+1]
                    X1_ch6,tokenList_ch6=CRF_featureCreationTest(phrase6)
                    predicted1_ch6_Si,prob_ch6_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch6)
                    predicted1_ch6_MoDi,prob_ch6_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch6)
                    predicted1_ch6_asso,prob_ch6_asso=CRF_predict(crfDic['pf/asso'],X1_ch6)
                    prob_All_6=zip(prob_ch6_Si,prob_ch6_MoDi,prob_ch6_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_6,preLabels,beg,end,preLabelLoc)
                
                if child=="report/positive_finding/architectural_distortion":
                    phrase7=phrase[beg:end+1]
                    X1_ch7,tokenList_ch7=CRF_featureCreationTest(phrase7)
                    predicted1_ch7_asso,prob_ch7_asso=CRF_predict(crfDic['pf/asso'],X1_ch7)
                    prob_All_7=zip(prob_ch7_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_7,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/asymmetry":
                    phrase8=phrase[beg:end+1]
                    X1_ch8,tokenList_ch8=CRF_featureCreationTest(phrase8)
                    predicted1_ch8_asso,prob_ch8_asso=CRF_predict(crfDic['pf/asso'],X1_ch8)
                    predicted1_ch8_Si,prob_ch8_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch8)
                    prob_All_8=zip(prob_ch8_asso,prob_ch8_Si,preLabelLoc[beg:end+1])
                    print prob_All_8
                    preLabels=max_label_cal(prob_All_8,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/mass":
                    phrase9=phrase[beg:end+1]
                    X1_ch9,tokenList_ch9=CRF_featureCreationTest(phrase9)
                    predicted1_ch9_Mar,prob_ch9_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch9)
                    prob_All_9=zip(prob_ch9_Mar,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_9,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/calcification":
                    phrase4=phrase[beg:end+1]
                    X1_ch4,tokenList_ch4=CRF_featureCreationTest(phrase4)
                    predicted1_ch4_MoDi,prob_ch4_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch4)
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
                    
                if child=="report/negative_finding":
                    phrase10=phrase[beg:end+1]
                    X1_ch10,tokenList_ch10=CRF_featureCreationTest(phrase10)
                    predicted1_ch10_loc,prob_ch10_loc=CRF_predict(crfDic['pfnf'],X1_ch10)
                    z2=0
                    for i1 in range(beg,end+1):
                        preLabelLoc[i1]=prob_ch10_loc[z2]
                        z2=z2+1
                    test_onPredicted(beg,end+1,phrase,preLabels,'negative_finding',child,preLabelLoc)
                    
                if child=="report/positive_finding/mass":
                    phrase5=phrase[beg:end+1]
                    X1_ch5,tokenList_ch5=CRF_featureCreationTest(phrase5)
                    predicted1_ch5_ShDen,prob_ch5_ShDen=CRF_predict(crfDic['pf/mass'],X1_ch5)
                    predicted1_ch5_Si,prob_ch5_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch5)
                    predicted1_ch5_Mar,prob_ch5_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch5)
                    predicted1_ch5_asso,prob_ch5_asso=CRF_predict(crfDic['pf/asso'],X1_ch5)
                    prob_All=zip(prob_ch5_ShDen,prob_ch5_Si,prob_ch5_Mar,prob_ch5_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/calcification":
                    phrase6=phrase[beg:end+1]
                    X1_ch6,tokenList_ch6=CRF_featureCreationTest(phrase6)
                    predicted1_ch6_Si,prob_ch6_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch6)
                    predicted1_ch6_MoDi,prob_ch6_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch6)
                    predicted1_ch6_asso,prob_ch6_asso=CRF_predict(crfDic['pf/asso'],X1_ch6)
                    prob_All_6=zip(prob_ch6_Si,prob_ch6_MoDi,prob_ch6_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_6,preLabels,beg,end,preLabelLoc)
                
                if child=="report/positive_finding/architectural_distortion":
                    phrase7=phrase[beg:end+1]
                    X1_ch7,tokenList_ch7=CRF_featureCreationTest(phrase7)
                    predicted1_ch7_asso,prob_ch7_asso=CRF_predict(crfDic['pf/asso'],X1_ch7)
                    prob_All_7=zip(prob_ch7_asso,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_7,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/positive_finding/asymmetry":
                    phrase8=phrase[beg:end+1]
                    X1_ch8,tokenList_ch8=CRF_featureCreationTest(phrase8)
                    predicted1_ch8_asso,prob_ch8_asso=CRF_predict(crfDic['pf/asso'],X1_ch8)
                    predicted1_ch8_Si,prob_ch8_Si=CRF_predict(crfDic['pf/mass_calc_asym'],X1_ch8)
                    prob_All_8=zip(prob_ch8_asso,prob_ch8_Si,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_8,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/mass":
                    phrase9=phrase[beg:end+1]
                    X1_ch9,tokenList_ch9=CRF_featureCreationTest(phrase9)
                    predicted1_ch9_Mar,prob_ch9_Mar=CRF_predict(crfDic['pfnf/mass'],X1_ch9)
                    prob_All_9=zip(prob_ch9_Mar,preLabelLoc[beg:end+1])
                    preLabels=max_label_cal(prob_All_9,preLabels,beg,end,preLabelLoc)
                    
                if child=="report/negative_finding/calcification":
                    phrase4=phrase[beg:end+1]
                    X1_ch4,tokenList_ch4=CRF_featureCreationTest(phrase4)
                    predicted1_ch4_MoDi,prob_ch4_MoDi=CRF_predict(crfDic['pfnf/calc'],X1_ch4)
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

def mainFunc(path_to_prediction_report):
    tree_all = Elem.parse(path_to_prediction_report)
    list_tree=tree_all.findall('report')    
    root=Element('radiology_reports')
    for list_tree_elem in list_tree:
        root.append(list_tree_elem)
    print "No.of reports:",len(root)
    global tree
    tree=ElementTree(root)
    
    dicListPre={}
    dicListTrue={}
    global crfDic
    
    #load models
    models_path="CRFmodelB_trainedmodel.pkl"
    models_file=open(models_path,'rb')
    crfDic=pickle.load(models_file)
    
    train_test_onTrue("report")
    
    cascadedOnPre=[]
    locLists=[]
    for i in range(len(tokenListAll)):
        cascadedOnPre.append([])
        locLists.append([])
        for j in range(len(tokenListAll[i])):
            cascadedOnPre[i].append(['',''])
            locLists[i].append('')    
    
    for i in range(len(tokenListAll)):
        preLabels=list(cascadedOnPre[i])
        phrase1=list(tokenListAll[i])
        locList=list(locLists[i])
        test_onPredicted(0,len(tokenListAll[i]),phrase1,preLabels,"report","report",locList)
        for j in range(len(cascadedOnPre[i])):
            cascadedOnPre[i][j]=preLabels[j]
    
    cascPreList=[]
    cascPreList1=[]

    predictPredictList1=[]
    tokenFor2levelList1=[]
    
    predictpredictPredictList1=[]
    tokenFor3levelList1=[]
    
    count=0
    for i in range(len(tokenListAll)):
        cascPreVal=[]
        cascPreVal1=[]
        predictPredict1=[]
        tokenFor2level=[]
        predictpredictPredict1=[]
        tokenFor3level=[]
        for j in range(len(tokenListAll[i])):
            cascadedOnPre[i][j][0]=cascadedOnPre[i][j][0].strip('/')
            cascadedOnPre[i][j][1]=cascadedOnPre[i][j][1].strip('/')
            cascPre=cascadedOnPre[i][j][0].split('/')
            if len(cascPre)>=2:
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

#example of how to call the function mainFunc
path='./../data/testSample_input.xml' #path to the xml file which contains the report to be labeled
mainFunc(path)
