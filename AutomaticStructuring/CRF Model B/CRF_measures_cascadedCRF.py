import string
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.utils import flatten
from sklearn.metrics import precision_recall_fscore_support

def tokenLevel_measures(predictedY,trueY,tokenList,label_dic):
    dic_tokenmeasure={}
    predictedYflat1=[]
    trueYflat1=[]
    predictedYflat=flatten(predictedY)
    trueYflat=flatten(trueY)
    tokenListFlat=flatten(tokenList)
    #out=open("confusion matrix.txt",'w')
    for i in range(len(predictedYflat)):
        if tokenListFlat[i] not in string.punctuation:
            labelSplitPre=predictedYflat[i].split('-')
            predictedYflat1.append(labelSplitPre[len(labelSplitPre)-1])
            labelSplitTrue=trueYflat[i].split('-')
            trueYflat1.append(labelSplitTrue[len(labelSplitTrue)-1])
        
    labels1=list(np.unique(trueYflat1))
    #labels1.remove('O')
    labels2=list(np.unique(trueYflat1))
    #labels2.remove('O')
    measuresprs=precision_recall_fscore_support(trueYflat1,predictedYflat1,labels=labels1)
    #print measuresprs[0]
    for i in range(len(labels2)):
        #if measuresprs[3][i]<=10:
        #    labels1.remove(labels2[i])
        if not label_dic.has_key(labels2[i]):
            label_dic[labels2[i]]=[[],[0]]
        label_dic[labels2[i]][0].append(measuresprs[2][i])
        label_dic[labels2[i]][1]=label_dic[labels2[i]][1]+measuresprs[3][i]
            #measuresprs1=zip(labels1,measuresprs[0],measuresprs[1],measuresprs[2],measuresprs[3])
    
    #print trueYflat
    #print predictedYflat
    #print labels1
    #labels1 = ['B-address', 'B-authors', 'B-booktitle', 'B-journal', 'B-pages', 'B-publisher', 'B-ref', 'B-title', 'B-volume', 'B-year', 'I-address', 'I-authors', 'I-booktitle', 'I-journal', 'I-pages', 'I-publisher', 'I-ref', 'I-title', 'I-volume', 'I-year']
    #sorted_labels = sorted(labels1,key=lambda name: (name[1:], name[0]))
    #F1score_micro=metrics.flat_f1_score(trueY, predictedY, labels=labels1, average='micro')
    F1score_micro=f1_score(trueYflat1, predictedYflat1, labels=labels1, average='micro')
    print "F1 measure:", F1score_micro
    precision_micro=precision_score(trueYflat1, predictedYflat1, labels=labels1, average='micro')
    print "precision measure:", precision_micro
    recallScore_micro=recall_score(trueYflat1, predictedYflat1, labels=labels1, average='micro')
    print "recall measure:", recallScore_micro
    #precisionScore_micro=metrics.flat_precision_score(trueY, predictedY, average='micro')
    #print precisionScore_micro
    #recallScore_micro=metrics.flat_recall_score(trueY, predictedY, average='micro')
    #print recallScore_micro
    classificationReport=classification_report(trueYflat1, predictedYflat1, labels=labels1, digits=3)
    print classificationReport
    conf_mat=confusion_matrix(trueYflat1,predictedYflat1, labels=labels1)
    for i in range(len(labels1)):
        dic_tokenmeasure[labels1[i]]=[measuresprs[0][i],measuresprs[1][i],measuresprs[2][i],measuresprs[3][i]]
    #out.write(str(labels1)+"\n")
    #for i in range(len(conf_mat)):
    #    out.write(str(conf_mat[i])+"\n")
    #print np.sum(conf_mat[1,:])
    accuracy=accuracy_score(trueYflat1, predictedYflat1)
    #print measuresprs1
    print "Accuracy:", accuracy
    return dic_tokenmeasure

def partialPhraseLevel_measures(tokensList,predictedY,trueY):
    phraseList=[]
    out=open('phraseList_CRF','w')
    for data1,pre1,true1 in zip(tokensList,predictedY,trueY):
        #print data1,"=>", pre1
        phrase=[[],[],[]]
        for j in range(len(true1)):
            labelStartTagPres=true1[j].split('-')[0]
            if j!=len(true1)-1:
                labelStartTagNext=true1[j+1].split('-')[0]
            else:
                labelStartTagNext=None
            if labelStartTagPres=='B':
                if data1[j] not in string.punctuation:
                    pre1List=pre1[j].split('-')
                    phrase[0].append(data1[j])
                    phrase[1].append(true1[j].split('-')[1])
                    phrase[2].append(pre1List[len(pre1List)-1])
                    if labelStartTagNext=='B' or labelStartTagNext=='O' or j==len(true1)-1:
                        phraseList.append(phrase)
                        phrase=[[],[],[]]
            elif labelStartTagPres=='I':
                if data1[j] not in string.punctuation:
                    pre1List=pre1[j].split('-')
                    phrase[0].append(data1[j])
                    phrase[1].append(true1[j].split('-')[1])
                    phrase[2].append(pre1List[len(pre1List)-1])
                if labelStartTagNext=='B' or labelStartTagNext=='O' or j==len(true1)-1:
                    phraseList.append(phrase)
                    phrase=[[],[],[]]
                    
        '''for i in range(0,len(data1)):
            out2.write(str(data1[i])+"\t"+str(pre1[i])+"\t"+str(true1[i])+"\n")
            #print data1[i],"\t",pre1[i],"\t",true1[i]'''
    out.write(str(phraseList))
    par_acc_dic,comp_acc_dic=partialPhraseLevel_MainStep(phraseList)
    return par_acc_dic,comp_acc_dic

def partialPhraseLevel_MainStep(phraseList):
    similarity_dic={}
    partial_accuracy_dic={}
    complete_accuracy_dic={}
    for phrase in phraseList:
        '''if len(np.unique(phrase[1]))==1:
            print "passed"
        else:
            print "problem"
            print phrase[0],phrase[1]'''
        Truelabel=phrase[1][0]
        true_pred=zip(phrase[1],phrase[2])
        intersect=true_pred.count((Truelabel,Truelabel))
        dice_coeff=float(2*intersect)/(len(phrase[1])+len(phrase[2]))
        
        if not similarity_dic.has_key(Truelabel):
            similarity_dic[Truelabel]=[dice_coeff]
            partial_accuracy_dic[Truelabel]=0
            complete_accuracy_dic[Truelabel]=0
        else:
            similarity_dic.get(Truelabel).append(dice_coeff)
        if dice_coeff>=0.75:
            partial_accuracy_dic[Truelabel]=partial_accuracy_dic.get(Truelabel)+1
        if dice_coeff==1.0:
            complete_accuracy_dic[Truelabel]=complete_accuracy_dic.get(Truelabel)+1
    for key in similarity_dic.iterkeys():
        #partial_accuracy_dic[key]=[accuracy_dic[key],len(similarity_dic[key]),float(accuracy_dic[key])/len(similarity_dic[key])]
        partial_accuracy_dic[key]=float(partial_accuracy_dic[key])/len(similarity_dic[key])
        complete_accuracy_dic[key]=float(complete_accuracy_dic[key])/len(similarity_dic[key])
    #print similarity_dic
    #print accuracy_dic
    #for key in partial_accuracy_dic.iterkeys():
    #    print key, partial_accuracy_dic[key]
    return partial_accuracy_dic, complete_accuracy_dic