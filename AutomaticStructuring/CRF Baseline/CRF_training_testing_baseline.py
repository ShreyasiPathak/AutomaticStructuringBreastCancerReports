import re
import sys
import nltk
import string
import math
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
import confusionmatrix_heatmap


def token_LabelCreation(reports):
    docs=[]
    labelList=[]
    report_count=0
    #label_report_count={}
    for report in reports:
    #print report
        report=report.strip('[(').strip('\n').strip(')]')
        lines=report.split('), (')
        EachReport=[]
        report_count=report_count+1
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
            if not label_count_report.has_key(label):
                label_count_report[label]=[1,report_count]
            else:
                if report_count!=label_count_report[label][1]:
                    label_count_report[label][0]=label_count_report[label][0]+1
                    label_count_report[label][1]=report_count
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
print len(list_tree)
k=len(list_tree)/4
label_dic_all={}
label_dic_2={}
out=open('CRF_baseline_file.txt','a')
out1=open('CRF_baseline_featurefile.txt','a')
out2=open('CRF_baseline_predictedvsTrue.txt','a')
label_dic_2_pre={}
conf_mat_agg=np.zeros((34,34))
root=Element('radiology_reports')
for list_tree_elem in list_tree:
        root.append(list_tree_elem)
out4=open('file_test','w')
EachReport1=[]
parsingReportsXML_baseline.print_path_of_elems(out4,EachReport1, root, root.tag)
out4.close()
f=open('file_test','r')
reports_te=f.readlines()
label_count_report={}
docs1_te=token_LabelCreation(reports_te)
print label_count_report
'''for i in range(0,4):
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
    #roott=tree.getroot()
    tree1=ElementTree(root1)
    #roott1=tree1.getroot()
    out3=open('file_train_'+str(i),'w')
    out4=open('file_test_'+str(i),'w')
    EachReport1=[]
    EachReport2=[]
    parsingReportsXML_baseline.print_path_of_elems(out3,EachReport1, root, root.tag)
    parsingReportsXML_baseline.print_path_of_elems(out4,EachReport2, root1, root1.tag)
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
        
    CRF_measures_baseline.tokenLevel_measures(predicted2,Y1,tokenList_te,label_dic_all,conf_mat_agg)
    predictPredictList1=[]
    tokenFor2levelList1=[]
    trueLabel_2labelsList1=[]
    predictList1=[]
    tokenFor1levelList1=[]
    trueLabel_1labelsList1=[]
    nfc=0
    for j in range(len(tokenList_te)):
        predictPredict1=[]
        tokenFor2level=[]
        trueLabel_2labels1=[]
        predict1=[]
        tokenFor1level=[]
        trueLabel_1labels1=[]
        for m in range(len(tokenList_te[j])):
            #print predicted2[j][k]
            #print Y1[j][k]
            pred1=predicted2[j][m].split('/')
            true1=Y1[j][m].split('/')
            if true1=="B-negative_finding" or true1=="I-negative_finding":
                nfc=nfc+1
            #Level2
            if len(pred1)>=2 or len(true1)>=2 or true1[0].split('-')[len(true1[0].split('-'))-1]=='negative_finding' or pred1[0].split('-')[len(pred1[0].split('-'))-1]=='negative_finding': 
                predictPredict="/".join(pred1[:2])
                trueTrue="/".join(true1[:2])
                tokenFor2level.append(tokenList_te[j][m])
                predictPredict1.append(predictPredict)
                trueLabel_2labels1.append(trueTrue)
            #Level1
            predictL="/".join(pred1[:1])
            trueL="/".join(true1[:1])
            tokenFor1level.append(tokenList_te[j][m])
            predict1.append(predictL)
            trueLabel_1labels1.append(trueL)
        #Level2
        predictPredictList1.append(predictPredict1)
        trueLabel_2labelsList1.append(trueLabel_2labels1)
        tokenFor2levelList1.append(tokenFor2level)
        #Level1
        predictList1.append(predict1)
        trueLabel_1labelsList1.append(trueLabel_1labels1)
        tokenFor1levelList1.append(tokenFor1level)
    #CRF_measures_baseline.tokenLevel_measures(predictPredictList1,trueLabel_2labelsList1,tokenFor2levelList1,label_dic_2_pre)
    #CRF_measures_baseline.tokenLevel_measures(predictList1,trueLabel_1labelsList1,tokenFor1levelList1,label_dic_2_pre)
    print nfc
    f_train.close()
    f_test.close()'''
    #partial_phrase_dic,complete_phrase_dic=CRF_measures_baseline.partialPhraseLevel_measures(tokenList_te,predicted1,Y_te)
    #print "Label\tTokenPrecision,recall,fmeasure,support\tPartialPhraseAccuracy\tCompletePhraseAccuracy"
    #for key in token_dic.iterkeys():
    #    total_measure_dic[key]=[token_dic[key],partial_phrase_dic[key],complete_phrase_dic[key]]
    #    print key,"\t",token_dic[key],"\t", partial_phrase_dic[key],"\t",complete_phrase_dic[key]'''
    
'''label_dic2_fscore={}
label_dic2_support={}
for key in label_dic_all.iterkeys():
    label_dic2_fscore[key]=float(sum(label_dic_all[key][0]))/len(label_dic_all[key][0])
    label_dic2_support[key]=label_dic_all[key][1][0]
print label_dic2_fscore
print label_dic2_support'''
label_dic_abb={'O':'O','breast_composition':'BC','positive_finding/mass/location':'PF/MS/L','positive_finding/mass/size':'PF/MS/SI','positive_finding/mass/margin':'PF/MS/MA','positive_finding/mass/density':'PF/MS/DE','positive_finding/mass/associated_features':'PF/MS/AF','positive_finding/mass/shape':'PF/MS/SH','positive_finding/mass':'PF/MS/O','positive_finding/calcification/location':'PF/C/L',\
               'positive_finding/calcification/size':'PF/C/SI','positive_finding/calcification/morphology':'PF/C/MO','positive_finding/calcification/distribution':'PF/C/DI','positive_finding/calcification/associated_features':'PF/C/AF','positive_finding/calcification':'PF/C/O','positive_finding/architectural_distortion/location':'PF/AD/L','positive_finding/architectural_distortion/associated_features':'PF/AD/AF',\
               'positive_finding/architectural_distortion':'PF/AD/O','positive_finding/associated_features/location':'PF/AF/L','positive_finding/associated_features':'PF/AF/O','positive_finding/asymmetry/location':'PF/AS/L','positive_finding/asymmetry/size':'PF/AS/SI','positive_finding/asymmetry/associated_features':'PF/AS/AF','positive_finding/asymmetry':'PF/AS/O','negative_finding/mass/location':'NF/MS/L',\
               'negative_finding/mass/margin':'NF/MS/MA','negative_finding/mass':'NF/MS/O','negative_finding/calcification/location':'NF/C/L','negative_finding/calcification/morphology':'NF/C/MO','negative_finding/calcification/distribution':'NF/C/DI','negative_finding/calcification':'NF/C/O','negative_finding/architectural_distortion/location':'NF/AD/L','negative_finding/architectural_distortion':'NF/AD/O',\
               'negative_finding/associated_features/location':'NF/AF/L','negative_finding/associated_features':'NF/AF/O','negative_finding/asymmetry/location':'NF/AS/L','negative_finding/asymmetry':'NF/AS/O','negative_finding/location':'NF/L','negative_finding':'NF/O'}
label_dic1={}
for key in label_dic_all.iterkeys():
    label_dic1[label_dic_abb[key]]=label_dic_all[key][2][0]
sns.set()
#df_data=pd.DataFrame(label_dic1)
print label_dic1
axis_labels=sorted(label_dic1,key=label_dic1.__getitem__)
conf_mat_agg=conf_mat_agg.astype(int)
#print conf_mat_agg
conf_mat_agg_norm=(np.zeros((34,34))).astype('float')
for i in range(len(conf_mat_agg)):
    s=np.sum(conf_mat_agg[i,:])
    for j in range(len(conf_mat_agg[i])):
        conf_mat_agg_norm[i,j]=float(conf_mat_agg[i,j])/s

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
f.savefig("ConfusionMatrixHeatmap_baseline.pdf",bbox_inches='tight')
'''boundaries = [0.0, 0.001, 0.003, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]  # custom boundaries
# here I generated twice as many colors, 
# so that I could prune the boundaries more clearly
hex_colors = sns.light_palette('navy', n_colors=len(boundaries) * 2 + 2, as_cmap=False).as_hex()
hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]

colors=list(zip(boundaries, hex_colors))

custom_color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
    name='custom_navy',
    colors=colors,
)
sns.heatmap(
        vmin=0,
        vmax=500,
        data=conf_mat_agg,
        cmap=custom_color_map,
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        linewidths=0.75,
)'''

'''conf_mat_agg_df=pd.DataFrame(conf_mat_agg,index=axis_labels,columns=axis_labels)
corr=conf_mat_agg_df.corr()
sns.heatmap(corr,cmap="magma_r")'''
'''def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb =matplotlib.colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict

#hc = ['#ffe5e5','#E6B3B3','#ffb2b2','#ff9999','#ff7f7f','#ff6666','#ff4c4c','#ff3232', '#ff1919','#acacdf', '#7272bf', '#39399f', '#000080'] #'#e5e5ff'
#th = [0, 0.001, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 1.0]
hc = ['#ffe5e5','#ff9999','#ff7f7f','#ff6666','#ff4c4c','#ff3232', '#ff1919', '#ff0000','#991900'] #'#e5e5ff'
th = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5, 0.6, 1.0]
cdict = NonLinCdict(th, hc)
cm = matplotlib.colors.LinearSegmentedColormap('test', cdict)

#plt.figure()
sns.heatmap(
        yticklabels=axis_labels,
        xticklabels=axis_labels,
        vmin=0,
        vmax=1374,
        data=conf_mat_agg,
        cmap=cm,
        annot=True,
        fmt="d",
        linewidths=0.75)'''

#ax=sns.heatmap(conf_mat_agg,yticklabels=axis_labels,xticklabels=axis_labels,annot=True,linewidths=0.75,fmt="d",cmap="rainbow")#cmap=sns.diverging_palette(220,30,as_cmap=True))#cmap="YlGnBu")
#fig=confusionmatrix_heatmap.print_confusion_matrix(conf_mat_agg,axis_labels)
#plt.show()
label_dic2_fscore={}
label_dic2_support={}
for key in label_dic_2_pre.iterkeys():
    label_dic2_fscore[key]=float(sum(label_dic_2_pre[key][0]))/len(label_dic_2_pre[key][0])
    label_dic2_support[key]=label_dic_2_pre[key][1][0]
print label_dic2_fscore
print label_dic2_support