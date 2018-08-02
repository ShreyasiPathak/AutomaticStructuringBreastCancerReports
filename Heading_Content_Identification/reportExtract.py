import csv
import re
import pandas
import sys

reload(sys)
sys.setdefaultencoding('utf8')

AllReport=[]
AllSentence=[]
ct=0
ctList=[]

df = pandas.read_excel('birads-2012-tm-2017.xlsx',None)
#print the column names
print df['birads-2016'].columns
#get the values for a given column
AllReport.append(df['birads-2012-tm-2017']['verslag'].values[:30])
AllReport.append(df['birads-2016']['verslag'].values[:30])
AllReport.append(df['birads-2015']['verslag'].values[:30])
AllReport.append(df['birads-2014']['verslag'].values[:30])
AllReport.append(df['birads-2013']['verslag'].values[:30])
AllReport.append(df['birads-2012']['verslag'].values[:30])
#get a data frame with selected columns
#FORMAT = ['verslag', 'onderzdat']
#df_selected = df[FORMAT]
#print df_selected

f1=open('Sentence180_H_NH.csv','w')
f1Write=csv.writer(f1)
f1Write.writerow(['Sentence','Heading'])

for eachYear in AllReport:
    for report in eachYear:
        line1=re.sub(r'_x000D_','',report)
        AllSentence.append(line1.split('\n'))

for sentenceList in AllSentence:
    ct=ct+1
    for eachSent in sentenceList:
        #print li,'\n'
        eachSent=eachSent.strip('\r')
        if eachSent!='':
            sent=[eachSent]
            print sent
            f1Write.writerow(sent)
            '''try:
                f1Write.writerow(sent)
            except (RuntimeError, TypeError, UnicodeError, NameError):
                if ct not in ctList:
                    ctList.append(ct)
                pass'''
#print ctList