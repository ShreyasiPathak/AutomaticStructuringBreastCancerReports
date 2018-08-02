import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopwordsList=set(stopwords.words('dutch'))
stemmer=SnowballStemmer("dutch")
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    #if '#NUM' in sent:
    #    digitInSeq=1
    #else:
    #    digitInSeq=0
    if word.lower() in stopwordsList:
        wrdStpwrd=1
    else:
        wrdStpwrd=0
    features = {
        #'bias': 1.0,
        'word.lower()': word.lower(),
        'word.stem()': stemmer.stem(word.lower()),
        'word.ispunctuation': 1 if word.lower() in string.punctuation else 0,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        #'word.isdigit()': word.isdigit(),
        'word.isstopword()': wrdStpwrd,
        'postag': postag,
        'postag[:2]': postag[:2],
        #'digitInSeq': digitInSeq
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        if word1.lower() in stopwordsList:
            wrdStpwrd1=1
        else:
            wrdStpwrd1=0
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.stem()': stemmer.stem(word1.lower()),
            #'-1:word.nopunctuation()': word1.lower().translate(None, string.punctuation),
            '-1:word.ispunctuation': 1 if word1.lower() in string.punctuation else 0,
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isstopword()': wrdStpwrd1,
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word2 = sent[i+1][0]
        postag2 = sent[i+1][1]
        if word2.lower() in stopwordsList:
            wrdStpwrd2=1
        else:
            wrdStpwrd2=0
        features.update({
            '+1:word.lower()': word2.lower(),
            '+1:word.stem()': stemmer.stem(word2.lower()),
            #'+1:word.nopunctuation()': word1.lower().translate(None, string.punctuation),
            '+1:word.ispunctuation': 1 if word2.lower() in string.punctuation else 0,
            '+1:word[-3:]': word2[-3:],
            '+1:word[-2:]': word2[-2:],
            '+1:word.istitle()': word2.istitle(),
            '+1:word.isupper()': word2.isupper(),
            '+1:word.isstopword()': wrdStpwrd2,
            '+1:postag': postag2,
            '+1:postag[:2]': postag2[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def posTagAdding(docs):
    data=[]
    tokensList=[]
    for i, doc in enumerate(docs):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]
        tokensList.append(tokens)
    
        # Perform POS tagging
        tagged = nltk.pos_tag(tokens)
    
        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return tokensList,data

def posTagAddingTest(docs):
    data=[]
    tokensList=[]
    tagged = nltk.pos_tag(docs)
    '''for i, doc in enumerate(docs):
        # Obtain the list of tokens in the document
        tokensList.append(doc)
    
        # Perform POS tagging
        tagged = nltk.pos_tag(tokens)
    
        # Take the word, POS tag, and its label
        data.append([(word, pos) for (word, pos) in tagged])'''
    return docs,tagged