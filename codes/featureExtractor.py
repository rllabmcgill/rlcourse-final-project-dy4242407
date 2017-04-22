# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:53:31 2017

@author: user
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize

def getPosition(sentence,documents):
    positionScore=0
    for document in documents:
        listOfSentencesInDocument=document.split('.')[0:-1]
        if(sentence in listOfSentencesInDocument):
            positionScore=(1.0/(listOfSentencesInDocument.index(sentence)+1))
        elif (sentence.lstrip() in listOfSentencesInDocument):
            positionScore=(1.0/(listOfSentencesInDocument.index(sentence.lstrip())+1))
    return positionScore



def getPositionSum(summary,documents):
    positionScore=0
    listOfSentencesInSummary=summary.split('.')[0:-1]
    for sentence in listOfSentencesInSummary:
        for document in documents:
            listOfSentencesInDocument=document.split('.')[0:-1]
            if(sentence in listOfSentencesInDocument):
                positionScore+=(1.0/(listOfSentencesInDocument.index(sentence)+1))
            elif (sentence.lstrip() in listOfSentencesInDocument):
                positionScore+=(1.0/(listOfSentencesInDocument.index(sentence.lstrip())+1))
    return positionScore
            
    
    

def getCoverageRatio(summary,top_features):
    count=0
    wordsOfSummary=word_tokenize(summary)
    for word in wordsOfSummary:
        if word in top_features:
            count+=1
            
    return count
    
def getRedundancyRatio(summary,top_features,threshhold):
    count=np.zeros(len(top_features))
    wordsOfSummary=word_tokenize(summary)
    redundancyRatio=0
    for word in wordsOfSummary:
        if word in top_features:
            count[top_features.index(word)]+=1
                  
    redundancyRatio=len(np.argwhere(count>threshhold))
    return redundancyRatio
    
def getCoverage(summary,top_features):
    coverage=np.zeros(len(top_features))
    wordsOfSummary=word_tokenize(summary)
    for word in wordsOfSummary:
        if word in top_features:
                coverage[top_features.index(word)]=1
    return coverage

def getLengthRatio(summary,K):
    return (len(summary.split('.'))-1)/K

#%%

document1 = "Python is a 2000 made-for-TV horror movie directed by Richard Clabaugh. The film features several cult favorite actors, including William Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy, Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean Whalen. The film concerns a genetically engineered snake, a python, that escapes and unleashes itself on a small town. It includes the classic final girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles, California and Malibu, California. Python was followed by two sequels: Python II (2002) and Boa vs. Python (2004), both also made-for-TV films."

document2 = "Python, from the Greek word, is a genus of nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are recognised. A member of this genus, P reticulatus, is among the longest snakes known."

document3 = "The Colt Python is a 357 Magnum caliber revolver formerly manufactured by Colt's Manufacturing Company of Hartford, Connecticut. It is sometimes referred to as a Combat Magnum.[1] It was first introduced in 1955, the same year a Smith &amp; Wesson's M29 44 Magnum. The now discontinued Colt Python targeted the premium revolver market segment. Some firearm collectors and writers such as Jeff Cooper, Ian V Hogg, Chuck Hawks, Leroy Thompson, Renee Smeets and Martin Dougherty have described the Python as the finest production revolver ever made."

summary= "Python is a 2000 made-for-TV horror movie directed by Richard Clabaugh. The Colt Python is a 357 Magnum caliber revolver formerly manufactured by Colt's Manufacturing Company of Hartford, Connecticut. A member of this genus, P reticulatus, is among the longest snakes known."

documents = [document1, document2, document3]
    
#lectures = ["Police are hunting a killer who shared a video of the moment he shot dead an innocent man in Cleveland and claims to have slaughtered 14 more.", "Steve Stephens, 37, is on the loose in the Ohio city after he filmed the murder and posted it on social media at around 2pm Eastern Time on Easter Sunday.", "An aggravated murder warrant was issued for his arrest late Sunday night, stating residents in the states of Pennsylvania, New York, Indiana and Michigan as well as Ohio should be on high alert."," The horrifying video, which police say is real, shows him driving up to an elderly man, getting out of his car, and opening fire."]
#%%


def getFeatureVector(documents,summary):
    if summary ==[]:
        return np.zeros(104)
    #documents = re.sub(‘[^A-Za-z .-]+’, ‘ ‘, documents)
    #documents = documents.replace(‘-’, ‘’)
    #documents = documents.replace(‘…’, ‘’)
    #documents = documents.replace(‘Mr.’, ‘Mr’).replace(‘Mrs.’, ‘Mrs’)
    
    else:
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
        X = vectorizer.fit_transform(documents)
        indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()
        top_n = 100
        top_features = [features[i] for i in indices[:top_n]]
        #print(top_features)
        coverage=getCoverage(summary,top_features)
        coverageRatio=getCoverageRatio(summary,top_features)
        redundancyRatio=getRedundancyRatio(summary,top_features,5)
        lengthRatio=getLengthRatio(summary,10.0)
        positionSum=getPositionSum(summary,documents)
        featureVector=np.zeros(104)
        featureVector[0:100]=coverage
        featureVector[100]=coverageRatio
        featureVector[101]=redundancyRatio
        featureVector[102]=lengthRatio
        featureVector[103]=positionSum
    return featureVector
    
    

featureVector=getFeatureVector(documents,summary)



#lectures = ["this is some food", "this is some drink"]
#vectorizer = TfidfVectorizer(ngram_range=(1,2))
#X = vectorizer.fit_transform(lectures)
#features_by_gram = defaultdict(list)
#for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
#    features_by_gram[len(f.split(' '))].append((f, w))
#top_n = 2
#for gram, features in features_by_gram.iteritems():
#    top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
#    top_features = [f[0] for f in top_features]
#    print '{}-gram top:'.format(gram), top_features



