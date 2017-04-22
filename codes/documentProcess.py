# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:09:54 2017

@author: user
"""

import glob
import nltk
import nltk.data
import re
import pandas as pd
#from pyrouge import Rouge155
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


goldStandardSummaries= sorted(glob.glob("/home/rldata/aemami1/DUCTask2-50DocumentClusters-WithGoldStandard/GoldStandard/*.txt"))
goldStandardSummaryIndex=0
for clustername in sorted(glob.glob("/home/rldata/aemami1/DUCTask2-50DocumentClusters-WithGoldStandard/docs/*")):  
    clusterID=clustername[-6:-1]
    #print clustername
    documents=[0]*10
    sentencesInDocuments=[0]*10
    numSentences=[0]*10
    index=0
    
    
    for filename in glob.glob(clustername+'/*.txt'):
    
        file = open(filename, "r") 
        lines=[lines.rstrip('\n') for lines in file]
        lines=lines[5:-2]
        document=''.join(lines)
        document=document.lower()
        sentencesInDocument=nltk.sent_tokenize(document)
        sentencesInDocuments[index]=sentencesInDocument
        documents[index]=document
        numSentences[index]=len(sentencesInDocument)
        index+=1
    
        
        
    #not super pythonic, no, not at all.
    #use extend so it's a big flat list of vocab
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in documents:
        allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
        
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    #print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'
    # -*- coding: utf-8 -*-
    """
    Created on Tue Apr 18 14:53:31 2017
    
    @author: user
    """
    
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import numpy as np
    from collections import defaultdict
    from nltk.tokenize import word_tokenize
    
    
    def getPositionSum(summary,documents):
        positionScore=0
        listOfSentencesInSummary=nltk.sent_tokenize(summary)
        for sentence in listOfSentencesInSummary:
            for document in documents:
                listOfSentencesInDocument=nltk.sent_tokenize(document)
                if(sentence in listOfSentencesInDocument):
                    positionScore+=(1.0/(listOfSentencesInDocument.index(sentence)+1))
                    break
                elif (sentence.lstrip() in listOfSentencesInDocument):
                    positionScore+=(1.0/(listOfSentencesInDocument.index(sentence.lstrip())+1))
                    break
        return positionScore
                
        
        
    
    def getCoverageRatio(coverage,n):
        coverageRatio=sum(coverage)/n
                
        return coverageRatio
        
    def getRedundancyRatio(summary,top_features,threshhold):
        count=np.zeros(len(top_features))
        wordsOfSummary=word_tokenize(summary)
        redundancyRatio=0
        for word in wordsOfSummary:
            if word in top_features:
                count[top_features.index(word)]+=1
                                  
        redundancyRatio=len(np.argwhere(count>threshhold))/100.0
        return redundancyRatio
        
    def getCoverage(summary,top_features):
        coverage=np.zeros(len(top_features))
        wordsOfSummary=word_tokenize(summary)
        for word in wordsOfSummary:
            if word in top_features:
                    coverage[top_features.index(word)]=1
        return coverage
    
    def getLengthRatio(summary,K):
        #return (len(summary.split('.'))-1)/K
        return (len(summary.split(' '))/K)
    
    
    #summary= "The Swiss government has ordered no investigation of possible bank accounts belonging to former Chilean dictator Augusto Pinochet, a spokesman said Wednesday. Weekend newspaper reports in Spain said a Spanish judge who ordered Pinochet's arrest has issued a petition aimed at freezing any accounts the 82-year-old general might have in Luxembourg and Switzerland. But government spokesman Achille Casanova said no accounts have so far been frozen in Switzerland and no investigation order has been given to federal banking authorities. Pinochet has been held at a London clinic since his arrest earlier this month."
    summary=' '.join([sentencesInDocuments[0][0],sentencesInDocuments[1][0],sentencesInDocuments[2][0]])
   
    def top_tfidf_feats(row, features, top_n=100):
        ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return df
        
    def top_feats_in_doc(Xtr, features, row_id, top_n=100):
        row = np.squeeze(Xtr[row_id].toarray())
        return top_tfidf_feats(row, features, top_n)
        
    def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=100):
        ''' Return the top n features that on average are most important amongst documents in rows
            indentified by indices in grp_ids. '''
        if grp_ids:
            D = Xtr[grp_ids].toarray()
        else:
            D = Xtr.toarray()
    
        D[D < min_tfidf] = 0
        tfidf_means = np.mean(D, axis=0)
        return top_tfidf_feats(tfidf_means, features, top_n)
        
        
    
    def getFeatureVector(documents,summary):
        vectorizer = TfidfVectorizer(analyzer='word', tokenizer=tokenize_only, ngram_range=(1,1), min_df = 0, stop_words = 'english')
        X = vectorizer.fit_transform(documents)
        
        #indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()
        #top_n = 100
        #top_features = [features[i] for i in indices[:top_n]]
        top_features=top_mean_feats(X,features)
        top_features2=[0]*100
        index=0
        for feature in top_features.feature:
            top_features2[index]=feature
            index+=1
    
        coverage=getCoverage(summary,top_features2)
        coverageRatio=getCoverageRatio(coverage,100)
        redundancyRatio=getRedundancyRatio(summary,top_features2,1)
        lengthRatio=getLengthRatio(summary,250.0)
        positionSum=getPositionSum(summary,documents)
        featureVector=np.zeros(104)
        featureVector[0:100]=coverage
        featureVector[100]=coverageRatio
        featureVector[101]=redundancyRatio
        featureVector[102]=lengthRatio
        featureVector[103]=positionSum
        return top_features2,featureVector
        
        
    
    top_features,featureVector=getFeatureVector(documents,summary.lower())
    
    print("clusterID:", clusterID)
    
    filenameGS=goldStandardSummaries[goldStandardSummaryIndex]
    goldStandardSummaryIndex+=1
    file = open(filenameGS, "r") 
    linesGS=[linesGS.rstrip('\n') for linesGS in file]
    linesGS=linesGS[2:]
    summaryGS=' '.join(linesGS)
    
    summaryGS=summaryGS[4:]

    summaryGS=re.sub("\[.*?\] ","",summaryGS)
    summaryGS=summaryGS.lower()
    summaryGS.rstrip("\r")
    outputFileName="DUC/reference/summary"+clusterID+"_reference" + clusterID +".txt"
    text_file = open(outputFileName, "w")
    sentencesInSummaryGS=nltk.sent_tokenize(summaryGS)
    for sentence in sentencesInSummaryGS:
    	text_file.write(sentence+'\n')
    
    outputFileName2="DUC/system/summary"+clusterID+"_system" + clusterID +".txt"
    text_file2 = open(outputFileName2, "w")

    sentencesInSummary=nltk.sent_tokenize(summary)
    for sentence in sentencesInSummary:
    	text_file2.write(sentence+'\n')







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







