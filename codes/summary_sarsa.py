#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:36:27 2017

@author: yuedong
"""

from summarization_efficient import actor_critic_e_trace, Features, Summarization
import matplotlib.pyplot as plt


import glob
import nltk
import nltk.data
import re
import pandas as pd

#from pyrouge import Rouge155
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

#%%
import os
os.chdir("/Users/yuedong/Downloads/comp767_final_proj/")

#%%
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
#%%
goldStandardSummaries= sorted(glob.glob("DUC/GoldStandard/*.txt"))
goldStandardSummaryIndex=0
for clustername in sorted(glob.glob("DUC/docs/*")):  
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
    

    print("clusterID:", clusterID)

    
    outputFileName2="DUC/system/summary"+clusterID+"_system" + clusterID +".txt"
    text_file2 = open(outputFileName2, "w")

    sentencesInSummary=nltk.sent_tokenize(summary)
    for sentence in sentencesInSummary:
    	text_file2.write(sentence+'\n')
#%%
env = Summarization(length_limit=3, documents=documents, lambda_s=0.9)
features = Features(documents)

#%%
agent = actor_critic_e_trace(env, features)

for i in range(0,100):
    #agent = actor_critic_e_trace(env, features)
    agent.train(3)
    
plt.plot(agent.greedy_returns, label='actor_critic, lambda=%s'%i)
print(agent.greedy_summary)

#%% 
plt.plot(agent.greedy_returns, label='sarsa')