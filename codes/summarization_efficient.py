#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:26:56 2017

@author: yuedong
"""
import os
os.chdir("/Users/yuedong/Downloads/comp767_final_proj/")
#%%
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
#%%
#%%
document1 = "Python is a 2000 made-for-TV horror movie directed by Richard Clabaugh. The film features several cult favorite actors, including William Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy, Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean Whalen. The film concerns a genetically engineered snake, a python, that escapes and unleashes itself on a small town. It includes the classic final girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles, California and Malibu, California. Python was followed by two sequels: Python II (2002) and Boa vs. Python (2004), both also made-for-TV films."

document2 = "Python, from the Greek word, is a genus of nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are recognised. A member of this genus, P reticulatus, is among the longest snakes known."

document3 = "The Colt Python is a 357 Magnum caliber revolver formerly manufactured by Colt's Manufacturing Company of Hartford, Connecticut. It is sometimes referred to as a Combat Magnum.[1] It was first introduced in 1955, the same year a Smith &amp; Wesson's M29 44 Magnum. The now discontinued Colt Python targeted the premium revolver market segment. Some firearm collectors and writers such as Jeff Cooper, Ian V Hogg, Chuck Hawks, Leroy Thompson, Renee Smeets and Martin Dougherty have described the Python as the finest production revolver ever made."

summary= "Python is a 2000 made-for-TV horror movie directed by Richard Clabaugh. Python, from the Greek word, is a genus of nonvenomous pythons[2] found in Africa and Asia."

#documents = [document1, document2, document3]
summary = summary.lower()
documents = [document1.lower(), document2.lower(), document3.lower()]

#%%
# The following defines the score function for a summary 

def tfidfFit(documents):
    
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(documents)
    freq_term_matrix = count_vect.transform(documents)
    
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)
        
    return count_vect, tfidf
   
    
def getTfidf(text_unit, count_vect, tfidf):
  # text unit is either a document or a sentence
  #transform a particular setence or doc to tfidf vector
    if type(text_unit)==str:
        doc_freq_term = count_vect.transform([text_unit])
    else:
        doc_freq_term = count_vect.transform(text_unit)
        
    doc_tfidf_matrix = tfidf.transform(doc_freq_term)
  
    return doc_tfidf_matrix


def getPosition(sentence,documents):
    positionScore=0
    
    for document in documents:
        listOfSentencesInDocument=document.split('.')[0:-1]
        listOfSentencesInDocument = [x.lstrip(' ') for x in listOfSentencesInDocument]
        #print(listOfSentencesInDocument)
        if(sentence in listOfSentencesInDocument):
            #print("index",listOfSentencesInDocument.index(sentence))
            positionScore=(1.0/(listOfSentencesInDocument.index(sentence)+1))
        elif (sentence.lstrip() in listOfSentencesInDocument):
            #print("index_lower",listOfSentencesInDocument.index(sentence.lstrip()))
            positionScore=(1.0/(listOfSentencesInDocument.index(sentence.lstrip())+1))
            
    return positionScore
    

def Relevant(xi, count_vect, tfidf, documents):
    #xi is a sentence in the summary
    return cosine_similarity(getTfidf(xi, count_vect, tfidf), 
                             getTfidf(' '.join(documents), count_vect, tfidf)
                             ).flatten()[0] + getPosition(xi, documents)
    

def Redundancy(xi, xj, count_vect, tfidf):
    #xi and xj are two sentences in the summary
    return cosine_similarity(getTfidf(xi, count_vect, tfidf), 
                             getTfidf(xj, count_vect, tfidf)).flatten()[0]
    
    
# input: lambda_s  the trade-off between relevance and redundancy 
def score(summary, documents, lambda_s):
    # summary is a string with multiple sentences
    # each document is a string with multiple sentences
    #print("summary",summary)
    sInSummary = summary.split('.')
    #print("sInSummary",sInSummary)
    count_vect, tfidf = tfidfFit(documents)
    
    sumRel = 0
    for s in sInSummary:
        #print("******")
        #print(s.lstrip())
        #print("getPostion:", getPosition(s, documents))
        sumRel += lambda_s * Relevant(s, count_vect, tfidf, documents)
    #print("sumRel",sumRel)
        
    
    sumRed = 0 
    for i in range(len(sInSummary)):
        for j in range(i+1, len(sInSummary)):
            sumRed += (1-lambda_s) * Redundancy(sInSummary[i],sInSummary[j],
                      count_vect, tfidf)
    #print("sumRed",sumRed)
            
    return sumRel-sumRed
##%%
#env = Summarization(3,documents,0.9)
#env._reset()
#env._step(1)
#print(score('. '.join(env.summary),documents, 0.9))
#env._step(2)
#print(score('. '.join(env.summary),documents, 0.9))
#
#print("**************")
#env._reset()
#env._step(2)
#print(score('. '.join(env.summary),documents, 0.9))
#env._step(1)
#print(score('. '.join(env.summary),documents, 0.9))
#%%

class Summarization():
    

    def __init__(self, length_limit, documents, lambda_s):
        self._reset()
        
        self.lambda_s = lambda_s
        self.documents = documents
        self.documentMerged = ''.join(documents).split('.')[0:-1]
        self.numSentences = len(self.documentMerged)
        
        self.nA = self.numSentences + 1
        self.actions=list(range(self.numSentences + 1))
        
        self.length_limit = length_limit
        self.penalty = -1
        
    # each episode starts from a random position and velocity
    # which are choosen with uniformly within the ranges
    def _reset(self):
        self.summary=[]
        return self.summary
    

    def _step(self, action):
    #position,v = self.position, self.velocity
        if not action in self.actions:
            print('Invalid action:', action)
            raise "StandardError"

        if len(self.summary) > self.length_limit:
            print("over length %s, length limit %s, return%s" 
                  %(len(self.summary), self.length_limit, self.penalty))
            # return summary, finished =1, penalty 
            return self.summary, 1, self.penalty
        elif action==0:
            reward_score = score('. '.join(self.summary), self.documents, self.lambda_s)
            #print("stop summarizing, score %s" %reward_score)
            # action == 0 means we choose to finish the summary
            # return summary, finished =1, score based on score function
            return self.summary, 1, reward_score
        else:
            #print("inserting sentence",action)
            self.summary.append(self.documentMerged[action-1])
            return self.summary, 0, 0

#%%
env = Summarization(length_limit=3, documents=documents, lambda_s=0.9)

#%%
#from featureExtractor import getFeatureVector
#from documentProcess import getFeatureVector
#%%
import glob
import nltk
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
import copy

    #top_features,featureVector=getFeatureVector(documents,summary.lower())
#%%
class Features():
    
    
    def __init__(self, documents):
        
        self.documents = documents
        self.documentMerged = ''.join(documents).split('.')[0:-1]
        self.numSentences = len(self.documentMerged)
        
        self.nA = self.numSentences + 1
        self.actions = list(range(self.numSentences + 1))
        
        
        vectorizer = TfidfVectorizer(analyzer='word', tokenizer=self.tokenize_only,
                                     ngram_range=(1,1), min_df = 0,
                                                 stop_words = 'english')
        X = vectorizer.fit_transform(self.documents)
        
        #indices = np.argsort(vectorizer.idf_)[::-1]
        features = vectorizer.get_feature_names()
        #top_n = 100
        #top_features = [features[i] for i in indices[:top_n]]
        top_features=self.top_mean_feats(X,features)
        self.top_features2=[0]*100
        index=0
        for feature in top_features.feature:
            self.top_features2[index]=feature
            index+=1
            
        
        self.list_sent_in_doc=[]
        for document in documents:
            self.list_sent_in_doc.append(document.split('.')[0:-1])
    
    def phi_state(self, summary):
        if type(summary) == list:
            summary = '.'.join(summary)
        tmp = self.getFeatureVector(summary)
        return tmp #returning 'vectorized' (1-dim) array
    
    def phi(self, summary, a):
        # only the features with the action a have feature values 
        # rest entries are all zeros
        tmp = np.zeros(0)
        features = self.phi_state(summary)
        len_features = len(features)
        
        for i in range(self.nA):
            if i != a:
                tmp = np.append(tmp, np.zeros(len_features)) 
            else:
                tmp= np.append(tmp,features)
        #print("non_zero index", (np.nonzero(tmp)[0]/104).astype(int))
        return tmp
    
    def tokenize_only(self, text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens

    
    def getPositionSum(self, summary, documents):
        positionScore=0
        listOfSentencesInSummary=summary.split('.')[0:-1]
        for sentence in listOfSentencesInSummary:
            for document in documents:
                listOfSentencesInDocument=document.split('.')[0:-1]
                listOfSentencesInDocument = [x.lstrip(' ') for x in listOfSentencesInDocument]
                if(sentence in listOfSentencesInDocument):
                    positionScore+=(1.0/(listOfSentencesInDocument.index(sentence)+1))
                    break
                elif (sentence.lstrip() in listOfSentencesInDocument):
                    positionScore+=(1.0/(listOfSentencesInDocument.index(sentence.lstrip())+1))
                    break
        return positionScore
    
#    def getPositionSum(self, summary):
#        positionScore=0
#        listOfSentencesInSummary=summary.split('.')[0:-1]
#        for sentence in listOfSentencesInSummary:
#            for listOfSentencesInDocument in self.list_sent_in_doc:
#                if(sentence in listOfSentencesInDocument):
#                    positionScore+=(1.0/(listOfSentencesInDocument.index(sentence)+1))
#                    break
#                elif (sentence.lstrip() in listOfSentencesInDocument):
#                    positionScore+=(1.0/(listOfSentencesInDocument.index(sentence.lstrip())+1))
#                    break
#        return positionScore
#    
                
    
    def getCoverageRatio(self, coverage,n):
        coverageRatio=sum(coverage)/n
                
        return coverageRatio
        
    def getRedundancyRatio(self, summary,top_features,threshhold):
        count=np.zeros(len(top_features))
        wordsOfSummary=word_tokenize(summary)
        redundancyRatio=0
        for word in wordsOfSummary:
            if word in top_features:
                count[top_features.index(word)]+=1
                                  
        redundancyRatio=len(np.argwhere(count>threshhold))/100.0
        return redundancyRatio
        
    def getCoverage(self, summary,top_features):
        coverage=np.zeros(len(top_features))
        wordsOfSummary=word_tokenize(summary)
        for word in wordsOfSummary:
            if word in top_features:
                    coverage[top_features.index(word)]=1
        return coverage
    
    def getLengthRatio(self, summary,K):
        #return (len(summary.split('.'))-1)/K
        return (len(summary.split(' '))/K)
    
        
    #summary= "The Swiss government has ordered no investigation of possible bank accounts belonging to former Chilean dictator Augusto Pinochet, a spokesman said Wednesday. Weekend newspaper reports in Spain said a Spanish judge who ordered Pinochet's arrest has issued a petition aimed at freezing any accounts the 82-year-old general might have in Luxembourg and Switzerland. But government spokesman Achille Casanova said no accounts have so far been frozen in Switzerland and no investigation order has been given to federal banking authorities. Pinochet has been held at a London clinic since his arrest earlier this month."
    #summary='. '.join([sentencesInDocuments[0][0],sentencesInDocuments[1][0],sentencesInDocuments[2][0]])
    #summary=summary+'.'    
    def top_tfidf_feats(self, row, features, top_n=100):
        ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats)
        df.columns = ['feature', 'tfidf']
        return df
        
    def top_feats_in_doc(self, Xtr, features, row_id, top_n=100):
        row = np.squeeze(Xtr[row_id].toarray())
        return self.top_tfidf_feats(row, features, top_n)
        
    def top_mean_feats(self, Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=100):
        ''' Return the top n features that on average are most important amongst documents in rows
            indentified by indices in grp_ids. '''
        if grp_ids:
            D = Xtr[grp_ids].toarray()
        else:
            D = Xtr.toarray()
    
        D[D < min_tfidf] = 0
        tfidf_means = np.mean(D, axis=0)
        return self.top_tfidf_feats(tfidf_means, features, top_n)
        
    
    def getFeatureVector(self, summary):
        coverage=self.getCoverage(summary,self.top_features2)
        coverageRatio=self.getCoverageRatio(coverage,100)
        redundancyRatio=self.getRedundancyRatio(summary,self.top_features2,1)
        lengthRatio=self.getLengthRatio(summary,250.0)
        positionSum=self.getPositionSum(summary,documents)
        #positionSum=self.getPositionSum(summary)
        featureVector=np.zeros(104)
        featureVector[0:100]=coverage
        featureVector[100]=coverageRatio
        featureVector[101]=redundancyRatio
        featureVector[102]=lengthRatio
        featureVector[103]=positionSum
        return featureVector[100:104]

# features = Features(documents)
#%%
# here is the code for the policy approximation


# this function calculates h(s,a, theta)

def preference_cal(theta, state, action, phi):
    h_s_a = np.dot(theta, phi(state,action))
    #print("h_s_a",phi(state,action),"for", action)
    return h_s_a
    
# this function computes pi(a|s,theta) given the action_set
# note the regular softmax overflow at np.exp(800)
# we use the trick at https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
# where we compute everything in log space
def policy_prob(theta, state, action, action_set, phi):
    
    pref_v = [preference_cal(theta, state, a, phi) for a in action_set]
    #print("pref_v",pref_v)
    
    m = np.max(pref_v)
    e_sum = np.sum(np.exp(pref_v - m))
    pi = np.exp(preference_cal(theta, state, action, phi)-m) / e_sum
    #print('pi_dist',pi)
    return  pi
    
# this function returns an action, where the action a is chosen with prob pi(a|s,theta)
# using policy parametrization
def policy_par(theta, state, action_set, phi):
    
    pi_actions = []
    for a in action_set:
        pi_actions.append(policy_prob(theta, state, a, action_set, phi))
    #print("action_set",action_set, "pi_actions", pi_actions)
    return np.random.choice(action_set, p= pi_actions)

def policy_greedy(theta, state, action_set, phi):
    
    pi_actions = []
    
    for a in action_set:
        pi_actions.append(policy_prob(theta, state, a, action_set, phi))
    
    one_hot=[]
    for i in pi_actions:
        one_hot.append(int(i==max(pi_actions)))
    #print("one_hot:", one_hot)
    
    if sum(one_hot) == 0:
         one_hot = [x+1 for x in one_hot]
         
    if sum(one_hot) != 1:
        divide = sum(one_hot) 
        one_hot = [x/divide for x in one_hot]
#        for i in range(len(one_hot)):
#            one_hot[i] = one_hot[i]/divide
    #print("one_hot:", one_hot)
    return np.random.choice(action_set, p= one_hot)
#%%
# actor-critic with eligibility trace


class actor_critic_e_trace:
    def __init__(self, environment, features, mlambda_theta=0.5, mlambda_w=0.5, gamma=1, alpha=0.01, beta=0.05):

        self.env = environment
        self.features = features
        
        self.gamma = gamma
        self.mlambda_w = mlambda_w
        self.mlambda_theta = mlambda_theta
        self.alpha = alpha
        self.beta = beta
        
        self.theta = np.zeros(4 * self.env.nA) 
        self.W = np.zeros(4) 
        
        self.iterations = 0
        self.returns = []
        
        self.greedy_returns=[]
        self.greedy_summary=[]
        
    
        # V is simply the dot product of phi and w
    def cal_V(self, s):
        return np.dot(self.features.phi_state(s),self.W)
    
    def greedy_select(self):
        s = self.env._reset()
        action_set = copy.copy(self.env.actions)
            #print(action_set)
        action_set.remove(0)
        
        rewards = []
        term = False
        while not term:
            
            if len(s)<self.env.length_limit:
                    a = policy_greedy(self.theta, s, action_set, self.features.phi)
                #cheat a bit here, force to stop before going over the length limit
            else:
                    a=0
                    
            s_next, term, r = self.env._step(a)
            rewards.append(r)
            
            if a != 0:
                    action_set.remove(a)
                #print(action_set)
            s = s_next
        return s, sum(rewards)
            
    
    def train(self, iterations):        
        # Loop episodes
        for episode in range(iterations):
            s = self.env._reset()
            
            action_set = copy.copy(self.env.actions)
            #print(action_set)
            action_set.remove(0)
            #print(action_set)
            
            e_theta = np.zeros(4 * self.env.nA) 
            e_W = np.zeros(4)
            
            I = 1
            rewards = []
            time_step = 0
            term = False
            
            # generate an episode untill the end
            while not term:
                
                if len(s)<self.env.length_limit:
                    a = policy_par(self.theta, s, action_set, self.features.phi)
                #cheat a bit here, force to stop before going over the length limit
                else:
                    a=0
                
                # execute action
                s_next, term, r = self.env._step(a)
                
                #remove a from the action set
                if a != 0:
                    action_set.remove(a)
                #print(action_set)
                
                rewards.append(r)
                
                if term == False:
                    # reassign s and a, add to trajactory
                    delta = r + self.gamma * self.cal_V(s_next) - self.cal_V(s)     
                else:
                    delta = r - self.cal_V(s)
                
                
                feature = self.features.phi(s,a)
                subtract = sum([policy_prob(self.theta, s, a_i, self.env.actions,self.features.phi) * 
                               self.features.phi(s,a_i) for a_i in self.env.actions])
                gradient = feature - subtract
                #gradient = feature
                
                e_W = self.mlambda_w * e_W + I * self.features.phi_state(s)
                e_theta = self.mlambda_theta * e_theta + I * gradient
                
            
#                print("non_zero_theta:", 
#                      (np.nonzero(e_theta)[0]/4).astype(int),
#                               "for",a)
#                
#                print("non_zero_theta:", 
#                      np.unique((np.nonzero(e_theta)[0]/4).astype(int)),
#                               "for",a)
                
                beta = self.beta*101/(100 + np.power(time_step+1, 1.1))
                self.W += beta * delta * e_W
                
                alpha = self.alpha*101/(100 + np.power(time_step+1, 1.1))
                self.theta += alpha * I * delta * e_theta
                #print("nonzero_theta:", np.unique((np.nonzero(self.theta)[0]/4).astype(int)), "for",a)
                #print("sum theta:", sum(self.theta))
                
                I = self.gamma * I
                s = s_next
                
                time_step += 1 
            #print("finished in %s time steps" % time_step)
            #print("time_step", time_step)
            
            self.returns.append(sum(rewards))
            greedy_summary, greedy_return = self.greedy_select()
            self.greedy_returns.append(greedy_return)
            #print("episode:%s, score:%s" %(episode, self.returns[episode]))
        
        self.iterations += iterations
        final_summary, final_return = self.greedy_select()
        print(final_summary)
        print("greedy_select", final_return)
        print("iterations:%s, score:%s" %(self.iterations, self.returns[self.iterations-1]))
        
#%%
env = Summarization(length_limit=3, documents=documents, lambda_s=0.9)
features = Features(documents)

#agent = actor_critic_e_trace(env, features)
#
#for i in range(0,100):
#    #agent = actor_critic_e_trace(env, features)
#    agent.train(3)
#%%    
#plt.plot(agent.greedy_returns, label='actor_critic, lambda=%s'%i)
#print(agent.greedy_summary)
#%%

def average_run(env,features, num_runs, episodes_per_run, mlambda_theta=0.5, mlambda_w=0.5):
    
    m = np.zeros((num_runs, episodes_per_run))
    
    for i in range (num_runs):
        
        agent = actor_critic_e_trace(env,features, mlambda_theta, mlambda_w)
        agent.train(episodes_per_run)
        
        m[i] = agent.greedy_returns
 
    return np.sum(m, axis=0)/num_runs

def runningMean(x, N=2):
    y = np.zeros((len(x)-N,))
    for ctr in range(len(x)-N):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N
#%%
from time import time
t=time()

plt.figure(1)
for i in np.linspace(0,1,5):
    avg_returns = average_run(env, features, 3, 300, i, i)
    reinforce_mean = runningMean(avg_returns, N=1)
    plt.plot(reinforce_mean, label='actor_critic, lambda=%s'%i)
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')

print("run time:", time()-t)
#%%
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#%%
# draw reward curves


#%%
## testing case for the score function
#env._step(8)
##env._step(3)
#env._step(0)