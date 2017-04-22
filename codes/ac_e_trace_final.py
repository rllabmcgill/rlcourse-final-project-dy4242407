#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:26:56 2017

@author: yuedong
"""
import os
os.chdir("/Users/yuedong/Downloads/comp767_assignment_5/")
#%%
import numpy as np
from pylab import cos
import matplotlib.pyplot as plt
#%%
'''
# mountain car has two variables: position(x-axis) and velocity
# -1.2 <= position <= 0.5
# -0.07 <= velocity <= 0.07
'''
class mountainCar():
    
    def __init__(self):
        self._reset()
        self.nA = 3
        
        self.position_space = [-1.2, 0.5]
        self.velocity_space = [-0.07, 0.07]
        
        self.discrete_position_space = np.linspace(-1.2,0.5,30)
        self.discrete_velocity_space = np.linspace(-0.07,0.07,30)
        
        
    # each episode starts from a random position and velocity
    # which are choosen with uniformly within the ranges
    def _reset(self):
        self.position = np.random.uniform(low=-1.2, high=0.5, size=None)
        self.velocity = np.random.uniform(low=-0.07, high=0.07, size=None)
        return self.position, self.velocity

    def _step(self, action):
    #position,v = self.position, self.velocity
        if not action in (-1,0,1):
            print('Invalid action:', action)
            raise "StandardError"

        self.velocity += 0.001*action - 0.0025*cos(3*self.position)
        if self.velocity < -0.07:
            self.velocity = -0.07
        elif self.velocity >= 0.07:
            self.velocity = 0.07
            
        self.position += self.velocity
        if self.position >= 0.5:
            return None, -1
        if self.position < -1.2:
            self.position = -1.2
            self.velocity = 0.0
        return (self.position, self.velocity), -1
    
#%%

# this code is inspired from https://github.com/ctevans/.../Tilecoder.py

# mountain car has two variables: position(x-axis) and velocity
# -1.2 <= position <= 0.5
# -0.07 <= velocity <= 0.07

# devide the 2D space into an 8 by 8 grid
# then shift this grid with 1/4 of a tile width to obtain 4 tilings (partitions) 
numTiles = 8 
numTilings = 4

positionTileMove = ((0.5 + 1.2) / numTiles) / numTilings
velocityTileMove = ((0.07 + 0.07) / numTiles) /numTilings

# in order to make sure all points are covered after shifting 
# add one extra row and one extra column for shifting
# numver of total features  = 9x9x4
numFeatures = numTilings * (numTiles+1) * (numTiles+1)


# x = position, y = velocity
# note move a tiling by (a, b), then find the index of a point
# is the same as moving the points by (-a, -b)
# shift direction in this code is to the left-bottom corner
def tilecode(x,y,tileIndices):
    
    # find the distance of x to the leftmost position
    x = x + 1.2
    # find the distance of y to smallest velocity
    y = y + 0.07

    for i in range (numTilings):
        
        # in tiling i, we move a points by 
        # (-i*positionTileMove,i*velocityTileMove) for feature encoding
        xMove = i * (-positionTileMove)
        yMove = i * (-velocityTileMove)
	
        xTileIdx = int(numTiles * (x - xMove)/1.7)
        yTileIdx = int(numTiles * (y - yMove)/0.14)

        tileIndices[i] = i * 81 + ( yTileIdx * 9 + xTileIdx)
    
    
def tileCoderIndices(x,y):
    tileIndices = [-1]*numTilings
    tilecode(x,y,tileIndices)
    #print('Tile indices for input (%s,%s) are : '%(x,y), tileIndices)
    return tileIndices
                                                
#printTileCoderIndices(0.5,0.07)
#[-1, -1, -1, -1]
#Tile indices for input (0.5,0.07) are :  [80, 161, 242, 323]

# INPUTS
#   s: state =(position,velocity) 
#   a: action, integer: throttle reverse (-1), zero throttle (0), throttle forwards (1)
# returns a binary vector of length (4*9*9)*3 representing the features
def phi(s, a):
    tmp = np.zeros(shape=(4*9*9,3)) #zeros array of dim 3*6*2
    #putting one where a feature is on
    for i in tileCoderIndices(s[0],s[1]):
            tmp[i,a] = 1 
    return(tmp.flatten()) #returning 'vectorized' (1-dim) array
    
def phi_state(s):
    tmp = np.zeros(4*9*9) #zeros array of dim 3*6*2
    for i in tileCoderIndices(s[0],s[1]):
            tmp[i] = 1 
    return tmp #returning 'vectorized' (1-dim) array

#%%
# here is the code for the policy approximation
    
# this function calculates h(s,a, theta)
def preference_cal(theta, state, action):
    h_s_a = np.dot(theta, phi(state,action))
    return h_s_a
    
# this function computes pi(a|s,theta) given the action_set
# note the regular softmax overflow at np.exp(800)
# we use the trick at https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
# where we compute everything in log space
def policy_prob(theta, state, action, action_set):
    
    pref_v = [preference_cal(theta, state, a) for a in action_set]
    
    m = np.max(pref_v)
    e_sum = np.sum(np.exp(pref_v - m))
    pi = np.exp(preference_cal(theta, state, action)-m) / e_sum
    
#    m = max(pref_v)
#    shift_perf_v=pref_v - m
#    
#    log_sum = m + np.log(sum(np.exp(shift_perf_v)))
#    
#    log_prob = preference_cal(theta, state, action) - log_sum
#                             
#    pi = np.exp(log_prob)


    
    return  pi
    
# this function returns an action, where the action a is chosen with prob pi(a|s,theta)
# using policy parametrization
def policy_par(theta, state, action_set):
    
    pi_actions = []
    for a in action_set:
        pi_actions.append(policy_prob(theta, state, a, action_set))
            
    return np.random.choice(action_set, p= pi_actions)

#%%
# actor-critic with eligibility trace

class actor_critic_e_trace:
    def __init__(self, environment, mlambda_theta=0.5, mlambda_w=0.5, gamma=1, alpha=0.05, beta=0.1):

        self.env = environment
        self.gamma = gamma
        self.mlambda_w = mlambda_w
        self.mlambda_theta = mlambda_theta
        self.alpha = alpha
        self.beta = beta
        
        self.theta = np.zeros(4*9*9*3) 
        self.W = np.zeros(4*9*9) 
        
        # o stick, 1 hit
        self.action_set = [-1,0,1]
        
        self.iterations = 0
        self.returns = []
        
    
        # V is simply the dot product of phi and w
    def cal_V(self, s):
        return np.dot(phi_state(s),self.W)
    
    def train(self, iterations):        
        # Loop episodes
        for episode in range(iterations):
            s = self.env._reset()
            
            e_theta = np.zeros(4*9*9*3) 
            e_W = np.zeros(4*9*9)
            I = 1
            rewards = []
            time_step = 0
            
            # generate an episode untill the end
            while s != None and time_step<200:
                
                a = policy_par(self.theta, s, self.action_set)
                # execute action
                s_next, r = self.env._step(a)
                rewards.append(r)
                
                if s_next != None:
                    # reassign s and a, add to trajactory
                    delta = r + self.gamma * self.cal_V(s_next) - self.cal_V(s)     
                else:
                    delta = r - self.cal_V(s)
                
                
                feature = phi(s,a)
                subtract = sum([policy_prob(self.theta, s, a_i, self.action_set) * 
                               phi(s, a_i) for a_i in self.action_set])
                gradient = feature - subtract
               
                e_W = self.mlambda_w * e_W + I * phi_state(s)
                e_theta = self.mlambda_theta * e_theta + I * gradient
                self.W += self.beta * delta * e_W
                self.theta += self.alpha * I * delta * e_theta
                
                I = self.gamma * I
                s = s_next
                
                time_step += 1 
            #print("finished in %s time steps" % time_step)
            #print("time_step", time_step)
            
            self.returns.append(sum(rewards))
            
        self.iterations += iterations
        
#%%
env = mountainCar()
#%%
def average_run(env, num_runs, episodes_per_run, mlambda_theta=0.5, mlambda_w=0.5):
    
    m = np.zeros((num_runs, episodes_per_run))
    
    for i in range (num_runs):
        
        agent = actor_critic_e_trace(env,mlambda_theta, mlambda_w)
        agent.train(episodes_per_run)
        
        m[i] = agent.returns
 
    return np.sum(m, axis=0)/num_runs

def runningMean(x, N=10):
    y = np.zeros((len(x)-N,))
    for ctr in range(len(x)-N):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N
#%%
plt.figure(1)
for i in np.linspace(0,1,5):
    avg_returns = average_run(env, 5, 200,i, i)
    reinforce_mean = runningMean(avg_returns, N=20)
    plt.plot(reinforce_mean, label='actor_critic, lambda=%s'%i)
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
#%%
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#%%
# draw reward curves


