# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import utils

def value_iteration(parking, gamma, theta):

    V = np.zeros(len(parking.State))
    
    while True:
      
        delta = 0
        for s in parking.State:
            v = V[s]
            bellman_optimality(parking, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
            
    pi = np.ones((len(parking.State), len(parking.Action))) / len(parking.Action)
    
    for s in parking.State:
        greedify_policy(parking, V, pi, s, gamma)
        
    return V, pi

def bellman_optimality(parking, V, s, gamma):

    q_values = np.zeros(len(parking.Action))    
    
    for action in parking.Action:        
        
        for state, (reward, probability) in enumerate (parking.transitions(s, action)):
            
            q_values[action] += probability*(reward + gamma * V[state])
    
    V[s] = np.max(q_values)
    
def greedify_policy(parking, V, pi, s, gamma):

    q_values = np.zeros(len(parking.Action))
    
    for action in parking.Action:
        
        for state, (reward, probability) in enumerate (parking.transitions(s, action)):
            
            q_values[action] += probability*(reward + gamma * V[state])
            
    
    pi[s] = np.zeros(len(parking.Action))
    pi[s][q_values.argmax()] = 1

env = utils.ParkingWorld(num_spaces=10, num_prices=4)
gamma = 0.9
theta = 0.1
V, pi = value_iteration(env, gamma, theta)

utils.plot(V, pi)