import numpy as np
import random
status=0
ACTIONS = ["l", "r"]

GAMMA = 0.9
EPSILON = 0.9
EPOCH = 20
N_STATE = 6
ALPHA = 0.1

q_table=np.zeros((6,2))



def get_value(S, A):
    if A == 'l':
        return q_table[S, 0]
    else:
        return q_table[S, 1]

def take_action(status):
    maxq=np.max(q_table[status])
    
        
    if status==0:
        action='r'
        
    else:
        if maxq == 0 or np.random.uniform() > 0.9:
            action = np.random.choice(ACTIONS)
        else:
            action='l' if np.argmax(q_table[status])==0 else 'r'
    n_status=status+1 if action=='r' else status-1
 
    maxq=np.max(q_table[n_status])
    
    return action,n_status,maxq

def get_reward(S, A):
    if S == 5:
        R = 1
    elif S == 4 and A == 'r':  # S!=5
        R = 1
    else:
        R = 0
    return R
def update_q(S, A, MQ):
    value = (1 - ALPHA) * get_value(s, A) + ALPHA * (get_reward(S, A) + GAMMA * MQ)
    set_value(S, A, value)

def set_value(S, A, V):
    if A == 'l':
        q_table[S, 0] = V
    else:
        q_table[S, 1] = V


for epoch in range(500):
    status=0
    action_set=[]
    while status!=5:
        
        action,n_status,maxq=take_action(status)
        
        value = (1 - ALPHA) * get_value(status, action) + ALPHA * (get_reward(status, action) + GAMMA * maxq)
   
        set_value(status, action, value)
        status=n_status
        action_set.append(action)
        
       
   
    # print(action_set) 
   
S = random.randint(0, 5)
print(S)
step_set=[]   
while S != 5:
    action, n_status,maxQ = take_action(S)
    print(n_status)
    S=n_status
    step_set.append(action)   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
