import numpy as np
import random
m=np.zeros((3,4))

env=m.copy()

env[0,3]=1
env[1,3]=-1
env[1,1]=None





def status_change(status,action):
    r,c=status
    if action=='u':
        r=status[0]-1
    if action=='d':
        r=status[0]+1
    if action=='l':
        c=status[1]-1
    if action=='r':
        c=status[1]+1
    return (r,c)







def get_action_set(position):
    ori=['u','d','l','r']
    
    r,c=position
    R,C=env.shape
    action_set=[]
    
    if r-1 in range(R) and [r-1,c]!=[1,1]:
        action_set.append('u')
        
    if r+1 in range(R) and [r+1,c]!=[1,1]:
        action_set.append('d')
    if c-1 in range(C) and [r,c-1]!=[1,1]:
        action_set.append('l')
    if c+1 in range(C) and [r,c+1]!=[1,1]:
        action_set.append('r')
    return action_set

get_action_set((0,1))
def build_table():
    table={}
    R,C=env.shape
    for i in range(R):
        for j in range(C):
            table[(i,j)]={}
            for p in get_action_set((i,j)):
                table[(i,j)][p]=0
    return table

def take_action(status):
    action_set=get_action_set(status)
    
    m_key=max(table[status])
    m_q=table[status][m_key]

    if m_q==0 or np.random.uniform() > 0.9:
        action=random.choice(action_set)
    else:
        action=m_key
   
    n_status=status_change(status,action)
    
    m_key=max(table[n_status])
    m_q=table[n_status][m_key]

    
    return action,n_status,m_q

def get_value(status,action):
    return table[status][action]    

def get_reward(status, action):
    if status==(0,2) and action=='r':
        reward=1
    elif status==(1,2) and action=='r':
        reward=-1
    elif status==(2,3) and action=='u':
        reward=-1
    else:
        reward=0
    return reward
    
def set_value(status, action, value):
    table[status][action]=value

table= build_table()   
GAMMA = 0.9
EPSILON = 0.9
EPOCH = 20
N_STATE = 6
ALPHA = 0.8
tables=[]
for epoch in range(100):
    status=(2,0)
    action_set=[]

    while True:
       
        if (status==(0,3) or status==(1,3)):
            break
        
        action,n_status,maxq=take_action(status)
   
       
        # value = (1 - ALPHA) * get_value(status, action) + ALPHA * (get_reward(status, action) + GAMMA * maxq)

       
        # new_q = reward + self.discount_factor * max(self.q_table[next_state])
        
        
        new_q = get_reward(status, action)+GAMMA * maxq
        # print(new_q)
        # self.q_table[state][action] += self.learning_rate * (new_q - current_q)
        
        table[status][action]+= 0.9 *(new_q-table[status][action])
     
        
        # set_value(status, action, value)
        
        status=n_status
        
        action_set.append(action)
    
    print(action_set)
     
        
    
    
# status=(2,0)    
# action_set=[]    
# while True:
#     if (status==(0,3) or status==(1,3)):
#             break
#     action,n_status,maxq=take_action(status)
#     status=n_status
#     action_set.append(action)
# print(status)
# print(action_set)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
