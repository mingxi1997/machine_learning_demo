import numpy as np
import random

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.9
EPOCH = 20
N_STATE = 6
ACTIONS = ["left", "right"]

np.random.seed(2)

Q_Table = np.zeros((N_STATE, len(ACTIONS)))


def get_reward(S, A):
    if S == 5:
        R = 1
    elif S == 4 and A == 'right':  # S!=5
        R = 1
    else:
        R = 0
    return R


def take_action(S):
    if S == 0:
        return 1, 'right', np.max(Q_Table[1, :])
    maxQ = np.max(Q_Table[S, :])
    if maxQ == 0 or np.random.uniform() > EPSILON:
        if maxQ == 0:
            print("maxQ is 0")
        else:
            print("Random 0.1 hit")
        action = np.random.choice(ACTIONS)
    else:
        idx = np.argmax(Q_Table[S, :])
        print("IDX:", idx)
        action = ACTIONS[idx]
    if action == 'left':
        SN = S - 1
    else:
        SN = S + 1
    maxQ = np.max(Q_Table[SN, :])
    return SN, action, maxQ


def get_value(S, A):
    if A == 'left':
        return Q_Table[S, 0]
    else:
        return Q_Table[S, 1]


def set_value(S, A, V):
    if A == 'left':
        Q_Table[S, 0] = V
    else:
        Q_Table[S, 1] = V


def update_q(S, A, MQ):
    value = (1 - ALPHA) * get_value(S, A) + ALPHA * (get_reward(S, A) + GAMMA * MQ)
    set_value(S, A, value)

EPOCH=20
for loop in range(EPOCH):
    print("Loop:", loop)
    S = 0
    step = 0
    steps = []
    while S != 5:
        
        SN, action, maxQ = take_action(S)
        update_q(S, action, maxQ)
        S = SN
        step += 1
        steps.append(action)
    print("State:", S, "TotalSteps:", step, "\nDetail:", steps)

print("Q_Table:", Q_Table)

S = random.randint(0, 5)
print("Initial S", S)
step = 0
steps = []
while S != 5:
    SN, action, maxQ = take_action(S)
    S = SN
    step += 1
    steps.append(action)
print("Real State:", S, "TotalSteps:", step, "\nDetail:", steps)
