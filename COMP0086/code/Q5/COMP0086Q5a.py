import numpy as np
import pandas as pd

def transition(text,symbols):
    T = []
    for s1 in symbols:
        count_s1 = text.count(s1)
        pairs = []
        for s2 in symbols:
            count_pairs = text.count(s1+s2)
            pairs.append(count_pairs)
        prob = np.round(np.array(pairs)/count_s1,6)
        trans_prob = np.round(prob / prob.sum(),3) # normalisation
        T.append(trans_prob)
    return np.array(T)

with open("symbols.txt",'r') as f1:
    contents1 = f1.readlines()
    symbols = [line.strip('\n') for line in contents1]

with open('war&peace.txt','r') as f2:
    text = f2.read().replace('\n','').lower()

# trainsition probabilities
T = transition(text,symbols)
df_transition = pd.DataFrame(T)
df_transition.columns = symbols
df_transition.index = symbols
df_transition.to_csv('transition.csv')

# stationary distribution
symbol_occur = []
all_occur = 0
for symbol in symbols:
    count_s = text.count(symbol)
    symbol_occur.append(count_s)
    all_occur += count_s
stationary = np.round(np.array(symbol_occur)/all_occur,6)
stationary = np.round(stationary/sum(stationary),3)
df_stationary = pd.DataFrame({'symbols':symbols,'stationary_prob':stationary})
df_stationary.to_csv("stationary.csv",index=False)