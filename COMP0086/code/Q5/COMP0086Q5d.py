import numpy as np

def transition(text,symbols):
    T = []
    for s1 in symbols:
        count_s1 = text.count(s1)
        pairs = []
        for s2 in symbols:
            count_pairs = text.count(s1+s2)
            pairs.append(count_pairs)
        prob = np.round(np.array(pairs)/count_s1,6)
        trans_prob = np.round(prob / prob.sum(),6) # normalisation
        T.append(trans_prob)
    return np.array(T)

def swap_sigma(original_sigma):
    new_sigma = original_sigma
    indx1 = np.random.randint(0,len(new_sigma),1).tolist()[0]
    indx2 = np.random.randint(0,len(new_sigma),1).tolist()[0]
    if indx1 == indx2:
        new_sigma = swap_sigma(original_sigma)
    else:
        new_sigma[indx1],new_sigma[indx2] = new_sigma[indx2],new_sigma[indx1]
    return new_sigma

def loglikelihood(message,T,sigma):
    score = 0
    for i in range(len(sigma)):
        for j in range(len(sigma)):
            count_pairs = message.count(sigma[i]+sigma[j])
            if T[i,j] != 0:
                score += np.log(T[i,j]) * count_pairs
            else:
                score += np.log(T[i,j]+0.000001) * count_pairs # add bias
    return score

def swap_element(lst,indx1,indx2):
    lst[indx1],lst[indx2] = lst[indx2],lst[indx1]
    return lst

with open("symbols.txt",'r') as f1:
    contents1 = f1.readlines()
    symbols = [line.strip('\n') for line in contents1]

with open('message.txt','r') as f2:
    message = f2.read().replace('\n','')

with open('war&peace.txt','r') as f2:
    text = f2.read().replace('\n','').lower()

# initialise the sigma map intelligently 
sigma = symbols[:]
freqs = []
for symbol in symbols:
    freqs.append(message.count(symbol))
max1_indx = freqs.index(max(freqs))
freqs[max1_indx] = 0
max2_indx = freqs.index(max(freqs))
max1_swap = symbols.index(" ")
max2_swap = symbols.index("e")
sigma = swap_element(sigma,max1_indx,max1_swap)
sigma = swap_element(sigma,max2_indx,max2_swap)

# create MH sampler
iteration = 10001
T = transition(text,symbols)
for i in range(iteration):
    new_sigma = swap_sigma(sigma[:]) # only pick values of sigma, instead of overwriting sigma
    logscore1 = loglikelihood(message,T,sigma)
    logscore2 = loglikelihood(message,T,new_sigma)
    accept = np.exp(logscore2-logscore1)
    if np.random.uniform(0,1,1) < min(1,accept):
        sigma = new_sigma[:]
    if i % 100 == 0:
        message_deco = message
        m_list = list(message_deco)
        for j in range(len(m_list)):
            m_list[j] = symbols[sigma.index(message[j])]
        message_deco = "".join(m_list) 
        print(f"{i}: {message_deco[:60]}")
    if i == 10000:
        file = open("decoding result.txt",'w')
        file.write(message_deco)
        file.close()

