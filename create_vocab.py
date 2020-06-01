import pickle

def create_vocab(path):

    with open(path, 'rb') as file:
        word2count = pickle.load(file)
        path2count = pickle.load(file)
        target2count = pickle.load(file)
        n_training_examples = pickle.load(file)
    
        word2idx = {'<unk>': 0, '<pad>': 1}
        path2idx = {'<unk>': 0, '<pad>': 1 }
        target2idx = {'<unk>': 0, '<pad>': 1}
        idx2target = {}

        for w in word2count.keys():
            word2idx[w] = len(word2idx)
            
        for p in path2count.keys():
            path2idx[p] = len(path2idx)
            
        for t in target2count.keys():
            target2idx[t] = len(target2idx)
            
        for k, v in target2idx.items():
            idx2target[v] = k
            
        return word2idx, path2idx, target2idx, idx2target