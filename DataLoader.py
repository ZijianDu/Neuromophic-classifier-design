import numpy as np

def DataLoader(seedval=1, fileName="ClaveVectors_Firm-Teacher_Model.txt", output_idx=-3):
    np.random.seed(seedval)
    dataMtx = getData("../Data/" + fileName)
    tr_idx,te_idx,va_idx = splitData(len(dataMtx),.6,.2,.2)
    train_set = labeledData(dataMtx[tr_idx, 0: -4], dataMtx[tr_idx, output_idx])
    valid_set = labeledData(dataMtx[va_idx, 0: -4], dataMtx[va_idx, output_idx])
    test_set = labeledData(dataMtx[te_idx, 0: -4], dataMtx[te_idx, output_idx])
    return train_set, valid_set, test_set

def splitData(dataLength, tr_r,te_r,va_r):
    order = np.random.permutation(dataLength)
    a = int(np.floor(tr_r*dataLength))
    b = int(np.floor((tr_r+te_r) * dataLength))
    return order[0:a], order[a+1:b], order[b+1:]
def getData(path):
    data = []
    with open(path, 'r') as inp:
        for line in inp:
            data.append([int(bit) for bit in line.strip().split()[0:20]])
    return np.array(data)
class labeledData:
    def __init__(self,X,y):
        self.X = X
        self.y = y

if __name__ == "__main__":
    DataLoader(1, "ClaveVectors_Firm-Teacher_Model.txt" )