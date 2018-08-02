import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
class model:
    def __init__(self):
        self.data = None
        self.names = None
        self.similarityMatrix = None
    def setData(self,data):
        self.data = data
        self.names = data.columns
    def preprocessData(self):
        self.data = normalize(self.data,norm='l1',axis=0)

    def generateSimilarityMatrix(self):
        size = data.shape[1]
        self.similarityMatrix = np.empty((size,size))
        for i in range(size):
            for j in range(size):
                t1 = self.data[:,i]
                t2 = self.data[:,j]
                self.similarityMatrix[i][j] = self.computeSimilarity(t1,t2)
    def computeSimilarity(self,t1,t2):
        return self.dist(t1,t2)
    def dist(self,t1, t2):
        return np.abs(t1-t2).sum()
    def visualise(self):
        pass
    def generateClusters(self):
        kmeans = KMeans(n_clusters=3).fit(self.similarityMatrix)

        print(zip(self.names,kmeans.labels_))

if __name__ == "__main__":
    m = model()
    data  = pd.read_csv('example.csv')
    m.setData(data)
    m.preprocessData()
    m.generateSimilarityMatrix()
    m.generateClusters()