import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
class model:
    def __init__(self):
        self.data = None
        self.names = None
        self.similarityMatrix = None
    def setData(self,data):
        self.data = data
        self.names = data.columns
    def getData(self):
        return self.data.copy()
    def getSimilarityMatrix(self):
        return self.similarityMatrix.copy()

    def preprocessData(self):
        self.data = normalize(self.data,norm='l1',axis=0)
    def generateSimilarityMatrix(self):
        size = self.data.shape[1]
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

    def generateClusters(self):
        kmeans = KMeans(n_clusters=3).fit(self.similarityMatrix)

        print(zip(self.names,kmeans.labels_))

    def reduceDimensionality(self):
        pca = PCA(n_components = 2)
        components = pca.fit_transform(self.similarityMatrix)
        return components

if __name__ == "__main__":
    m = model()
    data  = pd.read_csv('example.csv')
    m.setData(data)
    m.preprocessData()
    m.generateSimilarityMatrix()
    m.generateClusters()