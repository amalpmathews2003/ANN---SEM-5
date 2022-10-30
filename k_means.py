import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def KMeans(df, k, iter):
    data = df.iloc[:, [1, 2, 3, 4, 5]].values
    centroid = []
    clusters = {}

    for i in range(k):
        clusters[i] = []
        centroid.append(data[i][:4])

    while iter > 0:
        iter -= 1
        for i in range(k):
            clusters[i] = []

        for row in data:
            dist = []
            for i in range(k):
                norm = np.linalg.norm(row[:4]-centroid[i][:4])
                dist.append(norm)

            clusters[dist.index(min(dist))].append(list(row))

        for i in range(k):
            temp = list(map(lambda x: x[:4], clusters[i]))
            centroid[i] = np.average(temp, axis=0)

    return clusters, centroid


if __name__ == '__main__':

    df = pd.read_csv('Iris.csv')
    k = 3
    iter = 10
    clusters, centroid = KMeans(df, k, iter)

    for i in range(k):
        df = pd.DataFrame(clusters[i])
        plt.scatter(df[0], df[1])

    plt.show()

   
