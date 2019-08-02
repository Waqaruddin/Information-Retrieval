from sklearn.datasets import load_svmlight_file
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as pyplot
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings('ignore')
print ("Please wait while the values are computed:")
feature_vectors3, targets3 = load_svmlight_file("trainingdatafileTFIDF.txt")
numberOfClusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20, 21, 22, 23, 24, 25]
kMeansSilhouette = []
kMeansMutualInformation = []
agglomerativeClusteringSilhoutte = []
agglomerativeClusteringMutualInformation = []
topHundredFeatures = SelectKBest(mutual_info_classif,
k=100).fit_transform(feature_vectors3, targets3)
topHundredFeatures = topHundredFeatures.toarray()
for i in numberOfClusters:
 kmeans_model = KMeans(n_clusters=i).fit(topHundredFeatures)
 clustering_labels = kmeans_model.labels_
 silhouettescore = metrics.silhouette_score(topHundredFeatures,
clustering_labels, metric='euclidean')
 mutualInformationscore = metrics.normalized_mutual_info_score(targets3,
clustering_labels)
 kMeansSilhouette.append(silhouettescore)
 kMeansMutualInformation.append(mutualInformationscore)
#for i in numberOfClusters:
 single_linkage_model = AgglomerativeClustering(n_clusters=i,
linkage='ward').fit(topHundredFeatures)
 clustering_labels2 = single_linkage_model.labels_
 silhouettescore2 = metrics.silhouette_score(topHundredFeatures,
clustering_labels2, metric='euclidean')
 mutualInformationscore2 = metrics.normalized_mutual_info_score(targets3,
clustering_labels2)
 agglomerativeClusteringSilhoutte.append(silhouettescore2)
 agglomerativeClusteringMutualInformation.append(mutualInformationscore2)
pyplot.figure(figsize=(9,9))
pyplot.plot(numberOfClusters, kMeansSilhouette, label = "k-Means Silhouette Score")
pyplot.plot(numberOfClusters, agglomerativeClusteringSilhoutte, label =
"Agglomerative Clustering Silhouette Score")
pyplot.xlabel("Number Of Clusters")
pyplot.ylabel("K-Means & Agglomerative Clustering Silhouette scores")
pyplot.legend(loc = 'best')
pyplot.show()
pyplot.figure(figsize=(9,9))
pyplot.plot(numberOfClusters, kMeansMutualInformation, label = "k-Means MutualInformation Score")
pyplot.plot(numberOfClusters, agglomerativeClusteringMutualInformation, label =
"Agglomerative Clustering MutualInformation Score")
pyplot.xlabel("Number Of Clusters")
pyplot.ylabel("K-Means & Agglomerative Clustering MutualInformation Scores")
pyplot.legend(loc = 'best')
pyplot.show()
print("---Plot graph finish---")