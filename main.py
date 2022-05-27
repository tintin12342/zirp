import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score


def write_results(text):
    with open('results.txt', 'a') as f:
        f.writelines('\n'.join(text))


class DataPreprocessing:

    def __init__(self, file_path):
        self.file_path = file_path
        self.preprocessed_data = None

    def _data_processing(self):
        use_cols = ['Model', 'Vehicle Class', 'Engine Size', 'Cylinders', 'Transmission',
                    'Fuel', 'Fuel Consumption', 'Unnamed: 9', 'Unnamed: 10', 'CO2']
        df = pd.read_csv(self.file_path, sep=',', encoding='cp1252', low_memory=False, usecols=use_cols)

        df = df.rename(columns={'Unnamed: 9': 'Hwy Fuel Consumption',
                                'Unnamed: 10': 'Comb Fuel Consumption',
                                'CO2': 'CO2 Ratings'})
        df = df.dropna()

        self._factorize_data(df, ['Vehicle Class', 'Transmission', 'Fuel'])

        self.preprocessed_data = df

    def get_preprocessed_data(self):
        self._data_processing()
        return self.preprocessed_data

    @staticmethod
    def _factorize_data(df, column_names):
        for name in column_names:
            df[name] = pd.factorize(df[name])[0]


class ClusteringAlgorithms:

    def __init__(self, data):
        self.data = data

        X = self.data.loc[:, self.data.columns != 'CO2 Ratings']
        y = self.data['CO2 Ratings']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20)

    def k_means(self):
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)

        write_results(['\nK means',
                       'Rand index: ' + str(adjusted_rand_score(self.y_test, y_pred)),
                       'Homogeneity: ' + str(homogeneity_score(self.y_test, y_pred)),
                       'Completeness: ' + str(completeness_score(self.y_test, y_pred)),
                       'V-measure: ' + str(v_measure_score(self.y_test, y_pred))])

    def dbscan(self):
        dbscan = DBSCAN(eps=2.6, min_samples=3)
        dbscan.fit(self.X_train, self.y_train)
        y_pred = dbscan.fit_predict(self.X_test)

        write_results(['\n\nDBSCAN\n'
                       'Rand index: ' + str(adjusted_rand_score(self.y_test, y_pred)),
                       'Homogeneity: ' + str(homogeneity_score(self.y_test, y_pred)),
                       'Completeness: ' + str(completeness_score(self.y_test, y_pred)),
                       'V-measure: ' + str(v_measure_score(self.y_test, y_pred))])

    def optics(self):
        optics = OPTICS(min_samples=5, min_cluster_size=3)
        optics.fit(self.X_train, self.y_train)
        y_pred = optics.fit_predict(self.X_test)

        write_results(['\n\nOPTICS\n'
                       'Rand index: ' + str(adjusted_rand_score(self.y_test, y_pred)),
                       'Homogeneity: ' + str(homogeneity_score(self.y_test, y_pred)),
                       'Completeness: ' + str(completeness_score(self.y_test, y_pred)),
                       'V-measure: ' + str(v_measure_score(self.y_test, y_pred))])

    def agglomerative_clustering(self):
        ac = AgglomerativeClustering(n_clusters=6)
        ac.fit(self.X_train, self.y_train)
        y_pred = ac.fit_predict(self.X_test)

        write_results(['\n\nAgglomerative clustering\n'
                       'Rand index: ' + str(adjusted_rand_score(self.y_test, y_pred)),
                       'Homogeneity: ' + str(homogeneity_score(self.y_test, y_pred)),
                       'Completeness: ' + str(completeness_score(self.y_test, y_pred)),
                       'V-measure: ' + str(v_measure_score(self.y_test, y_pred))])

    def bisect_means(self):
        bm = BisectingKMeans(n_clusters=10)
        bm.fit(self.X_train, self.y_train)
        y_pred = bm.predict(self.X_test)

        write_results(['\n\nBisecting KMeans\n'
                       'Rand index: ' + str(adjusted_rand_score(self.y_test, y_pred)),
                       'Homogeneity: ' + str(homogeneity_score(self.y_test, y_pred)),
                       'Completeness: ' + str(completeness_score(self.y_test, y_pred)),
                       'V-measure: ' + str(v_measure_score(self.y_test, y_pred))])


class DetermineCluster:

    def __init__(self, data):
        self.data = data

    def elbow_method(self, rangeTo):
        distortions = []
        K = range(1, rangeTo)
        for k in K:
            kmean_model = KMeans(n_clusters=k)
            kmean_model.fit(self.data)
            distortions.append(kmean_model.inertia_)
            print(k, distortions)

        plt.figure(figsize=(10, 5))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def davies_bouldin_index(self, rangeTo):
        X = self.data.loc[:, self.data.columns != 'CO2 Ratings']
        K = range(2, rangeTo)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
            labels = kmeans.labels_
            print(k, davies_bouldin_score(X, labels))

    def calinski_harabasz_index(self, rangeTo):
        X = self.data.loc[:, self.data.columns != 'CO2 Ratings']
        K = range(2, rangeTo)

        for k in K:
            kmeans_model = KMeans(n_clusters=k).fit(X)
            labels = kmeans_model.labels_
            print(k, metrics.calinski_harabasz_score(X, labels))

    def silhouette_score(self, n_clusters):
        X = self.data.loc[:, self.data.columns != 'CO2 Ratings']

        fig, (ax1) = plt.subplots(1)
        fig.set_size_inches(10, 5)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.show()


if __name__ == '__main__':
    try:
        data = DataPreprocessing(r'C:\Users\tinva\Desktop\MY2022.csv').get_preprocessed_data()

        # determine = DetermineCluster(data)
        # determine.davies_bouldin_index(11)
        # determine.calinski_harabasz_index(11)
        # determine.silhouette_score(11)
        # determine.elbow_method(11)

        algorithm = ClusteringAlgorithms(data)
        algorithm.k_means()
        algorithm.dbscan()
        algorithm.optics()
        algorithm.agglomerative_clustering()
        algorithm.bisect_means()

    except:
        print('Error while loading data')
