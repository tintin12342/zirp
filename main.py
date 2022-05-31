import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from c_index import calc_c_index

from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans, MiniBatchKMeans


def write_results(text):
    with open('results.txt', 'a') as f:
        f.writelines('\n'.join(text))


def write_all_results(text):
    with open('k_means_results.txt', 'a') as f:
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


class ClusteringAlgorithms:

    def __init__(self, data):
        self.data = data

        self.X = self.data.loc[:, self.data.columns != 'CO2 Ratings']
        y = self.data['CO2 Ratings']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=0.20)

    def k_means(self):
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(self.X_train)
        labels = kmeans.labels_
        c_index_input = np.array(self.X_train).astype(float)

        write_results(['\nK means',
                       'Silhouette: ' + str(metrics.silhouette_score(self.X_train, labels, metric='euclidean')),
                       'Xalinski Harabasz: ' + str(metrics.calinski_harabasz_score(self.X_train, labels)),
                       'Davies Bouldin: ' + str(davies_bouldin_score(self.X_train, labels)),
                       'C-index: ' + str(calc_c_index(c_index_input, labels))])

    def k_means_all_data(self):
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(self.X)
        labels = kmeans.labels_
        c_index_input = np.array(self.X).astype(float)

        write_all_results(['\nK means all data',
                           'Silhouette: ' + str(metrics.silhouette_score(self.X, labels, metric='euclidean')),
                           'Xalinski Harabasz: ' + str(metrics.calinski_harabasz_score(self.X, labels)),
                           'Davies Bouldin: ' + str(davies_bouldin_score(self.X, labels)),
                           'C-index: ' + str(calc_c_index(c_index_input, labels))])

    def mini_batch_kmeans(self):
        kmeans = MiniBatchKMeans(n_clusters=4)
        kmeans.fit(self.X_train)
        labels = kmeans.labels_
        c_index_input = np.array(self.X_train).astype(float)

        write_results(['\n\nMini batch K means',
                       'Silhouette: ' + str(metrics.silhouette_score(self.X_train, labels, metric='euclidean')),
                       'Xalinski Harabasz: ' + str(metrics.calinski_harabasz_score(self.X_train, labels)),
                       'Davies Bouldin: ' + str(davies_bouldin_score(self.X_train, labels)),
                       'C-index: ' + str(calc_c_index(c_index_input, labels))])

    def agglomerative_clustering(self):
        ac = AgglomerativeClustering(n_clusters=4)
        ac.fit(self.X_train)
        labels = ac.labels_
        c_index_input = np.array(self.X_train).astype(float)

        write_results(['\n\nAgglomerative clustering',
                       'Silhouette: ' + str(metrics.silhouette_score(self.X_train, labels, metric='euclidean')),
                       'Xalinski Harabasz: ' + str(metrics.calinski_harabasz_score(self.X_train, labels)),
                       'Davies Bouldin: ' + str(davies_bouldin_score(self.X_train, labels)),
                       'C-index: ' + str(calc_c_index(c_index_input, labels))])

    def ward_hierarchical_clustering(self):
        ward = AgglomerativeClustering(n_clusters=4, linkage="ward")
        ward.fit(self.X_train)
        labels = ward.labels_
        c_index_input = np.array(self.X_train).astype(float)

        write_results(['\n\nWard hierarchical clustering',
                       'Silhouette: ' + str(metrics.silhouette_score(self.X_train, labels, metric='euclidean')),
                       'Xalinski Harabasz: ' + str(metrics.calinski_harabasz_score(self.X_train, labels)),
                       'Davies Bouldin: ' + str(davies_bouldin_score(self.X_train, labels)),
                       'C-index: ' + str(calc_c_index(c_index_input, labels))])

    def bisect_means(self):
        bm = BisectingKMeans(n_clusters=4)
        bm.fit(self.X_train)
        labels = bm.labels_
        c_index_input = np.array(self.X_train).astype(float)

        write_results(['\n\nBisect means',
                       'Silhouette: ' + str(metrics.silhouette_score(self.X_train, labels, metric='euclidean')),
                       'Xalinski Harabasz: ' + str(metrics.calinski_harabasz_score(self.X_train, labels)),
                       'Davies Bouldin: ' + str(davies_bouldin_score(self.X_train, labels)),
                       'C-index: ' + str(calc_c_index(c_index_input, labels))])


if __name__ == '__main__':
    try:
        data = DataPreprocessing(r'C:\Users\tinva\Desktop\MY2022.csv').get_preprocessed_data()

        # determine = DetermineCluster(data)
        # determine.davies_bouldin_index(11)
        # determine.calinski_harabasz_index(11)
        # determine.silhouette_score(11)
        # determine.elbow_method(11)

        algorithm = ClusteringAlgorithms(data)
        algorithm.k_means_all_data()
        """algorithm.k_means()
        algorithm.mini_batch_kmeans()
        algorithm.ward_hierarchical_clustering()
        algorithm.agglomerative_clustering()
        algorithm.bisect_means()"""

    except Exception as err:
        print(err)
