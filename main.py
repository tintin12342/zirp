import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def write_results(title, text):
    with open('results.txt', 'w') as f:
        f.write(title + '\n' + '\n')
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


class KNN:

    def __init__(self, data):
        self.data = data

    def train_data(self):
        X = self.data.loc[:, self.data.columns != 'CO2 Ratings']
        y = self.data['CO2 Ratings']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        write_results('KNN', [str(confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))),
                              str(classification_report(y_test, y_pred, labels=np.unique(y_pred)))])

        self.data.insert(len(self.data.columns), "Prediction Results", knn.predict(X))

        return self.data


class Estimator:

    def __init__(self, data):
        self.data = data

    def elbow_method(self, rangeTo):
        distortions = []
        K = range(1, rangeTo)
        for k in K:
            kmean_model = KMeans(n_clusters=k)
            kmean_model.fit(self.data)
            distortions.append(kmean_model.inertia_)

        plt.figure(figsize=(10, 5))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def silhouette_score(self, n_clusters):
        X = self.data.loc[:, self.data.columns != 'CO2 Ratings']

        fig, (ax1) = plt.subplots(1)
        fig.set_size_inches(10, 5)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

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
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.show()


if __name__ == '__main__':
    try:
        data = DataPreprocessing(r'C:\Users\tinva\Desktop\MY2022.csv').get_preprocessed_data()

        estimator = Estimator(data)
        estimator.silhouette_score(10)
        estimator.elbow_method(10)

        knn = KNN(data)
        knn.train_data()
    except:
        print('Error while loading data')
