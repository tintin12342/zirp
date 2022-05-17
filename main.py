import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
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

    def elbow_method(self):
        distortions = []
        K = range(1, 10)
        for k in K:
            kmean_model = KMeans(n_clusters=k)
            kmean_model.fit(self.data)
            distortions.append(kmean_model.inertia_)

        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()


if __name__ == '__main__':
    try:
        data = DataPreprocessing(r'C:\Users\tinva\Desktop\MY2022.csv')
        knn = KNN(data.get_preprocessed_data())
        knn.train_data()
        knn.elbow_method()
    except:
        print('Error while loading data')
