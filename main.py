from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class DataPreprocessing:

    def __init__(self, file_path):
        self.file_path = file_path
        self.preprocessed_data = None

    def _data_processing(self):
        use_cols = ['Model', 'Vehicle Class', 'Engine Size', 'Cylinders', 'Transmission',
                    'Fuel', 'Fuel Consumption', 'Unnamed: 9', 'Unnamed: 10', 'CO2']
        df = pd.read_csv(self.file_path, sep=',', encoding='cp1252', low_memory=False, usecols=use_cols)
        df = df.rename(columns={'Unnamed: 9': 'Hwy Fuel Consumption', 'Unnamed: 10': 'Comb Fuel Consumption'})
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
        X = self.data.loc[:, self.data.columns != 'CO2']
        y = self.data['CO2']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        prediction_results = knn.predict(X)
        self.data.insert(len(self.data.columns), "Prediction Results", prediction_results)
        return self.data


if __name__ == '__main__':
    try:
        data = DataPreprocessing(r'C:\Users\tinva\Desktop\MY2022.csv')
        knn = KNN(data.get_preprocessed_data())
        print(knn.train_data())
    except:
        print('Error while loading data')
