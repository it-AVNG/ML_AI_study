import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def create_model():
    iris_dataset = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=1
    )

    knn = KNeighborsClassifier(n_neighbors=1)
    predict_result = 0

    while (predict_result<0.97):
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        predict_result = np.mean(y_pred==y_test)

    file_name = 'check_iris_model.pkl'
    pickle.dump(knn, open(file_name, 'wb'))

if __name__ == '__main__':
    create_model()


