import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def preprocess_data(base_path: str):
     import numpy as np
     from sklearn import datasets
     from sklearn.model_selection import train_test_split
     from pathlib import Path

     path = Path(base_path) / 'preprocess'
     path.mkdir(parents=True, exist_ok=True)

     X, y = datasets.load_boston(return_X_y=True)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
     np.save(str(path / 'x_train.npy'), X_train)
     np.save(str(path / 'x_test.npy'), X_test)
     np.save(str(path / 'y_train.npy)', y_train)
     np.save(str(path / 'y_test.npy'), y_test)
     
if __name__ == '__main__':
     print('Preprocessing data...')
     _preprocess_data()
