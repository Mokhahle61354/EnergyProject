import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class MlModels:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def feature_selection(self, n_estimators=50):
        from sklearn.model_selection import train_test_split

        x, x1, y, y1 = train_test_split(self.x_train, self.y_train)
        clf = ExtraTreesClassifier(n_estimators=n_estimators)
        clf = clf.fit(x, y)
        importances = clf.feature_importances_
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(x)
        return X_new, importances

    def LinearRegression(self):
        regressor = LinearRegression()
        regressor.fit(self.x_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        msqe = (
            sum((y_pred - self.y_test) * (y_pred - self.y_test)) / self.y_test.shape[0]
        )
        rmse = np.sqrt(msqe)
        txt = f"""
        A simple linear regression was used for this prediction problem. Firstly, we trained our model without splitting our data and we achieved impressive results. Then we splitted our data into a training set and a test set to avoid overfitting and to obtain a realistic evaluation of our learned model. By doing this we got relatively good results.
        Simple Linear Regression for generation fossil hard coal and price day ahead\n

        ```
        RMSE:
            - {rmse}
        MSQE:
            - {msqe}
        ```

        """
        return dict(y_test=self.y_test, y_pred=y_pred, rmse=rmse, msqe=msqe, txt=txt)

    def RandomForest_sx(self, n_estimators=100, random_state=0):
        regressor = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        regressor.fit(self.x_test, self.y_test)
        y_pred = regressor.predict(self.x_test)
        # sort x
        msqe = (
            sum((y_pred - self.y_test) * (y_pred - self.y_test)) / self.y_test.shape[0]
        )
        rmse = np.sqrt(msqe)
        s_x = np.sort(self.x_test, axis=None).reshape(-1, 1)
        txt = f"""
        RamdomForest\n
        n_estimators={n_estimators}\n
        random_state={random_state}\n

        ```
        RMSE:\n
            - {rmse}\n
        MSQE:\n
            - {msqe}\n
        ```

        """
        return dict(
            y_test=self.y_test, y_pred=y_pred, msqe=msqe, rmse=rmse, s_x=s_x, txt=txt
        )
