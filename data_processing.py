import pandas as pd


class DealingWithData:
    def __init__(self, df_data: pd.DataFrame):
        self.df_data = df_data.copy()
        self.correct_dates()
        self.drop_objects_col()
        self.fillna()
        pass

    def clean_xy(self, x_col_idx=None, y_cols_idx=[-1]):
        X = self.df_data.copy().drop("time", axis=1)

        # Include all remaining cols
        self.y_cols = [X.columns[idx] for idx in y_cols_idx]
        y = X[self.y_cols]
        if x_col_idx is None:
            X = X.drop(self.y_cols, axis="columns")
        else:
            self.x_cols = [X.columns[idx] for idx in x_col_idx]
            X = X[self.x_cols]

        self.x_cols = X.columns
        print(
            f"""
        X data columns: {self.x_cols}\n
        y data colunms: {self.y_cols}\n
        """
        )

        return X.values, y.values

    def MlInputData(self, x_col_idx=None, y_cols_idx=[-1], test_size=0.25):
        from sklearn.model_selection import train_test_split

        X, y = self.clean_xy(x_col_idx=x_col_idx, y_cols_idx=y_cols_idx)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return [x_train, x_test, y_train, y_test]

    def count_na(self):
        return dict(NANs=self.df_data.isna().sum())

    def fillna(self):
        """
        2 of the colums have equal number of nan's
        # TODO : find corrolation
        """
        self.df_data.fillna(value=0, inplace=True)
        pass

    def correct_dates(self):
        self.df_data["time"] = pd.to_datetime(self.df_data["time"], utc=True)
        pass

    def drop_objects_col(self):
        for col in self.df_data.columns:
            if self.df_data[col].dtype == str("object"):
                # print(typ)
                self.df_data = self.df_data.drop(typ, axis=1)

    """
    visualization should have its own class.
    # TODO : decouple.
    """

    def plot_dist(self, x_plot="time", y_plot="generation biomass"):
        self.__viz.plot_distribution(self, x_plot=x_plot, y_plot=y_plot)
        pass

    def plot_dist_by_idx(self, idx_x=0, idx_y=1):
        # same as
        x_plot = self.df_data.columns[idx_x]
        y_plot = self.df_data.columns[idx_y]
        return self.__viz.plot_distribution(self, x_plot=x_plot, y_plot=y_plot)
        pass

    def scale_and_split_Xy(self, y_cols_idx=[-1], test_size=0.25) -> dict:
        # Extract.
        X = self.df_data.copy().drop("time", axis=1)
        self.y_cols = [X.columns[idx] for idx in y_cols_idx]
        y = X[self.y_cols]
        X = X.drop(self.y_cols, axis="columns")
        self.x_cols = X.columns
        print(
            f"""
        X data columns: {self.x_cols}\n
        y data colunms: {self.y_cols}\n
        """
        )
        from sklearn.preprocessing import MinMaxScaler

        # Scale data
        self.x_scaler = MinMaxScaler()
        self.x_scaler.fit(X)

        X = self.x_scaler.transform(X)

        self.y_scaler = MinMaxScaler()
        self.y_scaler.fit(y)
        y = self.y_scaler.transform(y)

        # split data
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return [x_train, y_train, x_test, y_test]

    class __viz:
        def plot_distribution(self, x_plot, y_plot):
            df_plot = dict(
                x=[k for k in range(self.df_data[x_plot].count())],
                y=self.df_data[y_plot],
            )
            fig = px.scatter(
                df_plot,
                x="x",
                y="y",
                color="y",
                marginal_y="violin",
                marginal_x="box",
                trendline="ols",
                template="simple_white",
            )

            return fig
            # fig.show()

        def plt_model_scater(self, x, y, y_list=[], params={}):
            plt.figure()
            plt.scatter(x, y, color="red")
            for y in y_list:
                plt.plot(x, y, color="blue")
            # plt.title("Generation Fossil Hard Coal vs Price Day Ahead")
            # plt.xlabel("Generation Fossil Hard Coal")
            # plt.ylabel("Price Day Ahead")
            return plt