import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from tensorflow.python.keras.engine.sequential import Sequential


class DealingWithData:
    def __init__(self, df_data):
        self.df_data = df_data
        pass

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

    def scale_and_split_Xy(self, y_cols_idx: list, test_size=0.25) -> dict:
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


class EnergyOrWeather(DealingWithData):
    def __init__(self, df_data, y_cols_idx: list):
        DealingWithData.__init__(self, df_data=df_data)
        self.count_na()
        self.fillna()
        # Train Data
        (
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
        ) = super().scale_and_split_Xy(y_cols_idx)
        self.model = self.seq_lstm(y_cols_idx=y_cols_idx)
        # self.transform_for_lstm()
        # self.x_scaler and
        # y_scaler, model
        pass

    #     def lstm(self,y_cols_idx = [-1]):
    #         import numpy as np
    #         from tensorflow import keras
    #         from tensorflow.keras import layers

    #         max_features = 20000  # Only consider the top 20k words
    #         maxlen = 200  #

    #         # Input for variable-length sequences of integers
    #         inputs = keras.Input(shape=(5,), dtype="int32")
    #         # Embed each integer in a 128-dimensional vector
    #         x = layers.Embedding(max_features, 128)(inputs)
    #         # Add 2 bidirectional LSTMs
    #         x = layers.LSTM(64, return_sequences=True)(x)
    #         x = layers.LSTM(64, return_sequences=True)(x)
    #         x = layers.LSTM(64)(x)
    #         # Add a classifier
    #         outputs = layers.Dense(len(self.y_cols), activation="sigmoid")(x)
    #         self.model = keras.Model(inputs, outputs)
    #         print(self.model.summary())

    # #         model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    # #         model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_test, y_test))
    #         return self.model
    #         pass

    @staticmethod
    def init_function_model(energy_shape: tuple, weather_shape):
        """
        Returns a layers for lstm [energy, weather] as -> [x,y]
        """
        from tensorflow.keras import Sequential, Input
        from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Concatenate

        max_features = 30000
        x_inputs = Input(shape=energy_shape, dtype="int32")
        # Embed each integer in a 128-dimensional vector
        x = Embedding(max_features, 128)(x_inputs)
        # Add 2 bidirectional LSTMs
        x = LSTM(64, activation="relu", return_sequences=True)(x)
        x = LSTM(64, activation="relu", return_sequences=True)(x)
        x = LSTM(64, activation="relu", return_sequences=True)(x)
        x = LSTM(64, activation="relu", return_sequences=True)(x)

        y_inputs = Input(shape=weather_shape, dtype="int32")
        y = Embedding(max_features, 128)(y_inputs)
        # Add 2 bidirectional LSTMs

        y = LSTM(64, activation="relu", return_sequences=True)(y)
        y = LSTM(64, activation="relu", return_sequences=True)(y)
        y = LSTM(64, activation="relu", return_sequences=True)(y)
        y = LSTM(64, activation="relu", return_sequences=True)(y)

        concatted = Concatenate()([x, y])

        z = LSTM(64, activation="relu", return_sequences=True)(concatted)
        z = LSTM(64, activation="relu", return_sequences=True)(z)
        z = LSTM(64, activation="relu", return_sequences=False)(z)
        z = LSTM(units=120, activation="tanh")(z)

        return z

    def seq_lstm(self, y_cols_idx=[-1]):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout

        len_outputs = len(self.y_cols)
        len_features = len(self.x_cols)

        regressior = Sequential()

        regressior.add(
            LSTM(
                units=60,
                activation="relu",
                return_sequences=True,
                input_shape=(self.x_train.shape[1], len_features),
            )
        )
        regressior.add(Dropout(0.2))

        regressior.add(LSTM(units=120, activation="relu", return_sequences=True))
        regressior.add(Dropout(0.2))

        regressior.add(LSTM(units=240, activation="relu", return_sequences=True))
        regressior.add(Dropout(0.2))

        regressior.add(LSTM(units=240, activation="relu", return_sequences=True))
        regressior.add(Dropout(0.2))

        regressior.add(LSTM(units=120, activation="tanh"))
        regressior.add(Dropout(0.2))

        regressior.add(Dense(units=len_outputs))

        self.model = regressior
        # self.save("my_model")
        # print(self.model.summary())
        # print("model done")
        return regressior
        pass

    import numpy as np

    @staticmethod
    def lstm_data_transform(x_data, y_data, num_steps=5):
        X, y = [], []
        # take num_steps element and stack then sequentially one after the other.
        for i in range(x_data.shape[0]):
            end_ix = i + num_steps
            # end of dataset
            if end_ix >= x_data.shape[0]:
                break
            # slice num_steps portion
            seq_X = x_data[i:end_ix]
            seq_y = y_data[end_ix]
            # stack them
            X.append(seq_X)
            y.append(seq_y)

        x_array = np.array(X)
        y_array = np.array(y)
        return x_array, y_array

    @staticmethod
    def lstm_data(x_data, y_data, num_steps):
        import numpy as np

        X_train = []
        y_train = []
        for i in range(num_steps, x_data.shape[0]):
            X_train.append(x_data[i - num_steps : i])
            y_train.append(y_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        return X_train, y_train

    def train(self, epochs=5, params=dict(name="lstm", num_steps=24)):
        import tensorflow as tf

        """
        params
        ======
        -Lstm -> dict(name, num_steps)
        """
        if params["name"] == "lstm":
            # Prepare data
            x_train, y_train = self.lstm_data(
                self.x_train, self.y_train, num_steps=params["num_steps"]
            )
            x_test, y_test = self.lstm_data(
                self.x_test, self.y_test, num_steps=params["num_steps"]
            )
            # Train model
            self.model.compile(
                optimizer="adam",
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()],
            )
            self.history = self.model.fit(
                x_train,
                y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(x_test, y_test),
            )
        else:
            pass

        # self.ShowEvaluationGraph()
        return self.history

    def predict_test(self, params=dict(name="lstm", num_steps=24)):
        x_test, y_test = self.lstm_data(
            self.x_test, self.y_test, num_steps=params["num_steps"]
        )
        y_pred = self.model.predict(x_test)

        return dict(y_test=y_test, y_pred=pred)

    def ShowEvaluationGraph(self):
        import plotly.express as px

        df = pd.DataFrame(self.history)
        fig = px.line(df, x="epoch", y="loss", color="country")
        fig.show()
        pass


class weather(DealingWithData):
    def __init__(self):
        df_data = pd.read_csv(
            "/kaggle/input/energy-consumption-generation-prices-and-weather/weather_features.csv"
        )
        DealingWithData.__init__(self, df_data=df_data)
        self.fillna()
        pass