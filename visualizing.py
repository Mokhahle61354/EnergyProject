from matplotlib import pyplot as plt
import plotly.express as px


class viz_cartisian:
    def __init__(self, df_data="") -> None:
        super().__init__()
        self.df_data = df_data

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

    def plt_model_scater(self, x, y1, y2, labels=dict(y1="y1", y2="y2")):
        fig, ax = plt.subplots()

        # if y_list == []:
        ax.scatter(x, y1, color="red")
        ax.scatter(x, y1, color="blue")
        # else:
        #     for y in y_list:
        # ax.plot(x, y, color="blue")
        #         pass
        plt.title(f"{labels['y1']} and {labels['y2']}")
        plt.xlabel("time")
        plt.ylabel("values")
        return fig