from visualizing import viz_cartisian
import streamlit as st
from functions import EnergyOrWeather
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb

# from graphviz import Digraph

import pandas as pd
import numpy as np

"""
Team Members
============
1735282 Kearabetswe Molaoa Tilo
1735294 Andreas Tsepang Motsie
1728085 Thabo Mokhahle\n

dataset url = https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather?select=energy_dataset.csv\n
## Title: Analysis of electrical consumption, in Spain
"""

chbx_show_intro = st.sidebar.checkbox("Introduction")
# chbx_miscellaneous = st.sidebar.checkbox("miscellaneous")
chbx_thabo = st.sidebar.checkbox("Thabo:")
chbx_tilo = st.sidebar.checkbox("Tilo:")
chbx_tshepang = st.sidebar.checkbox("Tshepang:")

# """
# Drop columns manually
# """
@st.cache(allow_output_mutation=True)
def LoadData() -> (pd.DataFrame, pd.DataFrame):
    df_data_energy = pd.read_csv("./data/energy_dataset.csv")
    df_data_weather = pd.read_csv("./data/weather_features.csv")
    return df_data_energy, df_data_weather


df_energy, df_weather = LoadData()
obj_energy = EnergyOrWeather(df_data=df_energy, y_cols_idx=[-1])
# y_drop_list = st.multiselect("drop which:", options=energy.df_data.columns)

# try:
#     energy.df_data = energy.df_data.drop(y_drop_list)
# except:
#     pass
# else:
#     pass

if chbx_show_intro:
    # energy = EnergyOrWeather(y_cols_idx=[-1])
    energy = obj_energy
    x_data, y_data = energy.x_train, energy.y_train
    # x,y = lstm_data_transform(x_data, y_data, num_steps=5)
    """
    ### DataFrame for Energy Dataset.
    """

    energy.df_data

    col_params, col_details = st.beta_columns(2)
    charts = st.beta_container()
    with col_params:
        """
        Select what to show:
        """
        rbtn_describe = st.radio("Stats", options=("describe", "NAN info", "plots"))

        if rbtn_describe == "describe":
            col_details.write(
                f"""
            dataset description:
            """
            )
            df_tmp = energy.df_data.describe()
            col_details.write(df_tmp)
            pass

        if rbtn_describe == "NAN info":
            col_details.write(energy.count_na()["NANs"])

            pass
        elif rbtn_describe == "plots":
            col_params.write("Select common axis:")
            x_axis_name = col_params.selectbox(
                "X axis:", options=energy.df_data.columns
            )
            y_axis_list = col_details.multiselect(
                "Select Y axis cols:", options=energy.df_data.columns
            )
            chart_name = col_details.selectbox(
                "Graph name", options=["distribution", "scatter", "line", "corrolation"]
            )

            if chart_name == "corrolation":
                # Energy heatmap
                corr = energy.df_data.corr()
                fig, ax = plt.subplots()
                ax = sb.heatmap(corr)
                # ax.set_ylim(0, 10)
                charts.write(fig)
                pass
            elif chart_name == "distribution":
                import plotly.figure_factory as ff

                dd = list(energy.df_data[y_axis_list].values.T)
                fig = ff.create_distplot(
                    dd,
                    y_axis_list,
                    bin_size=0.2,
                )

                # fig, ax = plt.subplots()
                # # y = y_axis_list.append("Time")
                # ax = energy.df_data[y_axis_list].plot.hist(
                #     figsize=(10, 8), bins=5, color="gray"
                # )
                charts.write(fig)

                pass

            elif chart_name == "scatter" or chart_name == "line":
                import plotly.express as px

                fig = px.line(energy.df_data, x=x_axis_name, y=y_axis_list)

                charts.write(fig)
                pass
        pass


if chbx_thabo:
    """
    LSTM model and visualization.
    =============================
    """
    from functions import DealingWithData

    import pandas as pd

    df_energy = pd.read_csv("./data/energy_dataset.csv")
    data_mutate = DealingWithData(df_data=df_energy)
    energy = EnergyOrWeather(df_energy, y_cols_idx=[-1])

    # plt = data_mutate.plot_dist_by_idx(idx_x=0)
    # st.write(plt)

    # """
    # ### Not iteractiveness not available due to.
    # """

    # Image(plot)
    col_seq_summary, col_seq_graph = st.beta_columns(2)

    with col_seq_summary:
        """
        ### Summary and Parameters
        LSTM regression:
        Sequence frame is on 24 samples.
        Total params: 1,089,721
        Trainable params: 1,089,721

        Matrice:\n
            - Mean square error: 0.0178
            - Root mean square error: 0.1332
            - Val_MSE = 0.0177
            - val_RMSE = 0.1332

        """

        img_model_graph = Image.open("./thabo/lstm_results.png")
        st.image(image=img_model_graph)

        """
        The above picture show 5 epochs of training.
        """

        inbx_lstm_frame = st.number_input(
            "lstm epoch frame",
            value=24,
            min_value=0,
            step=1,
            max_value=100,
            format="%i",
        )

        pass

    with col_seq_graph:
        img_model_graph = Image.open("./thabo/seq_model.png")
        st.image(image=img_model_graph)
        pass

    energy_lstm = EnergyOrWeather(df_data=df_energy, y_cols_idx=[-1])

    seq_lstm = energy_lstm.seq_lstm(y_cols_idx=[-1])

    plot = tf.keras.utils.plot_model(
        seq_lstm,
        to_file="./thabo/seq_model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    pass

    img_model_graph = Image.open("./thabo/func_model.png")
    st.image(image=img_model_graph)


if chbx_tilo:

    col_slt_dataset, col_slt_model = st.beta_columns(2)
    col_slt_x_inputs, col_slt_y_inputs = st.beta_columns(2)
    col_details, col_params = st.beta_columns(2)
    charts = st.beta_container()

    dataset_name = "energy"

    with col_slt_dataset:
        rbtn_dataset = st.radio(
            "Dataset",
            options=("energy", "weather"),
        )
        dataset_name = rbtn_dataset

        # add x_inputs, y_inputs multi select
    features_list = []
    y_list = []
    with col_params:
        x_axis_list = st.multiselect(
            "Select X axis cols:", options=obj_energy.df_data.columns
        )
        y_name = st.selectbox("Select Y axis cols:", options=obj_energy.df_data.columns)
        features_list = x_axis_list.append(y_name)
        f"""
        {y_name}
        """

    with col_slt_model:
        rbtn_model = st.radio(
            "ML model:",
            options=("LinearRegression", "RandomForest"),
        )
        from ml import MlModels
        from data_processing import DealingWithData

        obj_data = DealingWithData(df_data=df_energy)
        x_train, x_test, y_train, y_test = obj_data.MlInputData(x_col_idx=[-1])
        obj_models = MlModels(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
        )
        st.write(f"Selected dataset: {dataset_name}")

        if rbtn_model == "LinearRegression":
            results = obj_models.LinearRegression()
            # Write description or summary of the work
            col_details.write(results["txt"])
            # charts.plotly_chart(figure_or_data=)
            viz = viz_cartisian()
            lr_plot = viz.plt_model_scater(
                x=[idx for idx in range(results["y_test"].shape[0])],
                y1=results["y_test"],
                y2=results["y_pred"],
            )
            charts.write(lr_plot)

        elif rbtn_model == "RandomForest":
            results = obj_models.RandomForest_sx()
            # Write description or summary of the work
            col_details.write(results["txt"])
            # charts.plotly_chart(figure_or_data=)
            viz = viz_cartisian()
            lr_plot = viz.plt_model_scater(
                x=[idx for idx in range(results["y_test"].shape[0])],
                y1=results["y_test"],
                y2=results["y_pred"],
            )
            charts.write(lr_plot)
            pass
        elif rbtn_model == "FeatureSelection":
            """"""
            # results = obj_models.feature_selection()
            # # Write description or summary of the work
            # col_details.write(results["txt"])
            # charts.write(results)
            # charts.plotly_chart(figure_or_data=)
            pass

    """
                        End of Interactive parts
    """
    # x_axis_name = col_params.selectbox("X axis:", options=energy.df_data.columns)
    # y_axis_list = col_details.multiselect(
    #     "Select to train on:", options=energy.df_data.columns
    # )
    # chart_name = col_details.selectbox(
    #     "Graph name", options=["distribution", "scatter", "line"]
    # )
    # st.image(image=img_model_graph)

    img_model_graph = Image.open("./others/tilo1.png")
    st.image(image=img_model_graph)
    """
    1.Before splitting and training\n
        - Mean square error : 117.28459619
        - Root mean square error :10.8298013

    2.After splitting and training\n
        - Mean square error : 39.11486159
        - Root mean square error :6.25418752

    Random Forest for generation fossil hard coal and price day ahead

    """

    img_model_graph = Image.open("./others/tilo2.png")
    st.image(image=img_model_graph)

    """

    1.Before splitting and training\n
        - Mean square error : 81.87535854924823
        - Root mean square error :9.04850034808245
    2.After splitting and training\n
        - Mean square error : 102.0223056509064
        - Root mean square error :10.100609172268097

    ### Cross validation for estimating the maximum depth to yield maximum results. 
    """
    img_model_graph = Image.open("./others/tilo3.png")
    st.image(image=img_model_graph)
    """
    \nGeneration Solar and Forecast Solar Day Ahead

    Simple Linear Regression for generation solar and forecast solar day ahead

    """
    img_model_graph = Image.open("./others/tilo4.png")
    st.image(image=img_model_graph)
    """
    1.Before splitting and training\n
        - Mean square error : 68828.54394052
        - Root mean square error : 262.35194671

    2.After splitting and training\n
        - Mean square error : 22637.42901552
        - Root mean square error : 150.45739934

    Generation Fossil Hard Coal vs Price Actual

    Simple Linear Regression for generation fossil hard coal vs price actual

    """
    # img_model_graph = Image.open("./others/tilo4.png")
    # st.image(image=img_model_graph)
    """
    1.Before splitting and training\n
        - Mean square error : 158.00008834
        - Root mean square error : 12.5698086

    2.After splitting and training\n
        - Mean square error : 52.66496275
        - Root mean square error : 7.25706296

    Generation Solar and Temperature

    Decision tree for generation solar and temperature

    - accuracy : 24%

    Generation Fossil Gas and Price Actual

    Decision tree for generation fossil gas and price actual

    - Accuracy: 42%


    Generation Fossil Oil and Total Load Actual

    Random Forest for generation fossil oil and total load actual


    1.After splitting and training\n
        - Mean square error : 15173715.999576947
        - Root mean square error : 3895.3454275040804

    """
    pass
