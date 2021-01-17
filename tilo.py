import matplotlib.pyplot as plt
import seaborn as sb

# Energy heatmap
corr = energy_dataset_df.corr()
plt.figure(figsize=(20, 10))
a = sb.heatmap(corr, annot=True, fmt=".2f")
a.set_ylim(0, 10)

# histograms
plt.figure(figsize=(10, 10))
energy_dataset_df[
    [
        "generation biomass",
        "generation fossil gas",
        "generation fossil hard coal",
        "generation fossil oil",
    ]
].hist(figsize=(10, 8), bins=5, color="gray")
plt.tight_layout()
plt.show()

# scatter plot
plt.figure()
plt.scatter(
    energy_dataset_df["generation fossil gas"], energy_dataset_df["price day ahead"]
)
plt.xlabel("Generation Fossil Gas")
plt.ylabel("Price Day Ahead")


# Generation Fossil Hard Coal vs Price Day Ahead

x = energy_dataset_df[
    "generation fossil hard coal"
].values  # get column Generation Fossil Hard Coal
y = energy_dataset_df["price day ahead"].values  # get column Price Day Ahead


x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# simple linear regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x, y)
y_predicted_price_day_ahead = regressor.predict(x)

plt.figure()
plt.scatter(x, y, color="red")
plt.plot(x, y_predicted_price_day_ahead, color="blue")
plt.title("Generation Fossil Hard Coal vs Price Day Ahead")
plt.xlabel("Generation Fossil Hard Coal")
plt.ylabel("Price Day Ahead")
plt.show()

# finding error
msqe = (
    sum((y_predicted_price_day_ahead - y) * (y_predicted_price_day_ahead - y))
    / y.shape[0]
)
rmse = np.sqrt(msqe)
# -------------- before training ----------------------------------------------
print(msqe)
print(rmse)


# split dataset into train and test splits
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1 / 3, random_state=0
)

# fit simple linear regression model on training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

# After training
print("Accuracy of a model")
print("*****************************************")
print("mean square error:")
print(msqe)
print("-----------------------------------------")
print("Root mean sqaure error:")
print(rmse)
# ----------------

# random forest regression
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x, y)

y_pred_Random_Forest = regressor.predict(x)

# sort x
s_x = np.sort(x, axis=None).reshape(-1, 1)

plt.figure()
x_grid = np.arange(min(s_x), max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title("Generation Fossil Hard Coal vs Price Day Ahead")
plt.xlabel("Generation Fossil Hard Coal")
plt.ylabel("Price Day Ahead")
plt.show()

# finding error
msqe = sum((y_pred_Random_Forest - y) * (y_pred_Random_Forest - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print(msqe)
print(rmse)


# -----------------------------------------------------------------------

# polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

# sort x
s_x = np.sort(x, axis=None).reshape(-1, 1)
s_y = lin_reg2.predict(poly_reg.fit_transform(s_x)
# visualize polynomial linear regression
plt.scatter(x, y, color="red")
plt.plot(s_x, s_y, color="blue")
plt.title("Generation Fossil Hard Coal vs Total Load Actual")
plt.xlabel("Generation Fossil Hard Coal")
plt.ylabel("Total Load Actual")
plt.show()