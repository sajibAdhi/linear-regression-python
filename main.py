import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == '__main__':
    df = pd.read_csv("salaries.csv")

    x = df.iloc[:, :-1].values  # get all rows with all columns except the last one #
    y = df.iloc[:, -1].values  # get all rows with only the last column #

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2} ({r2:.3%})")

    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred, color="yellow")

    plt.show()
