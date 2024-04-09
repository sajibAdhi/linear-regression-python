import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df = pd.read_csv("salaries.csv")

    x = df.iloc[:, :-1].values  # get all rows with all columns except the last one #
    y = df.iloc[:, -1].values  # get all rows with only the last column #

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    plt.scatter(x, y)
    plt.show()

    print(x_train.shape)
    print(x_test.shape)
