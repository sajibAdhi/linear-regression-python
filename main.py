import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("salaries.csv")

    x = df.iloc[:, :-1].values  # get all rows with all columns except the last one #
    y = df.iloc[:, -1].values  # get all rows with only the last column #

    plt.scatter(x, y)
    plt.show()
