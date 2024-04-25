import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == '__main__':
    df = pd.read_csv("salaries-cities.csv")

    x = df.iloc[:, :-1].values  # Taking all columns except the last one, i.e. TWO columns
    y = df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)

    salaries = model.predict([[11, 1], [11, 2], [12, 1], [12, 2]])
    print(salaries)
