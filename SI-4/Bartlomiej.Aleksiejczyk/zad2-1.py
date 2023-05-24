import pandas as pd


def naj_reguła(df):
    reguły = []
    for column in df.columns[:-1]:
        for value in df[column].unique():
            reguła = (column, value)
            covered_examples = df[df[column] == value]
            class_distribution = covered_examples.iloc[:, -1].value_counts(normalize=True)
            accuracy = class_distribution.max()
            reguły.append((reguła, len(covered_examples), accuracy))
    naj = max(reguły, key=lambda r: (r[1], r[2]))
    return naj[0]


def sekwencyjne_pokr(df):
    reguly = []

    while len(df) > 0:
        reguła= naj_reguła(df)
        reguly.append(reguła)
        column, value = reguła
        df = df[df[column] != value]
    return reguly

data = pd.read_csv('Churn_Modelling.csv')
reguły = sekwencyjne_pokr(data)
for reguła in reguły:
    print(reguły)