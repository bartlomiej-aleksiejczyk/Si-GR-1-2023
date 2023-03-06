import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

#Zadamie 2 odpowiedź
# Wybrany system "Car"
#Zadanie 3 odpowiedź:
with open("car.txt") as car:
    lines = [line.split() for line in car]
    attributes = ["c1","c2","c3","c4","c5","c6","cc7"]
    #zadanie 3a
    decission_classes=attributes[0:6]
    # decission_classes == lista klas decyzyjnych
    print(decission_classes)
    #['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    #3A: Lista symboli klas decyzyjnych =['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    #Zadanie 3b
    objects_in_classes = {}
    values_count_temp = []
    for index in range(7):
        for object_car in lines:
            values_count_temp.append(object_car[index])
        objects_in_classes[attributes[index]] = len(values_count_temp)
        values_count_temp = []
    print(objects_in_classes)
    #{'c1': 1728, 'c2': 1728, 'c3': 1728, 'c4': 1728, 'c5': 1728, 'c6': 1728, 'cc7': 1728}
    #Wielkość klas decyzyjnych: {'c1': 1728, 'c2': 1728, 'c3': 1728, 'c4': 1728, 'c5': 1728, 'c6': 1728, 'cc7': 1728}
    #Zadanie 3d
    available_values={}
    values_temp=[]
    for index in range (7):
        for object_car in lines:
            values_temp.append(object_car[index])
        values_temp_unique=set(values_temp)
        available_values[attributes[index]]=values_temp_unique
        values_temp = []
    print(available_values)
    #{'c1': {'vhigh', 'high', 'low', 'med'}, 'c2': {'vhigh', 'high', 'low', 'med'}, 'c3': {'4', '3', '2', '5more'}, 'c4': {'4', 'more', '2'}, 'c5': {'small', 'big', 'med'}, 'c6': {'high', 'low', 'med'}, 'cc7': {'acc', 'vgood', 'good', 'unacc'}}
    #3D: Lista różnych dostępnych wartości dla każdego atrybutu {'c1': {'vhigh', 'high', 'low', 'med'}, 'c2': {'vhigh', 'high', 'low', 'med'}, 'c3': {'4', '3', '2', '5more'}, 'c4': {'4', 'more', '2'}, 'c5': {'small', 'big', 'med'}, 'c6': {'high', 'low', 'med'}, 'cc7': {'acc', 'vgood', 'good', 'unacc'}}
    #zadanie 3e
    number_of_object_in_classes={}
    for attribute in attributes[0:7]:
        number_of_object_in_classes[attribute]=len(available_values[attribute])
    print(number_of_object_in_classes)
    # {'c1': 4, 'c2': 4, 'c3': 4, 'c4': 3, 'c5': 3, 'c6': 3, 'cc7': 4}
    #3B: Wielkości klas decyzyjnych:{'c1': 4, 'c2': 4, 'c3': 4, 'c4': 3, 'c5': 3, 'c6': 3, 'cc7': 4}
    #Podpunkty : c i f nie dotyczą wybranego przeze mnie systemu decyzyjnego gdyż system Cars wybrany przeze mnie zawiera same atrybuty symboliczne

    pd.set_option('display.max_columns', None)
    #Zad 4 A
    #Do rozwiązania tego zadania wybieram system german-credit
    data_frame = pd.read_csv("german-credit.txt", sep=" ", header=None)
    print(data_frame)
    with open("german-credit-type.txt") as german_credit_names:
        lista_nazw = [line.rstrip("\n") for line in german_credit_names]
        lista_nazw.append('21 s')
        data_frame.columns=lista_nazw
    print(data_frame)
    data_frame.describe()
    #Stworzenie słownika z najczęstszymi lub średnimi wartoścmi, odpowiednio dla symbolicznych i numerycznych atrybutów
    #przy okazji tworzona jest lista z kolumanmi numerycznymi potrzebnymi do dalszego zadania
    num_columns=[]
    column_dictionary={}
    for column in data_frame.columns:
        if column[-1]==('n'):
            column_dictionary[column]=round(pd.to_numeric((data_frame[column])).mean())
            num_columns.append(column)
        else:
            column_dictionary[column]=data_frame[column].mode()[0]
    # Generowanie 10% nieznanych
    for column in data_frame.columns:
        data_frame.loc[data_frame.sample(frac=0.1).index, column] = "?"
    #Tablica z 10% nieznanych
    print(data_frame)
    zliczacz=0
    #sprawdzenie czy faktycznie 10% jest nieznanych
    for column in data_frame.columns:
        zliczacz+=data_frame[column].value_counts()["?"]
    print(zliczacz)
    #Naprawianie znaków zapytania używając stworzonego słownika
    for column in column_dictionary:
        data_frame.loc[data_frame[column] == "?", column] = column_dictionary[column]
    #4A Poniżej wyświetlają się naprawione dane i poniżej słownik naprawczy
    print(data_frame)
    print(column_dictionary)
    #Zad 4 B d;a przedziału-1,1
    interval_1 = MinMaxScaler(feature_range=(-1, 1))
    data_norm_1 = pd.DataFrame(interval_1.fit_transform(data_frame[num_columns]))
    data_norm_1.columns=num_columns
    #wynik
    print(data_norm_1)
    #dla przedziału 0,1
    interval_2 = MinMaxScaler(feature_range=(0, 1))
    data_norm_2 = pd.DataFrame(interval_2.fit_transform(data_frame[num_columns]))
    data_norm_2.columns = num_columns
    #wynik:
    print(data_norm_2)
    #dla przedziału -10,10
    interval_3 = MinMaxScaler(feature_range=(-10, 10))
    data_norm_3 = pd.DataFrame(interval_3.fit_transform(data_frame[num_columns]))
    data_norm_3.columns = num_columns
    #wynik:
    print(data_norm_3)
    # Zadanie 4c dla wszystkich kolumn numerycznyh
    data_frame3=data_frame[num_columns]
    data_frame3=(data_frame3 - data_frame3.mean()) / data_frame3.std()
    # Odpowiedź ustandaryzowane wartości numeryczne:
    print(data_frame3)
    #Srawdzenie wyniku
    print(data_frame3.mean())
    print(data_frame3.std())
    #a2 n     1.003642e-16
    #a5 n     3.991252e-17
    # a8 n    -1.380111e-16
    # a11 n   -1.403322e-16
    # a13 n    6.794565e-17
    # a16 n   -2.275957e-16
    # a18 n    1.398881e-16
    # dtype: float64
    # a2 n     1.0
    # a5 n     1.0
    # a8 n     1.0
    # a11 n    1.0
    # a13 n    1.0
    # a16 n    1.0
    # a18 n    1.0
    # dtype: float64
    #Jak widać na powyższym rezulatcie wariance jest równa 1, natomiast średnia wartość jest bardzo bliska zeru
    # (python użył notacji naukowej, e-16 oznacza szesnaste miejsca po przecinku) W przypadku obliczeń średniej ten bardzo niski wynik może
    #mieć związek z błędami numerycznymi popełnionymi podczas interpretacji skryptu

    #Zadanie 4 d
    data_frame_churn = pd.read_csv("Churn_Modelling.csv")
    print(data_frame_churn)
    #Zmiana atrybutu symbolicznego na dummy
    data_frame_churn_dummy=pd.get_dummies(data_frame_churn, columns=["Geography"])
    #Widok danych przed usunięciem zbędnego atrybutu dummy
    print(data_frame_churn_dummy)
    #Usunięcie atrybutu dummy
    data_frame_churn_dummy = data_frame_churn_dummy.drop("Geography_Spain", axis=1)
    #Widok danych przed usunięciem zbędnego atrybutu dummy, finalna odpowiedź:
    print(data_frame_churn_dummy)












