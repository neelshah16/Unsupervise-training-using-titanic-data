import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic
# parch	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# Variable Notes
#
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
#
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
#
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
#
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# all the features pclass,survived,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,home_dest


df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['name', 'body', 'home_dest', 'ticket', 'cabin', 'fare', 'embarked', 'boat'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def change_to_numeric(df):
    columns_to_change = []
    for h in df.columns.values:
        if df[h].dtype != np.int64 and df[h].dtype != np.float64:
            columns_to_change.append(h)

    for i in columns_to_change:
        category = set(df[i])
        k = 0
        category_new = {}
        for j in category:
            category_new[j] = k
            k+=1

        df[i] = df[i].map(category_new)
    return df

df = change_to_numeric(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_center = clf.cluster_centers_

original_df['cluster_group'] = labels

for i in range(len(cluster_center)):
    survived = original_df[(original_df['cluster_group'] == float(i))]
    males = survived[(survived['sex'] == float(0))]
    age = survived[(survived['age'] > float(25))]
    survival = survived[(survived['survived'] == float(1))]
    print(len(males)/len(survived))
    print(len(age)/len(survived))
    print(len(survival)/len(survived))
    print(" --- --- --- ")

