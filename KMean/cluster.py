import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import KMeans
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
df.drop(['name', 'body', 'home_dest', 'ticket', 'cabin', 'fare', 'embarked', 'boat'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def change_to_numeric(df):
    change_to_numeric = []
    for h in df.columns.values:
        if df[h].dtype != np.int64 and df[h].dtype != np.float64:
            change_to_numeric.append(h)

    for i in change_to_numeric:
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

clf = KMeans(n_clusters=2)
clf.fit(X)

corrects = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype('float'))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        corrects += 1

accuracy = corrects/len(y)

if accuracy > (1-accuracy):
    print(accuracy)
else:
    print(1-accuracy)



