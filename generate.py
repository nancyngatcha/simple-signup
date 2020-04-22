import numpy as np
import pandas as pd
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from collections import Counter
from numpy.random import choice

size_table = 10000
E = np.random.randint(2, size=size_table)
N = np.random.randint(2, size=size_table)
F = np.random.randint(2, size=size_table)
P = np.random.randint(2, size=size_table)


dessin = np.random.randint(2, size=size_table)
jeux_videos = np.random.randint(2, size=size_table)
photographie = np.random.randint(2, size=size_table)
voyage = np.random.randint(2, size=size_table)
musique = np.random.randint(2, size=size_table)



np.random.random_sample((5,))
E = np.random.random_sample((size_table,))
N = np.random.random_sample((size_table,))
F = np.random.random_sample((size_table,))
P = np.random.random_sample((size_table,))


formation = ["Montage","Cameraman","Ingénieur du son","Directeur de production","Responsable artistique", "publicité","design industriel","modélisme",
"photographie", "Graphiste","Animation 3D","Webdesigner","Developpement informatique"]

draw = choice(formation, number_of_items_to_pick,
              p=probability_distribution)

random.choices(population=[['a','b'], ['b','a'], ['c','b']],weights=[0.2, 0.2, 0.6],k=10 )



Y = []

for i in range(size_table):
    if photographie[i] == 1:
        Y += random.choices(["photographie", "Cameraman", "Montage"],weights=[0.2, 0.2, 0.6])
    elif jeux_videos[i] == 1:
        Y += random.choices(["Animateur 3D", "Graphiste", "Webdesigner","Developpement informatique"])
    elif dessin[i] == 1:
        Y += random.choices(["Responsable artistique", "publicité", "modélisme", "Graphiste"])
    elif musique[i] == 1:
        Y += ["Ingénieur du son"]
    else :
        Y += ["Directeur de production"]
    

wine = datasets.load_wine()
df = pd.DataFrame({'Extroversion': E, 'Intuition': N, 'Feeling': F, "Perception" : P, "Dessin" : dessin, "Musique" : musique, "Photographie" : photographie, 'Voyage' : voyage,'Jeux Videos' : jeux_videos, 'Formation' : np.array(Y)})

X = df.drop(columns=['Formation'])
y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#Prediction
y_pred = knn.predict(X_test)
predicted= knn.predict([[0,1,1,1,0,1,0,0,1]]) 
print(predicted)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

profil = [0,1,1,1]
compt = 0
dic = {}
formation_ = []

for i in range (size_table):
    if [df.Extroversion[i], df.Intuition[i], df.Feeling[i], df.Perception[i]] == profil:
        compt +=1
        formation_ += [df.Formation[i]]

dic = Counter(formation_)

"""print(compt)
print(formation_)
print(len(formation_))
print(Counter(formation_))"""

dic = Counter(formation_)
perc_audioV = 0
perc_design = 0
perc_JV = 0
perc_dev = 0
for k, v in dic.items(): 
    dic[k] =  round(v/compt,3)
    if k in ["Montage","Cameraman","Ingénieur du son","Directeur de production","Responsable artistique"]:
        perc_audioV += round(v/compt,3)
    if k in ["publicité","design industriel","modélisme","photographie"]:
        perc_design += round(v/compt,3)
    if k in ["Graphiste","Animation 3D"]:
        perc_JV += round(v/compt,3)
    if k in ["Webdesigner","Developpement informatique"]:
        perc_dev += round(v/compt,3)

print(dic)
print(perc_audioV,perc_design, perc_JV,perc_dev)
print(perc_audioV+perc_design+perc_JV+perc_dev)

"""df['Formation'].value_counts()
df[df["Feeling"]==1].value_counts()"""


"""#AUDIOVISUEL
Montage
Cameraman
Ingénieur du son
Directeur de production
Responsable artistique 


#DESIGN
publicité
design industriel
modélisme
photographie


#JEUX VIDEOS
Graphiste
Animation 3D

#DEV
Webdesign
Developpement informatique
"""