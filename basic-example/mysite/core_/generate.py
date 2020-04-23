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

def generate_dataset(size_table):
    E = np.random.randint(2, size=size_table)
    N = np.random.randint(2, size=size_table)
    F = np.random.randint(2, size=size_table)
    P = np.random.randint(2, size=size_table)


    dessin = np.random.randint(2, size=size_table)
    jeux_videos = np.random.randint(2, size=size_table)
    photographie = np.random.randint(2, size=size_table)
    voyage = np.random.randint(2, size=size_table)
    musique = np.random.randint(2, size=size_table)



    """np.random.random_sample((5,))
    E = np.random.random_sample((size_table,))
    N = np.random.random_sample((size_table,))
    F = np.random.random_sample((size_table,))
    P = np.random.random_sample((size_table,))"""


    formation = ["Montage","Cameraman","Ingénieur du son","Directeur de production","Responsable artistique", "publicité","design industriel","modélisme",
    "photographie", "Graphiste","Animation 3D","Webdesigner","Developpement informatique"]

    """draw = choice(formation, number_of_items_to_pick,
                p=probability_distribution)

    random.choices(population=[['a','b'], ['b','a'], ['c','b']],weights=[0.2, 0.2, 0.6],k=10 )

    """

    Y = []

    for i in range(size_table):
        if photographie[i] == 1:
            Y += random.choices(["photographie", "Cameraman", "Montage"])
        elif jeux_videos[i] == 1:
            Y += random.choices(["Animateur 3D", "Graphiste", "Webdesigner","Developpement informatique"])
        elif dessin[i] == 1:
            Y += random.choices(["Responsable artistique", "publicité", "modélisme", "Graphiste"])
        elif musique[i] == 1:
            Y += ["Ingénieur du son"]
        else :
            Y += ["Directeur de production"]
        

    """wine = datasets.load_wine()"""
    df = pd.DataFrame({'Extroversion': E, 'Intuition': N, 'Feeling': F, "Perception" : P, "Dessin" : dessin, "Musique" : musique, "Photographie" : photographie, 'Voyage' : voyage,'Jeux Videos' : jeux_videos, 'Formation' : np.array(Y)})
    return df

#profil_pred = [0,1,1,1,0,1,0,0,1]
def predict(df, profil):
    y = np.array(df.Formation)
    X = df.drop(columns=['Formation'])
    print(len(y))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 4)
    # Fit the classifier to the data
    knn.fit(X_train,y_train)
    #Prediction
    y_pred = knn.predict(X_test)
    if (len(profil) == 4):
        
        profil += [1,0,1,1,0]
    predicted= knn.predict([profil]) 
    print(predicted[0])

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    return predicted[0]


def stats(df, profil):
    #profil = [0,1,1,1]
    profil = profil[0:4] #on ne récupère que la partie MBTI du profil
    compt = 0
    dic = {}
    formation_profil_sim = [] #liste des formations choisies par les profils mbti similaires

    for i in range (df.shape[0]):
        if [df.Extroversion[i], df.Intuition[i], df.Feeling[i], df.Perception[i]] == profil:
            compt +=1
            formation_profil_sim += [df.Formation[i]]

    dic = Counter(formation_profil_sim) # nombres de profils similaires par formation


    perc_audioV = 0 # pourcentage audio visuel
    perc_design = 0 # pourcentage design
    perc_JV = 0 # pourcentage Jeux vidéos
    perc_dev = 0 # pourcentage developpeur

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

    #print(dic)
    print(round(perc_audioV,3),round(perc_design,3), round(perc_JV,3) ,round(perc_dev,3))
    print(round(perc_audioV+perc_design+perc_JV+perc_dev,3))

    return dic


def machine_learning(profil):
    size_table = 10000
    print("aaaaaa")
    df = generate_dataset(size_table)
    print("cccccc")
    stats_profil_similaire = stats(df, profil)
    print("bbbbb")
    formation = predict(df, profil)
    print("dddddd")

    return formation, stats_profil_similaire



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