import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.feature_selection import  RFECV
import joblib
import shap
from tqdm import tqdm
import base64 
from PIL import Image

import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
#!pip install factor_analyzer
#pip install factor_analyzer
# il faut installer factor_analyszer sur anaconda prompt (pip install factor_analyzer)
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

def sidebar_bg(side_bg):
   side_bg_ext = 'png'
   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,unsafe_allow_html=True,)


st.title("Prediction de la gravité d'un accident de la route")

#________________________________________________________Menu principal et Accueil________________________________________________________________________________________________________________________________________
page_selectionnee = st.sidebar.selectbox("Quelle page afficher ?", ["Accueil","Jeu de donnée","Classification binaire avec Logistic Regression","Prédire avec l'analyse factorielle exploratoire", "Classification binaire avec Random Forest",
                                                                    "Prédire avec un réseau de neurones","Conclusion"])



if page_selectionnee == "Accueil" :
    st.subheader("Projet sécurité routiere JUL 2023")
    st.subheader("Groupe : Hichem Ayadi / Justine Clement / Nicolas Leon / Eugene Edmond Tefong")
    st.write("")
    st.write("Objectif du projet : Prédire la gravité d'un accident en France.")
    st.write("Nos modèles ont été pensés dans l'optique d'un scénario d'aide des services publiques. Le but pourrait être de faciliter la décision d'envoi d'ambulance sur un lieu d'accident en fonction de ses caractéristiques.")

# ____________________________________________________Affichage du jeu de donnée _________________________________________________________________________________________________________________________
if page_selectionnee == "Jeu de donnée" :
    jeu_selectionne = st.sidebar.selectbox("Quel jeu de données afficher ?", ["Brut","Pré-traitement","Data Viz'"])
    noms_col= pd.read_csv('projet_DS\Streamlit\\001_df_processed.csv',nrows=1,).columns
    noms_col_raw= pd.read_csv('projet_DS\Streamlit\\000_df_RAW.csv',nrows=1,).columns
    if jeu_selectionne=="Brut":   
        st.subheader("Jeu de donnée brut")
        taille_nrows = 5
        alea=np.random.randint(0,1000000)         #pour avoir des valeurs aléatoire du csv
        st.subheader("Extrait du jeu de données")
        extrait_raw = pd.read_csv('projet_DS\Streamlit\\000_df_RAW.csv',nrows=taille_nrows,skiprows=alea,header=None)
        extrait_raw.columns = noms_col_raw
        st.write(extrait_raw)
        if st.button("Recharger un nouvel extrait"):
            alea=np.random.randint(0,1000000)
        st.write("Une ligne représente un usagers de la route pris dans un accident entre 2005 et 2021. \
                 Le jeu de donnée brut est obtenu à partir des données fusionnées de [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/).")
        st.write("Les jeu complet porte sur 2.5 millions d'usagers répartis sur 1 millions d'accidents. Les informations des accidents ont étés propagées sur celles des individus grâce au numéros d'accidents communs à chaque table. \
                 La concaténation avec les tables \"Usagers\" et \"Vehicules\" a nécessité de passer par une clé d'assemblage correspondant au numéro d'accident suivit du numéro de vehicule.")
        st.write("Le code utilisé est le suivant :")
        code="""#_Chargement des données concaténées de 2005 à 2021 :
lieux=pd.read_csv('000_lieux_RAW.csv',low_memory=False)
carac=pd.read_csv('000_carac_RAW.csv',low_memory=False)
vehi=pd.read_csv('000_vehi_RAW.csv',low_memory=False)
usagers=pd.read_csv('000_usagers_RAW.csv',low_memory=False)

#_Création d'une clé de merging temporaire dans usagers et vehi, 
#_qui prend le numero accident + numero véhicule
usagers['key']=usagers['Num_Acc'].astype(str)+usagers['num_veh']
vehi['key']=vehi['Num_Acc'].astype(str)+vehi['num_veh']

#_Merging progressif et drop des colonnes redondantes et de la clé
df=pd.merge(usagers,vehi.drop(columns=['Num_Acc',"num_veh","id_vehicule"]),
            on='key')
df=pd.merge(df.drop(columns="key"),carac,on='Num_Acc')
df=pd.merge(df,lieux,on='Num_Acc')
"""
        st.code(code)


    if jeu_selectionne=="Pré-traitement":
        taille_nrows = 5
        alea=np.random.randint(0,1000000)         #pour avoir des valeurs aléatoire du csv
        st.subheader("Extrait du jeu de données")
        extrait = pd.read_csv('projet_DS\Streamlit\\001_df_processed.csv',nrows=taille_nrows,skiprows=alea,header=None)
        extrait.columns = noms_col
        extrait['grav']=extrait['grav'].replace({1:0,2:1,3:1,4:0})
        st.write(extrait)
        if st.button("Recharger un nouvel extrait"):
            alea=np.random.randint(0,1000000)

        st.write("Les colonnes inexploitables ont été retirées, les NaN ont été traités pour chaque variable et les données catégorielles ont été encodées en one-hot.")

        code="""
#Dummification de catr__________________________________________
df['catr']=df['catr'].replace({1:"Nationnale/Autoroute",
                               2:'Nationnale/Autoroute',
                               3:'Departementale',
                               4:'Voix_Communale',
                               5:np.NaN,6:np.NaN,7:np.NaN,8:np.NaN,9:np.NaN})  
#Décidé suivant le value_count : 4=1169935, 3=826731, 1=229036, 2=213233, 
#les autres modalités représentents moins de 100k valeurs
df = pd.concat([df.drop(columns='catr'), 
                pd.get_dummies(df['catr'],prefix='catr')], axis=1)

#Correction de lum______________________________________________
df['lum']=df['lum'].replace({1:False,2:True,3:True,4:True,5:True})  
#Booléen de "Durant la Nuit ?"

#Correction de agg______________________________________________
df['agg']=df['agg'].replace({1:False,2:True})                    
#Booléen de "En Agglomération ?"
"""
        st.code(code)

# ___________________________________________DataViz __________________________________________________________________________________________________________________________________________
    if jeu_selectionne == "Data Viz'" :

        st.subheader("Valeurs manquantes dans les dataframes : ")
        st.image("projet_DS\Streamlit\\NAN.png")
        st.write("On remarque que les données géographiques ne sont renseignée qu'une foi sur deux.")
        st.write("")
        st.write("")

        st.subheader("Représentation géographique des accidents suivant la gravité : ")
        st.image("projet_DS\Streamlit\\france.png")
        st.write("Seuls les accidents dont les colonnes 'Lat' et 'Long' étaient renseignées sont représentés.")
        st.write("")
        st.write("")

        st.subheader("Histogramme de la gravité des blessures des usagers impliqués dans un accident : ")
        st.image("projet_DS\Streamlit\\Gravité histogramme.png")
        st.write("Le grand déséquilibre entre les classes est ce qui a motivé la classification binaire Léger / Grave de la cible pour la plupart de nos modèles.")
        st.write("")
        st.write("")

        st.subheader("Impact de l'année sur le nombre d'accidents : ")
        st.image("projet_DS\Streamlit\evolution.png")
        st.write("On aperçoit qu'après une grande baisse entre 2005 et 2012, le nombre d'accient est à peu près stable. Pourtant, le nombre de tué sur les routes Française semble continuer de se réduire.")
        st.write("")
        st.write("")

        st.subheader("Impact de l'âge sur la gravité des accidents : ")
        st.image("projet_DS\Streamlit\\accidents age 1", caption="Répartition de l'âge des accidentés suivant la gravité de l'accidents")
        st.image("projet_DS\Streamlit\\accidents age", caption="Taux d'accidents graves suivant l'âge")
        st.write("Sur les images précédentes, on voit bien l'impacte de l'age sur la gravité des accidents.")
        st.write("")
        st.write("")



# ___________________________________________Utiliser le RandomForest __________________________________________________________________________________________________________________________________________
if page_selectionnee == "Classification binaire avec Random Forest" :
    action_selectionne = st.sidebar.selectbox("Random Forest",["Présentation", "Prédiction"])

    if action_selectionne=="Présentation":  
        noms_col= pd.read_csv('projet_DS\Streamlit\\001_df_processed.csv',nrows=1,).columns
        noms_col_raw= pd.read_csv('projet_DS\Streamlit\\000_df_RAW.csv',nrows=1,).columns
        st.subheader("Pre-processing :")
        st.write("On ne garde qu'une ligne par accident, dont les variables seront celles de son usager le plus blessé. Ensuite, on rend la target binaire : accident léger ou accident grave.")
        code="""#_On ne garde qu'une ligne par accident
df['grav']=df['grav'].replace({1:0,2:3,3:2,4:1}) #_On ordonne les gravités
df = df.sort_values(by='grav')
df=df.drop_duplicates(subset='Num_Acc',keep='last')

X = df.drop(columns=['grav','date','Num_Acc']) 
y = df['grav']

#_Regroupement binaire de y (0 = accident léger et 1 = accident grave)
y=y.replace({0:'0',1:'0',2:'1',3:'1'}).astype(int)

#_Séparation du jeu en train et test
X_train_save,X_test_save,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)"""
        st.code(code)

        st.write("On cherche les variables les plus importantes avec un RFECV entrainé sur le recall. Ainsi on aura un modèle simple, rapide, et qui rate peu d'accidents graves \
                  (si on imagine que le modèle sert à décider de l'envoi d'une ambulance : il vaut mieux en envoyer pour rien que rater des blessés). Les variables qui en ressortent sont celles de l'extrait suivant :")
        taille_nrows = 5
        alea=np.random.randint(0,1000000)         #pour avoir des valeurs aléatoire du csv
        extrait = pd.read_csv('projet_DS\\Streamlit\\003_accidents.csv',nrows=taille_nrows,skiprows=alea,header=None)
        extrait.columns = noms_col
        extrait['grav']=extrait['grav'].replace({0:'0',1:'0',2:'1',3:'1'})
        extrait=extrait[['Num_Acc','grav','agg','age','prof_Pente','manv_Aucune_Manoeuvre','circ_Bidirectionnelle','catv_Voiture','catr_Departementale','region_Île-de-France']]
        st.write(extrait)
        if st.button("Recharger un nouvel extrait"):
            alea=np.random.randint(0,1000000)

        st.subheader("Entrainement :")
        st.write("Le modèle est volontairement limité à une profondeur de 3 dans un soucis d'interprétabilité.")
        code="""#_Modèle entrainé :
model = RandomForestClassifier(n_estimators=100, 
                               max_depth=3,  
                               n_jobs=10, 
                               class_weight='balanced', 
                               criterion='entropy',
                               random_state=42)
model.fit(X_train, y_train)"""
        st.code(code)

        st.subheader("Résultats :")

        txt="""Rapport de classification et matrice de confusion :

              precision    recall  f1-score   support

           0       0.72      0.77      0.75    128919
           1       0.66      0.60      0.63     95395

    accuracy                           0.70    224314
   macro avg       0.69      0.69      0.69    224314
weighted avg       0.70      0.70      0.70    224314

_____________________________________________________

grav             0      1
predictions              
0            99634  38159
1            29285  57236"""
        st.text(txt)
        st.image("projet_DS\\Streamlit\\arbre_exemple.png", caption="Exemple d'un arbre du modèle.")
        st.write("L'impact des variables sur la classification peut être observée avec le graphe de SHAP suivant :")
        st.image("projet_DS\\Streamlit\\shap.png", caption="Graph de SHAP du modèle")



    if action_selectionne=="Prédiction":   
        st.header("Sélection de valeurs")
        st.subheader("Décrivez votre accident :")
        age = st.slider("Votre age", 0, 100, 50)
        agg = st.checkbox("Etiez vous en agglomération ?")
        catr_Departementale = st.checkbox("Etiez-vous sur une départementale ?")
        prof_Pente = st.checkbox("La route était-elle en pente ?")
        manv_Aucune_Manoeuvre = st.checkbox("Faisiez-vous une manoeuvre ? (par exemple un changement de file ou un dépassement)")
        circ_Bidirectionnelle = st.checkbox("La circulation était-elle dans les deux sens ?")
        catv_Voiture = st.checkbox("Au moins un des véhicules impliqués était une voiture (véhicule de tourisme) ?")
        region_Île_de_France = st.checkbox("Etait-ce en Île-de-France ?")
    
    ## ordre des colonnes du model ['agg','age','prof_Pente','manv_Aucune_Manoeuvre','circ_Bidirectionnelle','catv_Voiture','catr_Departementale','region_Île-de-France']
        X_simu= pd.DataFrame({
        'agg': [agg],
        'age': [age],
        'prof_Pente': [prof_Pente],
        'manv_Aucune_Manoeuvre': [manv_Aucune_Manoeuvre],
        'circ_Bidirectionnelle': [circ_Bidirectionnelle],
        'catv_Voiture': [catv_Voiture],
        'catr_Departementale': [catr_Departementale],
        'region_Île-de-France': [region_Île_de_France]})
    

        model = joblib.load('projet_DS\\Streamlit\\stream_random_forest_model.pkl')
        y_simu       = model.predict(X_simu) 
        y_simu_proba = model.predict_proba(X_simu) 
        st.write("") 
        st.write("") 
        
        if y_simu==0 : 
            st.subheader("Bonne nouvelle : vous n'êtes probablement pas gravement blessé",)
            st.markdown("<span style='color:green;'>Probabilité estimée par le modèle = {:.2f}</span>".format(y_simu_proba[0, 0]), unsafe_allow_html=True)


        else:
            st.subheader("Mauvaise nouvelle : vous devez probablement allez à l'hopital")
            st.markdown("<span style='color:red;'>Probabilité estimée par le modèle = {:.2f}</span>".format(y_simu_proba[0, 1]), unsafe_allow_html=True)


        st.subheader("Graphe de SHAP de la prédiction :")
        explainer = joblib.load('projet_DS\\Streamlit\\explainer_shap.pkl')
        shap_values = explainer.shap_values(X_simu)
        shap.force_plot(explainer.expected_value[0], shap_values[0], X_simu.iloc[0, :], matplotlib=True,figsize=(15, 4))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(bbox_inches='tight', format='png')







##______________________________________ Regression Justine _______________________________________________________________________________________________________________


if page_selectionnee == "Classification binaire avec Logistic Regression":

    jeu_selectionne = st.sidebar.selectbox("Prédire avec Logistic Regression", ["Explication","RFE + GridSearchCV",]) #"Prédire avec Logistic Regression"
    if jeu_selectionne=="Explication":
        st.header("Logistic régression ")
        st.subheader("")
        st.write("La régression logistique fonctionne en utilisant une fonction logistique pour modéliser la probabilité que la variable cible appartienne à une classe particulière en fonction des variables de prédiction.")
        st.write("La régression logistique produit une probabilité dans l'intervalle [0, 1]. Cette probabilité est ensuite transformée en une classe prédite en utilisant un seuil de décision.")
    if jeu_selectionne=="RFE + GridSearchCV":
        st.header("Logistic Regression avec l'Élimination Récursive des Caractéristiques (RFE)")
        st.subheader("La precision en fonction du nombre de variables choisies")
        st.write("Cette étape cruciale permet de nettoyer les données d'avantage, en éliminant le bruit inutile. Cela permet au modèle de se concentrer sur les informations véritablement pertinentes. ")
# Définissez le nombre de caractéristiques sélectionnées et leurs précisions correspondantes
        # Définir le nombre de caractéristiques sélectionnées et leurs précisions correspondantes
        num_caracteristiques_selectionnees = [5, 10, 20, 25, 30, 40, 50, 60, 70]
        precisions = [0.60, 0.62, 0.68, 0.70, 0.71, 0.72, 0.72, 0.72, 0.72]
        indice_meilleure_precision = np.argmax(precisions)
        meilleure_precision = precisions[indice_meilleure_precision]
        meilleur_nombre_caracteristiques = num_caracteristiques_selectionnees[indice_meilleure_precision]
# Créez le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(num_caracteristiques_selectionnees, precisions, marker='o', linestyle='-')
        ax.set_xlabel('Nombre de Caractéristiques Sélectionnées (n)')
        ax.set_ylabel('Précision')
        ax.set_title('Précision en Fonction du Nombre de Caractéristiques Sélectionnées')
        ax.grid(True)
# Mettez en évidence le point avec la précision la plus élevée
        ax.scatter(meilleur_nombre_caracteristiques, meilleure_precision, color='red', marker='x', label=f'Meilleure Précision : {meilleure_precision:.2f}', s=100, zorder=3)
        ax.legend()
# Remplacez l'annotation pour le point (40, 0.76) (ajustez la position)
        ax.annotate('(40, 0.76)', (40, 0.76), textcoords="offset points", xytext=(0, -37), ha='center')
# Affichez le graphique à l'aide de st.pyplot()
        st.pyplot(fig)
        st.subheader("La malédiction de la dimensionnalité: 'Curse of dimensionality'")
        st.write("Le phénomène selon lequel l'ajout de plus de variables à un modèle conduit initialement à une amélioration significative des performances, mais après un certain point, l'ajout de variables offre des avantages décroissants, voire négligeables. ")
    # Add a section for Feature Selection with RFE
        st.subheader("Sélectionnez le nombre de variable par RFE pour voir son impact sur la precision")
        num_features = st.slider("Nombre de caractéristiques à inclure:", 10, 70, 10, 10)  # Adjust the range and step size
        st.write(f"Exécution de la Sélection de caractéristiques avec {num_features} caractéristiques...")
        scores = {
            10: {
                "classification_report": """
                Classification Report:
               precision    recall  f1-score   support
           0       0.61      0.98      0.75    128919
           1       0.84      0.14      0.25     95395
    accuracy                           0.62    224314
   macro avg       0.73      0.56      0.50    224314
weighted avg       0.71      0.62      0.54    224314
                """,
                "roc_auc_score": 0.75
            },
            20: {
                "classification_report": """
                  Classification Report:
               precision    recall  f1-score   support
           0       0.61      0.98      0.75    128919
           1       0.84      0.14      0.25     95395
    accuracy                           0.62    224314
   macro avg       0.73      0.56      0.50    224314
weighted avg       0.71      0.62      0.54    224314
""",
                "roc_auc_score": 0.76
            },
            30: {
                "classification_report": """
                  Classification Report:
               precision    recall  f1-score   support
           0       0.73      0.82      0.77    128919
           1       0.70      0.59      0.64     95395
    accuracy                           0.72    224314
   macro avg       0.71      0.70      0.70    224314
weighted avg       0.72      0.72      0.71    224314
                """,
                "roc_auc_score": 0.77
            },
           40: {
                "classification_report": """
                  Classification Report:
               precision    recall  f1-score   support
           0       0.73      0.82      0.77    128919
           1       0.70      0.59      0.64     95395
    accuracy                           0.72    224314
   macro avg       0.71      0.70      0.70    224314
weighted avg       0.72      0.72      0.71    224314
                """,
                "roc_auc_score": 0.77
            },
            50: {
                "classification_report": """
                  Classification Report:
               precision    recall  f1-score   support
           0       0.73      0.82      0.77    128919
           1       0.70      0.59      0.64     95395
    accuracy                           0.72    224314
   macro avg       0.71      0.70      0.70    224314
weighted avg       0.72      0.72      0.71    224314
                """,
                "roc_auc_score": 0.77
            },
            60: {
                "classification_report": """
                  Classification Report:
               precision    recall  f1-score   support
           0       0.73      0.82      0.77    128919
           1       0.70      0.59      0.64     95395
    accuracy                           0.72    224314
   macro avg       0.71      0.70      0.70    224314
weighted avg       0.72      0.72      0.71    224314
                """,
                "roc_auc_score": 0.77
            },
            70: {
                "classification_report": """
                  Classification Report:
               precision    recall  f1-score   support
           0       0.73      0.82      0.77    128919
           1       0.70      0.59      0.64     95395
    accuracy                           0.72    224314
   macro avg       0.71      0.70      0.70    224314
weighted avg       0.72      0.72      0.71    224314
                """,
                "roc_auc_score": 0.77
            },
        }
        # Display Classification Report and ROC AUC Score based on the selected features
        if num_features in scores:
            st.subheader(f"Rapport de Classification pour {num_features} caractéristiques :")
            st.code(scores[num_features]["classification_report"])
            st.subheader(f"Score ROC AUC pour {num_features} caractéristiques :")
            st.write(scores[num_features]["roc_auc_score"])
        st.subheader("Le meilleure resultats avec le moins de features:")
        st.write( "__40 features sélection:__")
        st.write(
       """
        ['lum', 'agg', 'age', 'Ceinture', 'Dispositif_Enfant', 'catu_Conducteur',
       'sexe_Masculin','trajet_Achats', 'trajet_Loisirs',
       'trajet_Utilisation_Professionnelle',
       'locp_Chausse_Hors_Passage_Pieton', 'senc_Repere_Croissant',
       'senc_Repere_Decroissant', 'plan_Courbe', 'manv_Autre',
       'manv_Contresens', 'manv_Deport', 'situ_Chausse',
       'circ_Bidirectionnelle', 'circ_Chausse_Separee', 'circ_Sens_Unique',
       'infra_Carrefour', 'obs_Fixe', 'catv_Lourd', 'choc_Arriere',
       'catr_Departementale', 'catr_Voix_Communale',
       'region_Bourgogne-Franche-Comté', 'region_Pays de la Loire',
       'region_Île-de-France']
       """)
        st.subheader('Le meilleure GridSearchCV avec 40 features:')
        st.write("Meilleure Hyperparameters: (C=0.001, penalty='l2', max_iter=1000, solver='saga')")



##______________________________________ Prediction Justine _______________________________________________________________________________________________________________
    if jeu_selectionne=="Prédire avec Logistic Regression":
        st.header("Predire la gravité d'un accident dans une situation")
        st.subheader("Décrivez votre accident :")
        st.subheader("Général")
        sexe_feminin = st.checkbox("Genre : Féminin")
        sexe_masculin = st.checkbox("Genre : Masculin")
        age = st.slider("Votre âge", 0, 100, 50)
        catu_conducteur = st.checkbox("Étiez-vous conducteur ?")
        locp_chausse = st.checkbox("Étiez-vous un piéton sur la chaussée, hors passage piéton ?")
        st.subheader("Sécurité")
        ceinture = st.checkbox("Portiez-vous la ceinture de sécurité ?")
        dispositif_enfant = st.checkbox("Y avait-il un dispositif enfant dans le véhicule ?")
        st.subheader("Circulation")
        agg = st.checkbox("Étiez-vous en agglomération ?")
        lum = st.checkbox("Faisait-il nuit ?")
        st.subheader("Type de trajet")
        trajet_achats = st.checkbox("Trajet : Achats")
        trajet_loisirs = st.checkbox("Trajet : Loisirs")
        trajet_pro = st.checkbox("Trajet : Utilisation Professionnelle")
        st.subheader("Autres Caractéristiques")
        senc_croissant = st.checkbox("Sens de circulation : Repère Croissant")
        senc_decroissant = st.checkbox("Sens de circulation : Repère Décroissant")
        plan_courbe = st.checkbox("Plan de la chaussée : Courbe")
        manv_autre = st.checkbox("Manœuvre : Autre")
        manv_contresens = st.checkbox("Manœuvre : Contresens")
        manv_deport = st.checkbox("Manœuvre : Déport")
        situ_chausse = st.checkbox("Situation : Chaussée")
        circ_bidirectionnelle = st.checkbox("Circulation : Bidirectionnelle")
        circ_chausse_sep = st.checkbox("Circulation : Chaussée Séparée")
        circ_sens_unique = st.checkbox("Circulation : Sens Unique")
        infra_carrefour = st.checkbox("Infrastructure : Carrefour")
        obs_fixe = st.checkbox("Obstacle : Fixe")
        catv_lourd = st.checkbox("Catégorie du véhicule : Lourd")
        choc_arriere = st.checkbox("Choc : Arrière")
        catr_departementale = st.checkbox("Catégorie de route : Départementale")
        catr_communale = st.checkbox("Catégorie de route : Voie Communale")
        region_bourgogne = st.checkbox("Région : Bourgogne-Franche-Comté")
        region_pays_loire = st.checkbox("Région : Pays de la Loire")
        region_ile_france = st.checkbox("Région : Île-de-France")
    # Construct the selected_features dictionary
        X_simu = pd.DataFrame({
            'lum': [lum],
            'agg': [agg],
            'age': [age],
            'Ceinture': [ceinture],
            'Dispositif_Enfant': [dispositif_enfant],
            'catu_Conducteur': [catu_conducteur],
            'sexe_Masculin': [sexe_masculin],
            'trajet_Achats': [trajet_achats],
            'trajet_Loisirs': [trajet_loisirs],
            'trajet_Utilisation_Professionnelle': [trajet_pro],
            'locp_Chausse_Hors_Passage_Pieton': [locp_chausse],
            'senc_Repere_Croissant': [senc_croissant],
            'senc_Repere_Decroissant': [senc_decroissant],
            'plan_Courbe': [plan_courbe],
            'manv_Autre': [manv_autre],
            'manv_Contresens': [manv_contresens],
            'manv_Deport': [manv_deport],
            'situ_Chausse': [situ_chausse],
            'circ_Bidirectionnelle': [circ_bidirectionnelle],
            'circ_Chausse_Separee': [circ_chausse_sep],
            'circ_Sens_Unique': [circ_sens_unique],
            'infra_Carrefour': [infra_carrefour],
            'obs_Fixe': [obs_fixe],
            'catv_Lourd': [catv_lourd],
            'choc_Arriere': [choc_arriere],
            'catr_Departementale': [catr_departementale],
            'catr_Voix_Communale': [catr_communale],
            'region_Bourgogne-Franche-Comté': [region_bourgogne],
            'region_Pays de la Loire': [region_pays_loire],
            'region_Île-de-France': [region_ile_france]})
        model = joblib.load("projet_DS\Streamlit\\lrmodel_with_specific_params.pkl")
        y_simu_proba = model.predict_proba(X_simu)
        y_simu = model.predict(X_simu)
   # y_simu = model.predict(X_simu)
   # y_simu_proba = model.predict_proba(X_simu)
    #st.write("")
    #st.write("")
        if y_simu==0 :
            st.subheader("Bonne nouvelle : vous n'êtes probablement pas gravement blessé",)
            st.markdown("<span style='color:green;'>Probabilité estimée par le modèle = {:.2f}</span>".format(y_simu_proba[0, 0]), unsafe_allow_html=True)
        else:
            st.subheader("Mauvaise nouvelle : vous devez probablement allez à l'hopital")
            st.markdown("<span style='color:red;'>Probabilité estimée par le modèle = {:.2f}</span>".format(y_simu_proba[0, 1]), unsafe_allow_html=True)





#__________________________ Partie Hichem ________________________________________________________________________________________________________________________________

if page_selectionnee == "Prédire avec un réseau de neurones":
    import pandas as pd
    noms_col= pd.read_csv('projet_DS\\Streamlit\\001_df_processed_nogps.csv',nrows=1,).columns
    #side_bg = 'HA_Capture.png'
    #sidebar_bg(side_bg)

    st.title("Réseau de neurones")
    # option = st.selectbox("",["Prédiction","Qu'est ce qu'un réseau de neurone ?","Schéma du réseaux de neurones", "Coding","Liens"])
    option = st.selectbox("",["Schéma du réseaux de neurones","Prédiction", "Coding","Infos + Liens"])

    if option =="Schéma du réseaux de neurones":
        texte=("<br>Il est composé de trois couches principales : \
                <br><li><u>Entrée</u> : La couche d'entrée qui prend comme données nos 89 variables.\
                <br><li><u>Couche cachée</u>: cette couche est composée de 32 neurones entièrement connectés. \
                Elle utilise comme fonction d'activation  la ReLU qui est définie mathématiquement \
                par f(x) = max(0, x), où x est l'entrée et f(x) est la sortie. Elle renvoie x si x ≥ 0, et zéro si x <0. \
                Concernant l'initialisation des poids glorot_normal a été choisie. \
                Cette initialisation de Glorot aide à maintenir les gradients stables lors de la rétropropagation, \
                ce qui favorise une formation plus rapide et plus stable du réseau.\
                De plus, une couche de régularisation dropout est insérée après la couche cachée pour éviter le surapprentissage.\
                Il désactive aléatoirement 50% des neurones de la couche précédente pour chaque itération.\
                <br><li><u>Sortie</u>: La couche de sortie contient un seul neurone avec une fonction d'activation sigmoid couramment utilisée \
                pour les tâches de classification binaire. Le réseau produit une valeur entre 0 et 1 pour chaque exemple d'entrée, \
                représentant la probabilité d'appartenance à la classe positive.<br>"
                "<br> Trois callbacks sont utilisés dans l'entraînement du modèle pour améliorer sa performance: ModelCheckpoint \
                    : qui permet de sauvegarder le modèle lorsque la performance du modèle sur l'ensemble de validation est la meilleure.\
                    EarlyStopping : qui surveille la performance du modèle sur l'ensemble de validation au fil des époques d'entraînement. \
                        Si la performance ne s'améliore après 5 époques, l'entraînement est automatiquement arrêté pour éviter le surapprentissage.\
                        ReduceLROnPlateau : qui ajuste automatiquement le taux d'apprentissage du modèle en fonction de la performance observée \
                            sur l'ensemble de validation. Si la performance stagne, le taux d'apprentissage est réduit.<br>"
                "<br>Le réseau de neurones utilise une fonction de perte spécifique visant à améliorer le rappel: \
                                cette fonction combine la cross-entropy, mesurant la divergence entre les distributions réelles et \
                                    prédites, avec une perte de rappel. L'objectif est d'obtenir un équilibre entre la \
                                        précision des prédictions et l'aptitude à rappeler les données pertinentes.<br>")
        with st.expander("Le réseau de neurones mis en place est de type à propagation avant (Feedforward)..."):
            st.write(texte, unsafe_allow_html=True)
        image = Image.open("projet_DS\\Streamlit\\HA_DNN_visu.png")
        st.image(image, caption="Vue (partielle) du réseau de neurones")
        image = Image.open("projet_DS\\Streamlit\\HA_DNN_plots.png")
        st.image(image, caption="Fonction de perte et accuracy au cours des Epochs")

    if option == "Coding":
        code_python = """
        callbacks = [
            ModelCheckpoint("model_checkpoint.h5", save_best_only=True),
            EarlyStopping(patience=5),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
        ]

        inputs = Input(shape=98, name="Input")
        dense1 = Dense(units=32, activation="relu", kernel_initializer='glorot_normal', name="Dense_1")
        dropout_layer = Dropout(0.5) #dropout layer
        dense2 = Dense(units=1, activation="sigmoid", kernel_initializer='glorot_normal', name="Dense_2")

        dense1 = dense1(inputs)
        dropout=dropout_layer(dense1) #dropout layer
        outputs = dense2(dropout)

        mod = Model(inputs=inputs, outputs=outputs)

        def loss_Nico(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            vp =tf.reduce_sum(y_true*y_pred)
            totalp=tf.reduce_sum(y_true)
            recall=vp/(totalp + tf.keras.backend.epsilon())
            recall_loss=(1-recall)
            crossentropy=tf.keras.backend.binary_crossentropy(y_true,y_pred)
            loss = 0.5*recall_loss+crossentropy
            return loss

        mod.compile(loss=loss_Nico,
                    optimizer=tf.optimizers.Adam(learning_rate= 0.00001),
                    metrics=["accuracy"])

        DNN = mod.fit(np.asarray(X_ru).astype('int'), np.asarray(y_ru).astype('int'), epochs=20, batch_size=32, validation_split=0.2, callbacks=callbacks)
        """

        st.code(code_python, language='python')

    if option == "Infos + Liens":
        st.write(" Un réseau de neurones est un modèle mathématique utilisé en apprentissage automatique pour effectuer\
        des tâches de classification et de régression. Il est composé d'unités de base appelées neurones artificiels, \
        organisés en couches. Chaque neurone reçoit des entrées pondérées, effectue une opération mathématique\
        (généralement une somme pondérée) sur ces entrées, puis applique une fonction d'activation pour produire une sortie. \
        Les poids des connexions entre les neurones sont ajustés lors de l'apprentissage afin de minimiser une fonction de perte, \
        permettant au réseau d'apprendre des modèles à partir des données d'entraînement. Les réseaux de neurones profonds, \
        ou 'deep learning', sont des réseaux comportant de nombreuses couches cachées et sont utilisés pour des tâches complexes \
            comme la vision par ordinateur et le traitement du langage naturel.")

        video_url = "https://youtu.be/DazUaVu5MO0?si=X0R8GPzkVYICXvbG"
        image = Image.open("projet_DS\\Streamlit\\HA_DS_image_youtube.jpg")
        st.image(image, caption="Masterclass de Deep Learning par Datascientest")
        st.markdown(f'<a href="{video_url}" target="_blank">Vidéo YouTube</a>', unsafe_allow_html=True)


        st.write("https://datascientest.com/fonctionnement-des-reseaux-neurones")
        st.write("https://www.ibm.com/fr-fr/topics/neural-networks")
        st.write("https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels")

    if option == "Prédiction":
        data_to_add = {}

        st.header("Sélection de valeurs")
        st.subheader("Décrivez votre accident :")
        age = st.slider("Age", 0, 100, 50)
        data_to_add['age'] = age

        #Homme/Femme
        col1, col2 = st.columns(2)  # Diviser la ligne en deux colonnes
        with col1:
            if st.checkbox("Homme ?"):
                data_to_add['sexe_Masculin'] = 1
        with col2:
            if st.checkbox("Femme ?"):
                data_to_add['sexe_Feminin'] = 1
        #Journée/Nuit
        col1, col2 = st.columns(2)  
        with col1:
            if st.checkbox("Journée ?"):
                data_to_add['lum'] = 1
        with col2:
            if st.checkbox("Nuit ?"):
                data_to_add['lum'] = 0
        #Aglo?
        if st.checkbox("Etiez vous en agglomération ?"):
            data_to_add['agg'] = 1
        #Véhicule/Piéton?
        col1, col2 = st.columns(2)  
        vehi=0
        pieton=0
        with col1:
            if st.checkbox("Véhiculé ?"):
                if st.checkbox("Conducteur principal ?"):
                    data_to_add['catu_Conducteur'] = 1
                #place
                options_place = [None,'place_Avant_Gauche','place_Avant_Droit','place_Arriere_Droit','place_Arriere_Gauche','place_Autre']
                selected_place = st.selectbox("Place dans voiture :", options_place)
                if selected_place!=None : data_to_add[selected_place] = 1
                # data_to_add[selected_place] = 1
                if selected_place=="'place_Avant_Gauche":
                    data_to_add['catu_Conducteur'] = 1
                else :
                    data_to_add['catu_Passager'] = 1
                #Nb vehicule
                col23_option=[None,'2_Vehicules', '3_Vehicules']
                selected_col23 = st.selectbox("Plusieurs vehicules ?", col23_option)
                if selected_col23!=None : data_to_add[str('col_'+selected_col23)] = 1 
                #cat vehicule
                cat_option=[None,'Voiture','2_roues', 'Lourd', 'Utilitaire']
                selected_cat = st.selectbox("Catégorie :", cat_option)
                if selected_cat!=None : data_to_add[str('catv_'+selected_cat)] = 1 
                #Manoeuvre    
                manv_option=[None,'manv_Arret/se_gare','manv_Aucune_Manoeuvre','manv_Contresens','manv_Depasse','manv_Deport','manv_Tourne','manv_Autre']
                selected_manv = st.selectbox("Info sur la manoeuvre :", manv_option)
                if selected_manv!=None : data_to_add[selected_manv] = 1
                #various option
                options_various = ['obs_Piéton','obs_Vehicule','obs_Fixe','choc_Arriere','choc_Avant','choc_Droit','choc_Gauche',\
                                'choc_Tonneaux','int_Autre','int_TouY','int_X']
                selected_various = st.multiselect("Autre infos sur l'accident :", options_various)
                for option in selected_various:
                    data_to_add[option] = 1
        with col2:
            if st.checkbox("Piéton ?"):
                data_to_add['catu_Pieton'] = 1
                etatp_option=['Seul','Groupe']
                selected_etatp = st.selectbox("Seul ou groupe ? :", etatp_option)
                data_to_add[str('etatp_'+selected_etatp)] = 1 
                #locp
                locp_option=[None,'Sur_Passage_Pieton','Chausse_Hors_Passage_Pieton', 'Trottoir/Bord_de_route']
                selected_locp = st.selectbox("Piétion localisé :", locp_option)
                if selected_locp!=None : data_to_add[str('locp_'+selected_locp)] = 1 
                #actp
                actp_option=[None,'Traversant','Deplacement_sens_vehicule','Deplacement_sens_inv_vehicule',\
                             'Jouant_Courant','Avec_Animal','Autre']
                selected_actp = st.selectbox("Action piéton :", actp_option)
                if selected_actp!=None : data_to_add[str('actp_'+selected_actp)] = 1

        #Info sur la route
        options_inforoute = ['prof_Pente','plan_Courbe','situ_Chausse','surf_Glissant',\
                             'situ_Trottoir/Bord_de_route','circ_Bidirectionnelle','circ_Chausse_Separee','circ_Sens_Unique',\
                             'infra_Carrefour','infra_Autre','vosp_Cycl/Reserv','senc_Repere_Croissant','senc_Repere_Decroissant']
        selected_inforoute = st.multiselect("Info sur la route :", options_inforoute)
        for option in selected_inforoute:
            data_to_add[option] = 1
        #Regions
        regions_option=[None,"region_Île-de-France",'region_Auvergne-Rhône-Alpes','region_Bourgogne-Franche-Comté','region_Bretagne',\
                        'region_Centre-Val de Loire','region_Corse','region_Grand Est','region_Hauts-de-France','region_Normandie',\
                            'region_Nouvelle-Aquitaine','region_Occitanie','region_Outre-Mer','region_Pays de la Loire',\
                                "region_Provence-Alpes-Côte d'Azur"]
        selected_region = st.selectbox("Région :", regions_option)
        if selected_region!=None : data_to_add[selected_region] = 1
        #Type de route
        route_option=[None,'Departementale','Nationnale/Autoroute', 'Voix_Communale']
        selected_route = st.selectbox("Type de route :", route_option)
        if selected_route!=None : data_to_add[str('catr_'+selected_route)] = 1
        #securité
        options_secu = ["Casque", 'Ceinture','Dispositif_Enfant', 'Equipement_Réfléchissant']
        selected_sec = st.multiselect("Sécurité :", options_secu)
        for option in selected_sec:
            data_to_add[option] = 1

        #trajet
        trajet_option=[None, 'Domicile_Travail','Loisirs', 'Utilisation_Professionnelle','Achats', 'Autre']
        selected_trajet = st.selectbox("Type de trajet :", trajet_option)
        if selected_trajet!=None : data_to_add[str('trajet_'+selected_trajet)] = 1

    #-------------------------DNN-----------------------------------------
        from tensorflow import keras
        import tensorflow as tf
        from keras.utils import get_custom_objects
        from sklearn.preprocessing import StandardScaler
        def loss_Nico(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            vp =tf.reduce_sum(y_true*y_pred)
            totalp=tf.reduce_sum(y_true)
            recall=vp/(totalp + tf.keras.backend.epsilon())
            recall_loss=(1-recall)
            crossentropy=tf.keras.backend.binary_crossentropy(y_true,y_pred)
            loss = 0.5*recall_loss+crossentropy
            return loss
        get_custom_objects().update({'loss_Nico': loss_Nico})
        DNN_model = tf.keras.models.load_model('projet_DS\\Streamlit\\HA_DNN.h5')
    #---------------------------------------------------------------------
        X = pd.DataFrame()
        X = pd.concat([X, pd.DataFrame([data_to_add])], ignore_index=True)
        st.sidebar.dataframe(X.age, width=400)
        st.sidebar.dataframe(X.drop(["age"], axis=1).columns, width=400)

        df = pd.read_csv('projet_DS\\Streamlit\\HA_sample_df_processed_nogps.csv',nrows=1)
        df.loc[:, :] = 0
        columns_to_add = [col for col in df.columns if col not in X.columns]
        X_simu = pd.concat([X, df[columns_to_add]], axis=1)
        scaler = joblib.load('projet_DS\\Streamlit\\HA_scaler.pkl')
        X_simu.age=scaler.transform(np.array(X_simu.age).reshape(1, -1))

        st.write('____________________________________________')
        st.write('Informations modélisation et prédiction')
        st.write('____________________________________________')
        st.write('vecteur de prédiction:')
        st.write(X_simu)

        prediction = DNN_model.predict([[X_simu]])
        pred_class = [0 if valeur < 0.5 else 1 for valeur in prediction]
        st.write(f'Prédiction : {prediction[0]}')
        # st.write(f'Class : {pred_class[0]}')


        if len(data_to_add)>=4: # nb de variables mins complétés
            if pred_class[0]==1:
                side_bg = 'projet_DS\\Streamlit\\HA_Capture2.png'
                sidebar_bg(side_bg)
            if pred_class[0]==0: 
                st.write('\U0001F7E9 Tout est OK')
                st.sidebar.write('\U0001F7E9 Tout est OK')
            if pred_class[0]==1: 
                st.write('\U0001F7E5 Il faut envoyer des urgences cas grave possible')
                st.sidebar.write('\U0001F7E5 Il faut envoyer des urgences cas grave possible')



#-------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------streamlit HA----Evaluation------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
if page_selectionnee == "Evaluation" :
    st.header("Evaluation des modèles")
    image = Image.open("projet_DS\\Streamlit\\HA_evaluation.png")
    st.image(image, caption="Precision/Recall")

if page_selectionnee == "Conclusion" :
    st.header("Conclusion")
    st.subheader("Evaluation des modèles :")
    image = Image.open("projet_DS\\Streamlit\\HA_evaluation.png")
    st.image(image, caption="Precision/Recall")
    st.write("")
    st.subheader("Pour aller plus loin...")
    st.write("- Mise en production")
    st.write("- Analytiques")
    st.write("- Plus de data pour plus d'informations ?")

    

#-------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------ EUGENE ------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------


# Modélisation
if page_selectionnee == "Prédire avec l'analyse factorielle exploratoire":
    kmo_failed=joblib.load('projet_DS\\Streamlit\\model_edmond\\failed.pkl')
    kmo_passed=joblib.load('projet_DS\\Streamlit\\model_edmond\\passed.pkl')
    kmo_model=joblib.load('projet_DS\\Streamlit\\model_edmond\\model.pkl')
    kmo_all=joblib.load('projet_DS\\Streamlit\\model_edmond\\all.pkl')
    kmo_model_temp=joblib.load('projet_DS\\Streamlit\\model_edmond\\model_temp.pkl')
    kmo_all_temp=joblib.load('projet_DS\\Streamlit\\model_edmond\\all_temp.pkl')
    factors=joblib.load('projet_DS\\Streamlit\\model_edmond\\factors.pkl')

    st.write("Dans cette partie, nous allons vous présenter l'analyse factorielle exploratoire.")
    st.title("Étapes impliquées dans la réalisation d'une analyse factorielle exploratoire")
    st.image("projet_DS\\Streamlit\\etape.png")

    st.write("Détermination de la taille de l'échantillon et adéquation de l'échantillonnage par la methode KMO")


    st.write("Overall KMO = {:.3f}".format(kmo_model_temp))
    st.write("KMO per feature:\n{}".format(kmo_all_temp.round(3)))
    # Variables considered adequate by the KMO test
    st.write("Les variables jugées adéquates par le test KMO pour l'analyse factorielle.")
    kmo_passed
    # Variables considered inadequate by the KMO test
    st.write("Les variables jugées inadéquates par le test KMO pour l'analyse factorielle.")
    kmo_failed

    st.write("Overall KMO = {:.3f}".format(kmo_model))

    st.write("KMO per feature:\n{}".format(kmo_all.round(3)))

    st.write("Détermination du nombre de facteurs dans l'analyse factorielle exploratoire")
    st.title("Représentation graphique des valeurs propres")
    st.image("projet_DS\Streamlit\scree_plot.png")

    st.title("Représentation graphique des facteurs ainsi que les variables liées")
    st.image("projet_DS\Streamlit\graphique_facteurs.png")

    st.write("Les différents groupes de facteurs sont représentés comme suit:")
    factors

    st.header("Machine Learning")
    st.write("Dans cette sous-section nous allons utilisé la régression logistique pour prédire la gravité de l'accident. D'une part nous allons entraîné un modéle de la régression logistique multiclasse sur les variables latentes en suite sur les variables sélectionées par le test KMO. D'autre part nous allons entraîné un modéle de la régression logistique binaire sur les variables latentes en suite sur les variables sélectionées par le test KMO.")
    if st.checkbox("Afficher les résultats de la matrice de confusion et de la classification report (multiclasse)"):

        st.subheader("Classification report et matrice de confusion appliquer sur les variables latentes")
        st.image("projet_DS\\Streamlit\\multi_class_latent.png")

        st.subheader("Classification report et matrice de confusion appliquer sur les variables selectionées par le test KMO")
        st.image("projet_DS\\Streamlit\\multi_class_kmo.png")
        
    if st.checkbox("Afficher les résultats de la matrice de confusion et de la classification report (binaire)"):
        st.subheader("Classification report et matrice de confusion appliquer sur les variables latentes")
        st.image("projet_DS\\Streamlit\\binary_class_latent.png")

        st.subheader("Classification report et matrice de confusion appliquer sur les variables selectionées par le test KMO")
        st.image("projet_DS\\Streamlit\\binary_class_kmo.png")