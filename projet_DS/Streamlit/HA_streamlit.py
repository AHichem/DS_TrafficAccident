import streamlit as st
import numpy as np
import pandas as pd
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


noms_col= pd.read_csv('001_df_processed_nogps.csv',nrows=1,).columns


st.title("Prediction de la gravité d'un accident de la route")

page_selectionnee = st.sidebar.selectbox("Quelle page afficher ?", ["Accueil","Afficher un extrait du jeu de donnée", "Prédire avec Random Forest",\
                                                                    "Prédire avec un réseau de neurones","Evaluation"])

if page_selectionnee == "Accueil" :
    st.subheader("Projet sécu routiere JUL 2023")

# Affichage du jeu de donnée
if page_selectionnee == "Afficher un extrait du jeu de donnée" :
    taille_nrows = st.slider("Nombre de lignes à afficher (nrows)", 1, 20, 5)
    alea=np.random.randint(0,1000000)         #pour avoir des valeurs aléatoire du csv
    st.subheader("Extrait du jeu de données")
    extrait = pd.read_csv('001_df_processed_nogps.csv',nrows=taille_nrows,skiprows=alea,header=None)
    extrait.columns = noms_col

    extrait['grav']=extrait['grav'].replace({1:0,2:1,3:1,4:0})
    st.write(extrait)
    if st.button("Recharger un nouvel extrait"):
        alea=np.random.randint(0,1000000)


# Utiliser le RandomForest
if page_selectionnee == "Prédire avec Random Forest" :

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
    

    model = joblib.load('RFC.joblib')
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


    # explainer = joblib.load('projet_DS\\explainer_shap.pkl')
    # shap_values = explainer.shap_values(X_simu)
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_simu.iloc[0, :], matplotlib=True,figsize=(15, 4))
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot(bbox_inches='tight', format='png')



import base64 
from PIL import Image
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

#-------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------streamlit HA----DNN------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------

if page_selectionnee == "Prédire avec un réseau de neurones":
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
                Concernant l’initialisation des poids glorot_normal a été choisie. \
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
        image = Image.open("HA_DNN_visu.png")
        st.image(image, caption="Vue (partielle) du réseau de neurones")
        image = Image.open("HA_DNN_plots.png")
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
        image = Image.open("HA_DS_image_youtube.jpg")
        st.image(image, caption="Masterclass de Deep Learning par Datascientest")
        st.markdown(f'<a href="{video_url}" target="_blank">Vidéo YouTube</a>', unsafe_allow_html=True)


        st.write("https://datascientest.com/fonctionnement-des-reseaux-neurones")
        st.write("https://www.ibm.com/fr-fr/topics/neural-networks")
        st.write("https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels")

    if option == "Prédiction":
        X = pd.DataFrame()
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
        DNN_model = tf.keras.models.load_model('HA_DNN.h5')
    #---------------------------------------------------------------------
        X = X.append(data_to_add, ignore_index=True)
        st.sidebar.dataframe(X.age, width=400)
        st.sidebar.dataframe(X.drop(["age"],1).columns, width=400)

        df = pd.read_csv('HA_sample_df_processed_nogps.csv',nrows=1)
        df.loc[:, :] = 0
        columns_to_add = [col for col in df.columns if col not in X.columns]
        X_simu = pd.concat([X, df[columns_to_add]], axis=1)
        scaler = joblib.load('HA_scaler.pkl')
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
                side_bg = 'HA_Capture2.png'
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
    image = Image.open("HA_evaluation.png")
    st.image(image, caption="Precision/Recall")