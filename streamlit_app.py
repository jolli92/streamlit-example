# Base Python Libraries
import os
import io
import streamlit as st
from io import StringIO

# Data Manipulation
import numpy as np
import pandas as pd
import itertools


# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Machine Learning - Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Machine Learning - Model Selection & Evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, r2_score, mean_squared_error, mean_squared_log_error

# Machine Learning - Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Model Persistence
import pickle
import joblib
import dill
from joblib import load
# Statistical Analysis
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Advanced ML & Explanation Tools
import xgboost
import shap

df = pd.read_csv('bank.csv')
#euros = "EUROS.jpg"
def create_visualisations(df, variables):
    rows = len(variables)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=[f'Distribution de {var}' for var in variables])
    for i, var in enumerate(variables, start=1):
        if df[var].dtype == 'object':
            data = go.Bar(x=df[var].value_counts().index, y=df[var].value_counts(), name=var)
        else:
            data = go.Histogram(x=df[var], nbinsx=30, name=var)
        fig.add_trace(data, row=i, col=1)
    fig.update_layout(height=300 * rows, width=800, showlegend=False)
    return fig
def create_plotly_countplot(df, x, hue, title):
        # Définition de la palette de couleurs
    color_discrete_map = {'Yes': 'blue', 'No': 'red'}
    # Création du graphique avec la palette personnalisée
    fig = px.histogram(df, x=x, color=hue, barmode='group', color_discrete_map=color_discrete_map)
    fig.update_layout(title=title, xaxis_title=x, yaxis_title='Count')
    return fig
def create_plotly_histplot(df, x, color, title):
    # Utilisation de la même palette de couleurs pour la cohérence
    color_discrete_map = {'Yes': 'blue', 'No': 'red'}
    # Création du graphique avec la palette personnalisée
    fig = px.histogram(df, x=x, color=color, barmode='overlay', nbins=50, color_discrete_map=color_discrete_map)
    fig.update_layout(title=title, xaxis_title=x, yaxis_title='Count')
    return fig
def get_correlation_plot(variable):
    if variable in ['age', 'balance','duration','pdays','campaign','previous','day']:
        return create_plotly_histplot(df, variable, 'deposit', f'Relation entre {variable} et deposit')
    else:
        return create_plotly_countplot(df, variable, 'deposit', f'Relation entre {variable} et deposit')
def plot_knn_scores(X_train, y_train, X_test, y_test):
    score_mi = []
    score_eu = []
    score_ma = []
    score_ch = []

    for k in range(1, 100):
        knn_mi = KNeighborsClassifier(n_neighbors=k, metric="minkowski")
        knn_mi.fit(X_train, y_train)
        score_mi.append(knn_mi.score(X_test, y_test))

    for k in range(1, 100):
        knn_eu = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn_eu.fit(X_train, y_train)
        score_eu.append(knn_eu.score(X_test, y_test))

    for k in range(1, 100):
        knn_ma = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
        knn_ma.fit(X_train, y_train)
        score_ma.append(knn_ma.score(X_test, y_test))

    for k in range(1, 100):
        knn_ch = KNeighborsClassifier(n_neighbors=k, metric="chebyshev")
        knn_ch.fit(X_train, y_train)
        score_ch.append(knn_ch.score(X_test, y_test))

    plt.plot(range(1, 100), score_mi, color='blue', linestyle='dashed', lw=2, label='Minkowski')
    plt.plot(range(1, 100), score_eu, color='black', linestyle='dashed', lw=2, label='Euclidean')
    plt.plot(range(1, 100), score_ma, color='orange', linestyle='dashed', lw=2, label='Manhattan')
    plt.plot(range(1, 100), score_ch, color='red', linestyle='dashed', lw=2, label='Chebyshev')
    plt.title('Accuracy Score - valeur de K')
    plt.xlabel('Valeur de K')
    plt.ylabel('Accuracy')
    plt.legend()
    return plt
st.title('Projet : MARKETING BANCAIRE')
st.sidebar.title("SOMMAIRE")
pages=["Présentation du Projet", "Datavisualisation","Modélisation","Prédictions", "Utilisation professionnelle du projet", "Conclusion"]

page=st.sidebar.radio("Aller vers", pages)


st.sidebar.markdown("### Équipe du Projet")
st.sidebar.markdown("- [Jérémy Ollier](https://www.linkedin.com/in/jérémy-ollier-25812a275)")
st.sidebar.markdown("- [Coralie Touodop](https://www.linkedin.com/in/coralie-touodop)")
st.sidebar.markdown("- [Heinrich Efoulou](https://www.linkedin.com/in/heinrich-efoulou-4a2626156)")
st.sidebar.markdown("- [Thi-Thuy Tran](https://www.linkedin.com/in/thi-thuy-tran-266610276/)")
st.sidebar.markdown("- [Basile Tekam](https://www.linkedin.com/in/basile-tekam-49a267156)")

if page == pages[0] :
  st.markdown(
        """
        <style>
            .big-font {
                font-size: 32px !important;
                color: #1E90FF;  /* Dodger Blue */
                text-align: center;
            }
            .highlight {
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .section {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 30px;
                border-radius: 20px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
  st.markdown('<p class="big-font">Présentation du projet</p>', unsafe_allow_html=True)  
  #st.header("Présentation du projet")
  
  st.write("Dans le cadre de notre formation de Data Analyst, nous avons eu l\'opportunité de développer \
  un projet axé sur la prédiction de souscription à un compte à terme (CAT) par les clients d\'une institution financière.")
  
  st.write("Dans un contexte économique marqué par le blocage du taux de rémunération du livret A, une remontée des taux et une forte inflation,\
  les comptes à terme sont devenus une alternative de plus en plus privilégiée, comme en témoigne l\'augmentation significative de leurs encours,\
  passant de 80 milliards d’euros en janvier 2023 à 122 milliards en juillet 2023 selon la Banque de France.")

  st.write("Notre objectif principal était de construire un modèle prédictif fiable, capable de déterminer la probabilité de souscription à un CAT.")

  
  st.write("Nous avons utilisé des méthodes d\'analyse de données avancées et des techniques de machine learning, incluant la régression logistique, \
  KNN, les arbres de décision, Random Forest et XGBoost, pour analyser un dataset riche et effectuer des prédictions.")
  
  st.write("Ces méthodes ont permis de dégager des insights cruciaux pour un service commercial, afin de mieux cibler les clients potentiels.")
  
  st.write("Ce projet ne se limite pas seulement à une application académique, mais offre des perspectives concrètes pour améliorer les stratégies \
  de marketing et optimiser la prise de décision dans le secteur financier.")

#if page == pages[0] :
  #st.image(euros)

if page == pages[1] :
  st.markdown(
        """
        <style>
            .big-font {
                font-size: 32px !important;
                color: #1E90FF;  /* Dodger Blue */
                text-align: center;
            }
            .highlight {
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .section {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 30px;
                border-radius: 20px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Section 1: Modélisation
  st.markdown('<p class="big-font">Exploration</p>', unsafe_allow_html=True)
  #st.header('Exploration')
  explo_choisi = st.selectbox(label = "Choix", options = ['-- Sélectionnez un menu --', 'Source et présentation du jeu de données', 'description du jeu de données'])
  if explo_choisi == 'Source et présentation du jeu de données':
      st.write("Le jeu de données du projet provient de Kaggle dont le lien se trouve ci-dessous:")
      lien_http = "[https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)"
      st.markdown(lien_http, unsafe_allow_html=True)
      st.write("Le jeu présente les détails sur une campagne marketing menée par une institution financière.")
      st.write("L'objectif du projet est d’analyser ces données pour identifier les opportunités, développer des stratégies futures, \
      améliorer les campagnes marketing à venir de la banque afin d'aboutir à une hausse du taux de souscription d'un compte de dépôts à terme.")
      
  if explo_choisi == 'description du jeu de données':
      st.write("Le jeu de Données se présente sur **11 162 lignes et 17 colonnes**, aucune lignes dupliquées, ni de valeurs manquantes avec **7 variables quantitatives (int64) et 10 variables qualitatives (Object)**")
      st.dataframe(df.head())
      st.write("Nous constatons ici que le jeu de Données affiche une variable cible **déposit** et **deux axes d'analyses** dont le premier axe est sur **le profil des clients** et le deuxième axe est sur **le déroulement de la campagne déjà réalisée** et les retours de cette dernière.")
      st.write("**Description des variables utilisées**")
      st.write("**AGE** : Variable quantitative qui représente l'âge de la personne.")
      st.write("**JOB** : Variable catégorielle qui désigne le métier de la personne.")
      st.write("**MARITAL** : Variable qualitative indiquant le statut matrimonial de la personne.")
      st.write("**EDUCATION** : Variable qualitative qui annonce le niveau d'études de la personne.")
      st.write("**DEFAULT** : Variable catégorielle désignant le risque de solvabilité d’un client, cela nous permet de savoir si un client est en défaut de paiement ou pas.")
      st.write("**BALANCE** : Variable quantitative désignant le solde bancaire de la personne prospectée.")
      st.write("**HOUSING**: Variable qualitative informant si la personne a un crédit immobilier ou non.")
      st.write("**LOAN** : Variable catégorielle représentant l'ensemble des clients endettés par un crédit de consommation.")
      st.write("**CONTACT** : Variable catégorielle désignant la façon dont les clients ont été contacté pendant la campagne marketing précédente.")
      st.write("**DAY** : Variable quantitative désignant le jour où le client a été contacté pour la dernière fois.")
      st.write("**MONTH** : Variable catégorielle correspondant au dernier mois ou l’on a contacté le client.")
      st.write("**DURATION** : Variable quantitative représentant le temps en seconde échangé lors du dernier contact.")
      st.write("**CAMPAIGN** : Variable quantitative indiquant le nombre de contacts réalisé durant la campagne par individu.")
      st.write("**PDAYS** : Variable quantitative indiquant le nombre de jours écoulé depuis le dernier contact échangé avec le client (lors de la campagne précédente). Sachant que -1 signifie que le client n’a pas été contacté lors de la campagne précédente")
      st.write("**PREVIOUS** : Variable quantitative indiquant le nombre de contacts avec le client lors de la campagne précédente.")
      st.write("**POUTCOME** : Variable catégorielle montrant le résultat de la campagne de marketing précédente.")
  
  #st.header('Visualisation')
  st.markdown('<p class="big-font">Visualisation</p>', unsafe_allow_html=True)
  #st.subheader('Distribution des variables')
  st.markdown(
        """
        <style>
            .highlight {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text with adjusted styling
  st.markdown('<p class="highlight">Distribution des variables</p>', unsafe_allow_html=True)


    # Widgets pour la sélection des variables et l'affichage des commentaires
  with st.container():
      selected_vars = st.multiselect('Sélectionnez les variables à visualiser:', df.columns)
      show_annotations = st.checkbox('Afficher les commentaires')

  if selected_vars:
      fig = create_visualisations(df, selected_vars)
      st.plotly_chart(fig)

      if show_annotations:
          st.write("***Commentaire de la variable selectionnée***")

            # Dictionnaire des commentaires pour chaque variable
          comments = {
          'job' : "La variable 'job' est catégorielle, indiquant le métier des clients avec 12 catégories uniques, notamment 'admin.', 'technician', 'services', etc. La majorité (23%) travaille dans le management, suivie des ouvriers (17,4%) et des techniciens (16,3%). Un faible pourcentage (0,6%) est classé comme 'unknown', indiquant une absence de déclaration professionnelle.",
          'age' : "\'Age' est une variable quantitative, représentant l'âge des individus, variant de 18 à 95 ans avec une moyenne de 41 ans. La moitié des individus a 39 ans ou moins. La majorité se situe entre 25 et 59 ans.",
          'marital' : "La variable qualitative 'marital' décrit le statut matrimonial avec trois catégories : 'married' (56,9%), 'single' (31,5%) et 'divorced' (11,6%).",
          'education' : "'education' catégorise le niveau d'étude des clients en 'unknown', 'secondary', 'primary', 'tertiary'. La majorité (49,1%) a un niveau secondaire, suivie par l'enseignement supérieur (33%) et primaire (13,4%). 'unknown' peut indiquer une non-disclosure ou l'absence d'éducation formelle.",
          'balance' : "'balance' est une variable quantitative, reflétant le solde bancaire des clients, variant de -6 847 à 81 204 euros, avec une moyenne de 1 528,54 euros. La moitié des clients a un solde autour de 550 euros, la plupart se situant entre 122 et 1 708 euros. Des valeurs extrêmes existent, nécessitant une attention particulière.",
          'default' : "'default', une variable catégorielle booléenne, indique si un client est en défaut de paiement (Yes/No). La grande majorité (98,5%) n'est pas en défaut.",
          'housing' : "La variable catégorielle 'housing' indique la possession d'un crédit immobilier (Yes/No), avec 52,7% des clients sans crédit immobilier et 47,3% en ayant un.",
          'loan' : "'loan' est une variable catégorielle booléenne indiquant si un client a des dettes (Yes/No). 86,9% des clients n'ont pas de dette et 13,1% en ont.",
          'contact' : "'contact' catégorise le mode de contact pendant la campagne en 'cellular', 'unknown', et 'telephone'. 72% ont été contactés par téléphone, et 21% de manière inconnue, possiblement par mail ou en présentiel.",
          'day' : "'day', une variable quantitative, représente le jour du dernier contact avec le client, avec une répartition équilibrée (moyenne et médiane = 15), et des pics d'appels les jours 1, 10, 24, et 31.",
          'month' : "La variable 'month' indique le dernier mois de contact, avec 12 catégories (jan, feb, mar, ..., nov, dec). Mai est le mois le plus actif (25,3%), suivi par juillet, août, et juin.",
          'duration' : "'duration' mesure la durée du dernier appel en secondes, une variable quantitative avec une moyenne de 371,99 secondes. Des appels dépassant une heure sont à analyser attentivement.",
          'campaign' : "'campaign' quantifie le nombre de contacts par client pendant la campagne, variant jusqu'à 36 fois. La majorité a été contactée une seule fois. Un pic à 63 contacts nécessite une analyse approfondie.",
          'pdays' : "'pdays' est le nombre de jours écoulés depuis le dernier contact de la campagne précédente, avec -1 indiquant aucun contact antérieur. La moyenne est de 51,33 jours, mais la médiane à -1 suggère que 50% des clients n'avaient pas été contactés auparavant.",
          'previous' : "'previous' compte les contacts lors de la campagne précédente. Avec une majorité de clients (plus de 8000) non contactés auparavant, la moyenne est presque nulle, soulignant un contact rarement répété.",
          'poutcome' : "'poutcome' révèle le résultat de la campagne marketing précédente avec 'unknown', 'other', 'failure', 'success'. Une majorité (74,6%) est classée 'unknown', souvent due à l'absence de contact précédent.",
          'deposit' : "'deposit', la variable cible, indique si un client a souscrit à un dépôt à terme (Yes/No), avec 52,6% de refus et 47,4% de souscriptions."}

# Affichage des commentaires avec st.info
          for var in selected_vars:
              if var in comments:
                  st.info(f"{var}: {comments[var]}")
              else:
                  st.info(f"Aucun commentaire disponible pour {var}.")
  else:
      st.write("Veuillez sélectionner des variables pour afficher les graphiques et les commentaires associés.")
  st.write("L'étude sur les profils clients a révélé que la majorité se situe entre 30 et 60 ans, avec une éducation majoritairement au niveau secondaire et un statut matrimonial principalement marié. Les professions sont variées, dominées par les managers, suivis des ouvriers, techniciens et employés administratifs. La plupart des soldes bancaires annuels sont inférieurs à 20 000€. Les prêts immobiliers sont les crédits les plus courants, tandis que les prêts personnels et autres types de crédits sont moins fréquents.")
  st.write("Il en ressort que les clients ont été plus réceptifs durant l'été, surtout en mai, avec une baisse de contact en septembre, octobre, décembre, et mars. La plupart des contacts ont été faits par téléphone cellulaire. Les interactions avec les clients ont rarement dépassé trois contacts, pour éviter de les agacer. Concernant la variable « poutcome », 74,6% des clients sont classifiés comme « unknown », souvent dû à l'absence de contact antérieur. La variable cible « deposit » montre que 52,6% des clients ont refusé de souscrire, contre 47,4% ayant souscrit. L'analyse a révélé l'importance de toutes les variables, malgré la présence de nombreux outliers et la valeur « unknown » fréquente, qui ne doit pas être supprimée pour éviter la perte de données, surtout pour les nouveaux clients.")
# Analyses des corrélations et tests statistiques
  #st.subheader("Analyse des corrélations avec tests statistiques des variables explicatives")
  st.markdown(
        """
        <style>
            .highlight {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text with adjusted styling
  st.markdown('<p class="highlight">Analyse des corrélations avec tests statistiques des variables explicatives</p>', unsafe_allow_html=True)
# Widget pour choisir les heatmaps à afficher
  heatmap_choices = st.multiselect("Choisissez les heatmaps à afficher:",
                                ["Corr Numérique", "Corr Catégorielle", "Corr Num-Cat"])

# Boucle sur les choix de l'utilisateur et affichage des heatmaps correspondantes
  for choice in heatmap_choices:
      if choice == "Corr Numérique":
        # Affichage de la heatmap numérique
          st.subheader("Analyse de la corrélation entre les variables numériques")
           
    # Sélectionner uniquement les colonnes numériques
          numeric_columns = df.select_dtypes(include=['number']).columns
          df_numeric = df[numeric_columns]

        # Calculer la matrice de corrélation pour les variables numériques
          correlation_matrix = df_numeric.corr()

        # Créer et afficher la heatmap
          fig = px.imshow(correlation_matrix,
                      x=correlation_matrix.columns,
                      y=correlation_matrix.columns,
                        text_auto=True,
                        color_continuous_scale='RdBu')
          st.plotly_chart(fig)

        # Checkbox pour afficher le commentaire
          if st.checkbox("Afficher le commentaire sur la corrélation numérique", key="Num"):
              st.markdown("""L'analyse de corrélation montre peu de liens linéaires forts entre la plupart des variables numériques, à l'exception des paires 'pdays' et 'previous', 'age' et 'balance' ainsi que 'campaign' et 'day', qui montrent des corrélations positives notables.
L'absence de corrélations élevées est favorable pour éviter la multi-collinéarité dans le modèle d'apprentissage automatique.
""")

      elif choice == "Corr Catégorielle":
        # Affichage de la heatmap catégorielle
          st.subheader("Analyse de la corrélation entre les variables catégorielles")
          st.subheader("Heatmap des valeurs-p des tests du Chi-carré")
          def create_p_matrix(df):
              variables = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
              p_matrix = pd.DataFrame(np.nan, index=variables, columns=variables)

              for i in range(len(variables)):
                  for j in range(i+1, len(variables)):
                      var1 = variables[i]
                      var2 = variables[j]
                      contingency_table = pd.crosstab(df[var1], df[var2])
                      _, p, _, _ = chi2_contingency(contingency_table)
                      p_matrix.loc[var1, var2] = p
                      p_matrix.loc[var2, var1] = p

              np.fill_diagonal(p_matrix.values, 1)
              return p_matrix
          p_matrix = create_p_matrix(df)
          fig = px.imshow(p_matrix, x=p_matrix.columns, y=p_matrix.index, text_auto=True, color_continuous_scale='plasma')
          st.plotly_chart(fig)

        # Checkbox pour afficher le commentaire
          if st.checkbox("Afficher le commentaire sur la corrélation catégorielle", key="cat"):
              st.markdown("""la plupart des variables catégorielles dans notre ensemble de données sont interdépendantes, bien que certaines paires, telles que 'marital' et 'default', 'education' et 'default', 'default' et 'housing', ainsi que 'loan' et 'contact', ne montrent pas de dépendance significative. La majorité des tests indiquent des p-values inférieures à 5%, justifiant le rejet de l'indépendance entre ces variables catégorielles.
                            """)

      elif choice == "Corr Num-Cat":
        # Affichage de la heatmap ANOVA
          st.subheader("Analyse de la corrélation entre les variables catégorielles et les variables numériques")


# Identification des variables numériques et catégorielles, à l'exception de 'deposit'
          numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
          categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
          categorical_vars.remove('deposit')

# Création d'un DataFrame pour les résultats ANOVA
          anova_p_values = pd.DataFrame(np.nan, index=numeric_vars, columns=categorical_vars)

# Calcul des valeurs-p pour chaque paire de variables numériques et catégorielles
          for numeric_var, categorical_var in itertools.product(numeric_vars, categorical_vars):
              formula = f'{numeric_var} ~ C({categorical_var})'
              model = ols(formula, data=df).fit()
              anova_result = sm.stats.anova_lm(model, typ=2)
              p_value = anova_result["PR(>F)"][0]
              anova_p_values.loc[numeric_var, categorical_var] = p_value

# Création de la heatmap avec Plotly
          fig = px.imshow(anova_p_values, text_auto=True)
          fig.update_layout(title="Heatmap des valeurs-p de l'ANOVA", xaxis_title="Variables Catégorielles", yaxis_title="Variables Numériques")
          st.plotly_chart(fig)

        # Checkbox pour afficher le commentaire
          if st.checkbox("Afficher le commentaire sur la Corrélation Numérique-Catégorielle", key="Num-Cat"):
              st.markdown("""La majorité des tests ont révélé des relations statistiquement significatives. Des exceptions notables concernent certaines interactions impliquant le jour du dernier contact, bien que quelques-unes d'entre elles, notamment avec le mois du contact, le résultat de la campagne précédente, et la variable cible 'deposit', aient montré une significativité statistique élevée.
Nous avons décidé d'un commun accord le maintien de la variable 'day' dans notre analyse.
""")

#visualisation des corrélations avec la variable cible déposit



    # Commentaires pour chaque corrélation
  correlation_comments = {
  'previous' : "Les clients non contactés auparavant présentent une faible souscription. En revanche, ceux contactés plusieurs fois montrent un taux de souscription plus élevé, suggérant que des efforts marketing répétés favorisent la fidélisation.",
    'pdays' : "Les clients contactés après une longue période (999 jours suggèrent une absence de contact antérieur) souscrivent moins. Ceux contactés plus récemment sont plus enclins à souscrire, mettant en lumière l'importance de contacts réguliers.",
    'default' : "La corrélation entre 'default' et la souscription est modeste (inférieure à 0,5). Les clients en défaut de paiement semblent moins intéressés par les dépôts à terme, probablement en raison de contraintes financières.",
    'campaign' : "La souscription est maximale quand les clients sont contactés 1 à 3 fois. Au-delà, la probabilité de souscrire diminue, indiquant une saturation dans les efforts de communication.",
    'duration' : "La durée de l'appel est un indicateur clé de souscription, avec des appels plus longs corrélant avec une plus grande probabilité de souscription.",
   'day' : "La distribution des souscriptions est relativement uniforme sur le mois, malgré de légères variations qui méritent une analyse plus poussée pour optimiser le timing des contacts.",
    'poutcome' : "Les clients avec un résultat positif ('success') dans la campagne précédente sont beaucoup plus susceptibles de souscrire à nouveau, soulignant l'importance d'une relation client positive et continue.",
    'month' : "Mai est le mois le plus actif en termes de contact, mais Mars, Décembre, Octobre et Septembre se distinguent par une plus haute réussite de souscription, indiquant une saisonnalité dans l'efficacité des campagnes.",
    'loan' : "Les clients sans prêt personnel sont plus susceptibles de souscrire, suggérant que moins de dettes favorise l'intérêt pour de nouveaux services financiers.",
    'housing' : "Les clients sans prêt immobilier ont tendance à souscrire davantage, ce qui peut refléter une plus grande flexibilité financière ou une moindre aversion au risque.",
    'contact' : "La plupart des contacts ont été établis via les téléphones cellulaires des clients, ce qui peut également expliquer une communication efficace, attribuable à la mobilité des téléphones portables",
    'education' : "Les clients avec une éducation tertiaire affichent un taux de souscription plus élevé, indiquant une influence possible du niveau d'éducation sur la décision de souscrire.",
    'marital' : "Les célibataires présentent un taux de souscription légèrement supérieur, suggérant que le célibat peut être un facteur positif pour la souscription aux services.",
    'balance' : "La majorité des clients souscrivant au dépôt à terme possèdent des soldes bancaires entre 0 et 10 000 euros.",
    'age' : "Les distributions d'âge des souscripteurs et des non-souscripteurs au dépôt à terme sont similaires, suggérant que l'âge n'est pas un facteur déterminant majeur pour la souscription.",
    'job' : "Les étudiants, managers et ouvriers présentent une forte propension à souscrire. Bien que la souscription soit généralement bien répartie parmi les différentes professions, ces groupes se distinguent."

  }

    # Variables explicatives à sélectionner pour la visualisation
  variables_to_choose = ['marital', 'education', 'default', 'housing', 'loan', 'month','previous',
                       'poutcome', 'day', 'age', 'job', 'balance', 'contact', 'duration','campaign','pdays']

  #st.subheader("Analyse de la corrélation des variables explicatives et de la variable cible")
  st.markdown(
        """
        <style>
            .highlight {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text with adjusted styling
  st.markdown('<p class="highlight">Analyse de la corrélation des variables explicatives et de la variable cible</p>', unsafe_allow_html=True)
  

    # Choix de la variable explicative pour la corrélation
  selected_variable = st.selectbox("Sélectionnez une variable pour visualiser la corrélation avec 'deposit':",
                                 options=variables_to_choose)




    # Affichage du graphique sélectionné
  if selected_variable:
      fig = get_correlation_plot(selected_variable)
      st.plotly_chart(fig)

    # Affichage du commentaire
  show_comment = st.checkbox("Afficher le commentaire sur la corrélation")
  if show_comment and selected_variable:
      st.info(correlation_comments.get(selected_variable, "Pas de commentaire pour cette variable."))


  #st.subheader("validation Statistique")
  st.markdown(
        """
        <style>
            .highlight {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text with adjusted styling
  st.markdown('<p class="highlight">validation Statistique</p>', unsafe_allow_html=True)
if page == pages[1]:
    # Checkbox pour la première partie (Test statistique du Chi Carré)
  if st.checkbox("Test statistique du Chi Carré"):
     st.markdown("""
    Afin d'évaluer l'impact des variables catégorielles sur la variable cible, nous avons mis en œuvre le test du chi carré. Cette méthode statistique est conçue pour déterminer l'existence d'une corrélation entre deux variables catégorielles. Nos résultats montrent que les valeurs des statistiques de test pour chaque variable catégorielle sont significativement inférieures au seuil de 5%. Cela nous amène à rejeter l'hypothèse nulle, qui postule l'indépendance entre les variables catégorielles et la variable cible (dépôt à terme). Par conséquent, nous concluons que ces variables exercent une influence notable sur la décision des clients de souscrire ou non à un dépôt à terme
    """)
     cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']
     chi2_p_values = {}

     for feature in cat_features:
        if feature != 'deposit':
            contingency_table = pd.crosstab(df[feature], df['deposit'])
            _, p, _, _ = chi2_contingency(contingency_table)
            chi2_p_values[feature] = p

# Conversion en DataFrame pour la visualisation
     chi2_df = pd.DataFrame(list(chi2_p_values.items()), columns=['Feature', 'P-value'])

# Création du graphique à barres
     fig = px.bar(chi2_df, x='Feature', y='P-value', text='P-value')
     fig.update_layout(yaxis=dict(range=[0, 0.05]))
     fig.add_hline(y=0.05, line_dash="dash", line_color="red")

     st.plotly_chart(fig)

    # Checkbox pour la deuxième partie (Test de Student)
if page == pages[1]:
 if st.checkbox("Test de Student"):
    st.markdown("""
        Le test de Student est une méthode statistique employée pour vérifier l'existence d'une relation significative entre des variables numériques et une variable catégorielle. Nos analyses révèlent que les valeurs des statistiques de test (t de Student) pour chaque variable numérique sont toutes inférieures à 5%. Cette observation nous permet d'affirmer avec confiance que les caractéristiques numériques examinées jouent un rôle déterminant dans la décision du client de souscrire ou non à un dépôt à terme.

    """)
    num_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    ttest_p_values = {}

    for feature in num_features:
         group1 = df[df['deposit'] == 'yes'][feature]
         group2 = df[df['deposit'] == 'no'][feature]
         _, p = ttest_ind(group1, group2)
         ttest_p_values[feature] = p

# Conversion des résultats en DataFrame pour la visualisation
    ttest_df = pd.DataFrame(list(ttest_p_values.items()), columns=['Feature', 'P-value'])

# Création d'un graphique Plotly
    fig = px.bar(ttest_df, x='Feature', y='P-value', text='P-value')
    fig.update_layout(yaxis=dict(range=[0, 0.05]), title="Résultats des tests de Student pour les caractéristiques numériques")
    st.plotly_chart(fig)



    # Suite de l'analyse

 st.markdown("""En bref, malgré les informations substantielles fournies par l'analyse exploratoire des variables, il est crucial de noter que la relation statistique ne garantit pas la causalité. Une investigation plus approfondie, telle qu'une modélisation prédictive, serait nécessaire pour comprendre comment ces variables influent réellement sur la souscription aux dépôts à terme.
Nous allons donc procéder à la modélisation de notre jeu de données pour faire de bonnes prédictions, en commençant par le Pre-processing.""")
    
if page == pages[2]:
    st.markdown(
        """
        <style>
            .big-font {
                font-size: 32px !important;
                color: #1E90FF;  /* Dodger Blue */
                text-align: center;
            }
            .highlight {
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .section {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 30px;
                border-radius: 20px;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Section 1: Modélisation
    st.markdown('<p class="big-font">Modélisation</p>', unsafe_allow_html=True)

# Section 2: Preprocessing
    st.markdown('<p class="highlight">Preprocessing</p>', unsafe_allow_html=True)

    
    #st.write("### **Modélisation**")
    #st.write("### I - Preprocessing")

    #On écarte les valeurs -1 de pdays pour ne pas les traiter lors du pre-processing
    pdays_filtered = df['pdays'][df['pdays'] != -1]
# Pour 'campaign'
    Q1_campaign = df['campaign'].quantile(0.25)
    Q3_campaign = df['campaign'].quantile(0.75)
    IQR_campaign = Q3_campaign - Q1_campaign
    Sbas_campaign = Q1_campaign - 1.5 * IQR_campaign
    Shaut_campaign = Q3_campaign + 1.5 * IQR_campaign

# Pour 'pdays' (excluding -1 values)
    Q1_pdays = pdays_filtered.quantile(0.25)
    Q3_pdays = pdays_filtered.quantile(0.75)
    IQR_pdays = Q3_pdays - Q1_pdays
    Sbas_pdays = Q1_pdays - 1.5 * IQR_pdays
    Shaut_pdays = Q3_pdays + 1.5 * IQR_pdays

# Pour 'previous'
    Q1_previous = df['previous'].quantile(0.25)
    Q3_previous = df['previous'].quantile(0.75)
    IQR_previous = Q3_previous - Q1_previous


    Sbas_previous = Q1_previous - 1.5 * IQR_previous
    Shaut_bound_previous = Q3_previous + 1.5 * IQR_previous

#Pour 'Duration'
    Q1_duration = df['duration'].quantile(0.25)
    Q3_duration = df['duration'].quantile(0.75)
    IQR_duration = Q3_duration - Q1_duration
    Sbas_duration = Q1_duration - 1.5 * IQR_duration
    Shaut_bound_duration = Q3_duration + 1.5 * IQR_duration

    moyenne_pdays = pdays_filtered.mean()
    moyenne_campaign = df['campaign'].mean()
    moyenne_previous = df['previous'].mean()
    moyenne_duration = df['duration'].mean()

# Remplacer les valeurs aberrantes de 'pdays' par sa moyenne (en excluant les valeurs -1)
    df.loc[(df['pdays'] > Shaut_pdays) & (df['pdays'] != -1), 'pdays'] = moyenne_pdays

# Remplacer les valeurs aberrantes de 'campaign' par sa moyenne
    df.loc[df['campaign'] > Shaut_campaign, 'campaign'] = moyenne_campaign

# Remplacer les valeurs aberrantes de 'previous' par la moyenne de 'campaign'
    df.loc[df['previous'] > Shaut_bound_previous, 'previous'] = moyenne_previous

# Remplacer les valeurs aberrantes de 'duration' par la moyenne de 'campaign'
    df.loc[df['duration'] > Shaut_bound_duration, 'duration'] = moyenne_duration

  #Transformation des colonnes age et balance

    age_bins = [18, 25, 35, 50, 65, 100]
    age_labels = ["18_25", "25_35", "35_50", "50_65", "65_100"]
# On applique le changement sur le dataset pour creer la colonne
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
#Création du bins et des étiquettes
    balance_bins = [-6848, 0, 122, 550, 1708, 81205]
    balance_labels = ["negatif", "tres_faible", "faible", "moyen", "eleve"]
# Cut the balance column into bins
    df['balance_group'] = pd.cut(df['balance'], bins=balance_bins, labels=balance_labels, right=False)
# On applique le changement sur le dataset pour creer la colonne
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    df['age_group'] = df['age_group'].astype('object')
    df['balance_group'] = df['balance_group'].astype('object')

    # Séparation des données en ensembles d'entraînement et de test
    X = df.drop(columns=['deposit'])
    y = df['deposit']
    TEST_SIZE = 0.25
    RAND_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RAND_STATE)

# Encodage de la variable cible
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

# Sélection des colonnes catégorielles
    categorical_columns = X_train.select_dtypes(include=['object']).columns

# Encodage des caractéristiques catégorielles
    encoder = OneHotEncoder(drop=None, sparse=False)

# Utilisation de  fit sur l'ensemble d'entraînement
    encoder.fit(X_train[categorical_columns])

# Transformations des ensembles d'entraînement et de test
    encoded_train = encoder.transform(X_train[categorical_columns])
    encoded_test = encoder.transform(X_test[categorical_columns])

# Conversion des caractéristiques encodées en dataframes
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns))
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

# Fusion des dataframes encodés avec les originaux
    X_train_encoded = X_train.drop(columns=categorical_columns).reset_index(drop=True).merge(encoded_train_df, left_index=True, right_index=True)
    X_test_encoded = X_test.drop(columns=categorical_columns).reset_index(drop=True).merge(encoded_test_df, left_index=True, right_index=True)

# Suppression des colonnes inutiles
    X_train_encoded = X_train_encoded.drop(columns=['balance', 'age'])
    X_test_encoded  = X_test_encoded.drop(columns=['balance', 'age'])

  #Création d'un X_train et X_test normalisé pour les besoins de certains modéles

  # Identifier les colonnes numériques
    cols_numeriques = X_train_encoded.select_dtypes(include=['int64', 'float64']).columns
    X_train_normalised = X_train_encoded
    X_test_normalised = X_test_encoded
# Initialiser le StandardScaler
    scaler = StandardScaler()

# Normaliser les colonnes numériques dans l'ensemble d'entraînement
    X_train_normalised[cols_numeriques] = scaler.fit_transform(X_train_encoded[cols_numeriques])

# Normaliser les colonnes numériques dans l'ensemble de test
    X_test_normalised[cols_numeriques] = scaler.transform(X_test_encoded[cols_numeriques])


    st.write("#### A - Étapes du prétraitement des données")

    # Étape 1: Gestion des valeurs aberrantes
    #st.write("**Étape 1 - Gestion des valeurs aberrantes**")

    # Expliquer la méthode IQR
    with st.expander("**Étape 1 - Gestion des valeurs aberrantes**"):
        highlighted_text = "La méthode des IQR nous a permis de remplacer les valeurs extrêmes par la moyenne respective de chaque variable."

        # Colored box with highlighted text
        colored_box = f'<div style="background-color:#ECECEC; padding:10px; border-radius:5px;">{highlighted_text}</div>'

        # Display the colored box with highlighted text
        st.markdown(colored_box, unsafe_allow_html=True)

    # Étape 2: Transformation des colonnes 'age' et 'balance'
    #st.write("**Étape 2 - Transformation des colonnes 'age' et 'balance'**")

    # Expliquer la transformation pour atténuer l'impact des valeurs extrêmes
    with st.expander("**Étape 2 - Transformation des colonnes 'age' et 'balance'**"):
        highlighted_text = "Afin d'atténuer l'impact des valeurs extrêmes tout en les conservant pour ne pas perdre plus de données."

        # Colored box with highlighted text
        colored_box = f'<div style="background-color:#ECECEC; padding:10px; border-radius:5px;">{highlighted_text}</div>'

        # Display the colored box with highlighted text
        st.markdown(colored_box, unsafe_allow_html=True)

    # Étape 3: Séparation en ensembles d'entraînement et de test
    #st.write("**Étape 3 - Séparation en ensembles d'entraînement et de test**")

    # Expliquer la séparation des ensembles
    with st.expander("**Étape 3 - Séparation en ensembles d'entraînement et de test**"):
        highlighted_text = "Nous avons séparé les données en ensembles d'entraînement et de test pour évaluer notre modèle."

        # Colored box with highlighted text
        colored_box = f'<div style="background-color:#ECECEC; padding:10px; border-radius:5px;">{highlighted_text}</div>'

        # Display the colored box with highlighted text
        st.markdown(colored_box, unsafe_allow_html=True)

    # Étape 4: Encodage de la variable cible
    #st.write("**Étape 4 - Encodage de la variable cible**")

    # Expliquer l'encodage de la variable cible 'deposit'
    with st.expander("**Étape 4 - Encodage de la variable cible**"):
        highlighted_text = "La variable cible 'deposit' a été encodée en valeurs numériques."

        # Colored box with highlighted text
        colored_box = f'<div style="background-color:#ECECEC; padding:10px; border-radius:5px;">{highlighted_text}</div>'

        # Display the colored box with highlighted text
        st.markdown(colored_box, unsafe_allow_html=True)

    # Étape 5: Encodage One-Hot des variables catégorielles
    #st.write("**Étape 5 - Encodage One-Hot des variables catégorielles**")

    # Expliquer l'encodage One-Hot des variables catégorielles
    with st.expander("**Étape 5 - Encodage One-Hot des variables catégorielles**"):
        highlighted_text = "Les variables catégorielles ont été transformées en utilisant l'encodage One-Hot."

        # Colored box with highlighted text
        colored_box = f'<div style="background-color:#ECECEC; padding:10px; border-radius:5px;">{highlighted_text}</div>'

        # Display the colored box with highlighted text
        st.markdown(colored_box, unsafe_allow_html=True)

    # Étape 6: Nettoyage final
    #st.write("**Étape 6 - Nettoyage final**")

    # Expliquer le nettoyage final
    with st.expander("**Étape 6 - Nettoyage final**"):
        highlighted_text = "Les colonnes redondantes ou non nécessaires (« balance » et « age ») ont été retirées des ensembles de données."

        # Colored box with highlighted text
        colored_box = f'<div style="background-color:#ECECEC; padding:10px; border-radius:5px;">{highlighted_text}</div>'

        # Display the colored box with highlighted text
        st.markdown(colored_box, unsafe_allow_html=True)
        
    st.write("#### B - Quelques visualisations pour notre preprocessing")

    # On cree la variable du checkboxe
    show_boxplots = st.checkbox("Visualisez les boxplots des variables affichant des valeurs extrêmes et abérantes avant leurs traitements")
    
    # On ajoute un séparateur entre les boxplots et le tableau
    st.markdown("---")

    # On cree les boxplots à l'aide d'une fonction et on génère le tout sur deux ligne

    # La fonction pour génerer et afficher les boxplots
    def generate_boxplot(column_name, title, axis):
       sns.boxplot(data=df, x=column_name, ax=axis)
       axis.set_title(f"Boxplot of {column_name}")

    if show_boxplots:
    # On crée un subplot avec les boxplots
        # On crée un subplot avec deux lignes
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # Boxeplot sur la première ligne
        generate_boxplot("pdays", "pdays", ax[0, 0])
        generate_boxplot("campaign", "campaign", ax[0, 1])

    # Boxeplot sur la deuxième ligne
        generate_boxplot("previous", "previous", ax[1, 0])
        generate_boxplot("duration", "duration", ax[1, 1])
        plt.tight_layout()
        st.pyplot(fig)

    # On ajoute un séparateur entre les boxplots et le tableau
    #On definit la variable checboxe
    button_show_data = st.checkbox("Affichez le nouveau jeu de données résultant du preprocessing")

    # Condition pour afficher les données grace au checkboxe
    if button_show_data:
        buffer = io.StringIO()
        X_train_encoded.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    st.markdown("---")


      # Set the background color and font size for the text
    st.markdown(
        """
        <style>
            .highlight {
                background-color: #F0F8FF;  /* Alice Blue */
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 24px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Text with adjusted styling
    st.markdown('<p class="highlight">Machine Learning</p>', unsafe_allow_html=True)


    #st.write("### Machine Learning")
    st.write("#### Nous avons entrainé et testé les models suivants :")
    st.write("**- Regression Logistique**")
    st.write("**- KNN**")
    st.write("**- Decision Tree**")
    st.write("**- Random Forest**")
    st.write("**- XGBoost**")

    # On cree un selecteur qui nouspermettra de choisir parmi les differents modèles

    def plot_learning_curve(train_sizes, train_scores, test_scores, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Learning Curve - {title}")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax.legend(loc="best")

        return fig

    # 
    st.write("#### Evaluation des modèles Machine Learning")

    # Choix du modèle
    model_choisi = st.selectbox(label="Choisissez un modèle à évaluer",
                                options=['-- Sélectionnez un modèle --', 'Regression Logistique', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'])
    model_folder = 'C:\\Users\\hbago\\OneDrive\\Bureau\\Data\\Bank_market\\'
    loaded_bst = xgboost.Booster()

    # On charge le modèle selectionné 
    if model_choisi == 'Regression Logistique':
        model = joblib.load('LogisticRegression')
        y_pred = model.predict(X_test_normalised)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {accuracy:.2%}")
        train_sizes, train_scores, test_scores = learning_curve(model, X_train_normalised, y_train, n_jobs=-1,
                                                                train_sizes=np.linspace(.1, 1.0, 5))

        # On affiche le composant select pour choisir le metric desiré
        selected_metric = st.selectbox("Sélectionnez une métrique d'evaluation du modèle choisi",
                                       options=['-- Sélectionnez une métric --', 'Learning Curve', 'Confusion Matrix', 'Classification Report', 'ROC Curve'])

        # On affiche l'information choisie
        if selected_metric == 'Learning Curve':
            st.pyplot(plot_learning_curve(train_sizes, train_scores, test_scores, model_choisi))
        elif selected_metric == 'Confusion Matrix':
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            # On affiche la confusion matrice avec une heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Classe prédite")
            plt.ylabel("Classe réelle")
            st.pyplot(plt)
        elif selected_metric == 'Classification Report':
            st.text(classification_report(y_test, y_pred))
        elif selected_metric == 'ROC Curve':
            fig, ax = plt.subplots(figsize=(8, 8))
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test_normalised)[:, 1]
                fpr, tpr, seuils = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
                ax.axhline(y=0.35, color='red', lw=2, label='Seuil = 0.35')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.warning("This option is only available for classifiers that support predict_proba (e.g., Logistic Regression).")

        with st.expander("Observation"):
            highlighted_text = "Ceci est un texte de test, à votre avis serait-il pertinent de faire de petites observations comme ceci pour chaque evaluation ?"
        # Colored box avec mise en avant du texte
            colored_box = f'<div style="background-color:#ADD8E6; padding:10px; border-radius:5px;">{highlighted_text}</div>'
        # On affiche le Colored box avec mise en avant du texte
            st.markdown(colored_box, unsafe_allow_html=True)

    elif model_choisi == 'KNN':
        model = joblib.load('knn_ma')
        X_test_contiguous = np.ascontiguousarray(X_test_normalised.values)
        y_pred = model.predict(X_test_contiguous)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {accuracy:.2%}")
        train_sizes, train_scores, test_scores = learning_curve(model, X_train_normalised.values, y_train, n_jobs=-1,
                                                                train_sizes=np.linspace(.1, 1.0, 5))

        # On affiche le composant select pour choisir le metric desiré
        selected_metric = st.selectbox("Sélectionnez une métrique d'evaluation du modèle choisi",
                                       options=['-- Sélectionnez une métric --', 'Learning Curve', 'Confusion Matrix', 'Classification Report', 'KNN Metrics'])

        if selected_metric == 'Learning Curve':
            st.pyplot(plot_learning_curve(train_sizes, train_scores, test_scores, model_choisi))
        elif selected_metric == 'Confusion Matrix':
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            #On affiche la confusion matrice avec une heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Classe prédite")
            plt.ylabel("Classe réelle")
            st.pyplot(plt)
        elif selected_metric == 'Classification Report':
            st.text(classification_report(y_test, y_pred))
        
        elif selected_metric == 'KNN Metrics':
        #On génère le graphe sur streamlit grace à la fonction
            st.subheader("KNN Accuracy Scores with Different Distance Metrics")
            #plot_knn_scores(X_train_normalised, y_train, X_test_normalised, y_test)
            st.pyplot(plot_knn_scores(X_train_normalised, y_train, X_test_normalised, y_test))

    elif model_choisi == 'Decision Tree':
        with open('clf_dt_gini.dill', 'rb') as f:
            clf_dt_ginis = dill.load(f)
            # On permet à l'utilisateur de modifier la profondeur
            max_depth = st.slider("Maximum Depth", min_value=3, max_value=20, value=10)

            # On reentraine le Decision Tree model avec la max_depth souhaitée
            clf_dt_ginis = DecisionTreeClassifier(max_depth=max_depth, criterion ='gini')
            clf_dt_ginis.fit(X_train_normalised, y_train)

            y_pred = clf_dt_ginis.predict(X_test_encoded)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {accuracy:.2%}")
            train_sizes, train_scores, test_scores = learning_curve(clf_dt_ginis, X_train_normalised, y_train,
                                                                n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
            

        # On affiche le composant select pour choisir le metric desiré
        selected_metric = st.selectbox("Sélectionnez une métrique d'evaluation du modèle choisi",
                                       options=['-- Sélectionnez une métric --', 'Learning Curve', 'Confusion Matrix', 'Classification Report', 'Decision Tree'])

        if selected_metric == 'Decision Tree':
            if model_choisi == 'Decision Tree':
                plt.figure(figsize=(20, 10))
                if isinstance(clf_dt_ginis, DecisionTreeClassifier):
                    plot_tree(clf_dt_ginis, filled=True, feature_names=X_train_encoded.columns,
                              class_names=["No", "Yes"], rounded=True)
                    st.pyplot(plt.gcf())
                else:
                    st.warning("Cette option n'est possible que pour le Model Decision Tree.")

        elif selected_metric == 'Learning Curve':
            st.pyplot(plot_learning_curve(train_sizes, train_scores, test_scores, model_choisi))

        elif selected_metric == 'Confusion Matrix':
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            # On affiche la confusion matrice avec une heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Classe prédite")
            plt.ylabel("Classe réelle")
            st.pyplot(plt)

        elif selected_metric == 'Classification Report':
            st.text(classification_report(y_test, y_pred))

    elif model_choisi == 'Random Forest':
        with open('random_forest_model.dill', 'rb') as f:
            clf_optimizedd = dill.load(f)
            y_pred = clf_optimizedd.predict(X_test_encoded)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {accuracy:.2%}")
            train_sizes, train_scores, test_scores = learning_curve(clf_optimizedd, X_train_encoded, y_train,
                                                                    n_jobs=-1,
                                                                    train_sizes=np.linspace(.1, 1.0, 5))

        # On affiche la metric a selectionner
            selected_metric = st.selectbox("Sélectionnez une métrique d'evaluation du modèle choisi", 
                                           options=['-- Sélectionnez une métric --', 'Learning Curve', 'Confusion Matrix', 'Classification Report'])

            if selected_metric == 'Learning Curve':
                st.pyplot(plot_learning_curve(train_sizes, train_scores, test_scores, model_choisi))
            
            elif selected_metric == 'Confusion Matrix':
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                # On affiche la confusion matrice avec une heatmap
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Classe prédite")
                plt.ylabel("Classe réelle")
                st.pyplot(plt)
            elif selected_metric == 'Classification Report':
                st.text(classification_report(y_test, y_pred))

    elif model_choisi == 'XGBoost':
     # On charge le model xgboost
      XGBoost = joblib.load("xgb_optimized")
      X_test_encoded = xgboost.DMatrix(X_test_encoded)
      loaded_bst = xgboost.Booster() 

      loaded_bst.load_model('xgb_optimizedbst.model')
      y_pred = loaded_bst.predict(X_test_encoded)
      #y_pred2 = XGBoost.predict(X_test_encoded)
      y_pred_labels = (y_pred > 0.5).astype(int)
      accuracy = accuracy_score(y_test, y_pred_labels)
      st.success(f"Accuracy: {accuracy:.2%}")
      train_sizes, train_scores, test_scores = learning_curve(XGBoost, X_train_encoded, y_train, n_jobs=-1,
                                                            train_sizes=np.linspace(.1, 1.0, 5))

    # On affiche la metric a selectionner
      selected_metric = st.selectbox("Sélectionnez une métrique d'evaluation du modèle choisi", 
                                     options=['-- Sélectionnez une métric --', 'Learning Curve', 'Confusion Matrix', 'Classification Report'])

      if selected_metric == 'Learning Curve':
        st.pyplot(plot_learning_curve(train_sizes, train_scores, test_scores, model_choisi))

      elif selected_metric == 'Confusion Matrix':
          st.subheader("Confusion Matrix")
          cm = confusion_matrix(y_test, y_pred_labels)
          # On affiche la confusion matrice avec une heatmap
          fig, ax = plt.subplots(figsize=(8, 6))
          sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
          ax.set_xlabel("Classe prédite")
          ax.set_ylabel("Classe réelle")
          st.pyplot(fig)
      elif selected_metric == 'Classification Report':
        st.text(classification_report(y_test, y_pred_labels))

    
if page == pages[3] :
  st.header("Prédictions")
  df = pd.read_csv('bank.csv')
    #On écarte les valeurs -1 de pdays pour ne pas les traiter lors du pre-processing
  st.write("Ce script démontre l'utilisation standard de Streamlit et XGBoost pour développer une application web \
  interactive axée sur les prédictions, en utilisant des données fournies par l'utilisateur.")

  if st.checkbox("Informations complémentaires"):
    st.write("- ***Choix des caractéristiques par l'utilisateur :***")
    st.write("Le script emploie la fonction st.selectbox de Streamlit pour générer des menus déroulants. \
    Ces menus permettent aux utilisateurs de sélectionner des options pour divers attributs tels que le métier, \
    le mois, l'éducation etc. à partir d'un DataFrame nommé df.")
    st.write("Pour chaque attribut sélectionné, le script crée une colonne correspondante dans un autre DataFrame \
    encoded_data.")
    st.write("La catégorie choisir reçoit la valeur 1, tandis que toutes les autres catégories reçoivent la valeur 0.")
    st.write("- ***Préparation des données complémentaires :***")
    st.write("Le script assigne automatiquement des valeurs par défaut à certaines colonnes, telles que age_group et \
    balance_group, basées sur des catégories pré-établies. Il utilise également des statistiques descriptives telles que \
    la médiane et la moyenne du DataFrame df pour compléter d'autres colonnes, notamment day, duration, pdays, campaign et previous.")
    st.write("- ***Finalisation de la préparation des données :***")
    st.write("Avant la prédiction, le DataFrame encoded_data est réorganisé pour correspondre à la structure requise par le \
    modèle prédictif.")
    st.write("- ***Processus de Prédiction :***")
    st.write("Lorsque l'utilisateur clique sur le bouton 'Prédictions', le modèle génère une prédiction basée sur les données \
    entrées. Le modèle XGBoost, utilisé pour la prédiction, fournit des probabilités pour une classification binaire.")
    st.write("Un seuil spécifique, comme 0.5, est appliqué pour déterminer la classe prédite.")
  
  pdays_filtered = df['pdays'][df['pdays'] != -1]
    # Pour 'campaign'
  Q1_campaign = df['campaign'].quantile(0.25)
  Q3_campaign = df['campaign'].quantile(0.75)
  IQR_campaign = Q3_campaign - Q1_campaign
  Sbas_campaign = Q1_campaign - 1.5 * IQR_campaign
  Shaut_campaign = Q3_campaign + 1.5 * IQR_campaign
    # Pour 'pdays' (excluding -1 values)
  Q1_pdays = pdays_filtered.quantile(0.25)
  Q3_pdays = pdays_filtered.quantile(0.75)
  IQR_pdays = Q3_pdays - Q1_pdays
  Sbas_pdays = Q1_pdays - 1.5 * IQR_pdays
  Shaut_pdays = Q3_pdays + 1.5 * IQR_pdays
    # Pour 'previous'
  Q1_previous = df['previous'].quantile(0.25)
  Q3_previous = df['previous'].quantile(0.75)
  IQR_previous = Q3_previous - Q1_previous
  Sbas_previous = Q1_previous - 1.5 * IQR_previous
  Shaut_bound_previous = Q3_previous + 1.5 * IQR_previous
    #Pour 'Duration'
  Q1_duration = df['duration'].quantile(0.25)
  Q3_duration = df['duration'].quantile(0.75)
  IQR_duration = Q3_duration - Q1_duration
  Sbas_duration = Q1_previous - 1.5 * IQR_duration
  Shaut_bound_duration = Q3_duration + 1.5 * IQR_duration
  moyenne_pdays = pdays_filtered.mean()
  moyenne_campaign = df['campaign'].mean()
  moyenne_previous = df['previous'].mean()
  moyenne_duration = df['duration'].mean()
    # Remplacer les valeurs aberrantes de 'pdays' par sa moyenne (en excluant les valeurs -1)
  df.loc[(df['pdays'] > Shaut_pdays) & (df['pdays'] != -1), 'pdays'] = moyenne_pdays
    # Remplacer les valeurs aberrantes de 'campaign' par sa moyenne
  df.loc[df['campaign'] > Shaut_campaign, 'campaign'] = moyenne_campaign
    # Remplacer les valeurs aberrantes de 'previous' par la moyenne de 'campaign'
  df.loc[df['previous'] > Shaut_bound_previous, 'previous'] = moyenne_previous
    # Remplacer les valeurs aberrantes de 'duration' par la moyenne de 'campaign'
  df.loc[df['duration'] > Shaut_bound_duration, 'duration'] = moyenne_duration
    
    # Charger le modèle
  encoded_data = pd.DataFrame(index=[0])
    
  job = st.selectbox('Job', df['job'].unique())
  encoded_data['job_' + job] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for job_category in df['job'].unique():
      if job_category != job:
          encoded_data['job_' + job_category] = 0
  month = st.selectbox('Month', df['month'].unique())
  encoded_data['month_' + month] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for month_category in df['month'].unique():
      if month_category != month:
          encoded_data['month_' + month_category] = 0
            
  education = st.selectbox('Education', df['education'].unique())
  encoded_data['education_' + education] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for education_category in df['education'].unique():
      if education_category != education:
          encoded_data['education_' + education_category] = 0
  default = st.selectbox('Default', df['default'].unique())
  encoded_data['default_' + default] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for default_category in df['default'].unique():
      if default_category != default:
          encoded_data['default_' + default_category] = 0
            
  marital = st.selectbox('Marital', df['marital'].unique())
  encoded_data['marital_' + marital] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for marital_category in df['marital'].unique():
      if marital_category != marital:
          encoded_data['marital_' + marital_category] = 0
            
  housing = st.selectbox('Housing', df['housing'].unique())
  encoded_data['housing_' + housing] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for housing_category in df['housing'].unique():
      if housing_category != housing:
          encoded_data['housing_' + housing_category] = 0
  loan = st.selectbox('Loan', df['loan'].unique())
  encoded_data['loan_' + loan] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for loan_category in df['loan'].unique():
      if loan_category != loan:
          encoded_data['loan_' + loan_category] = 0
            
  contact = st.selectbox('Contact', df['contact'].unique())
  encoded_data['contact_' + contact] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for contact_category in df['contact'].unique():
      if contact_category != contact:
          encoded_data['contact_' + contact_category] = 0
  poutcome = st.selectbox('poutcome', df['poutcome'].unique())
  encoded_data['poutcome_' + poutcome] = 1
# Remplir les autres colonnes de la DataFrame encodée avec des zéros
  for poutcome_category in df['poutcome'].unique():
      if poutcome_category != poutcome:
          encoded_data['poutcome_' + poutcome_category] = 0
    
  encoded_data['balance_group_faible'] = 0
  encoded_data['age_group_25_35'] = 0
  encoded_data['age_group_50_65'] = 0
  encoded_data['balance_group_eleve'] = 0
  encoded_data['balance_group_tres_faible'] = 0
  encoded_data['balance_group_moyen'] = 0
  encoded_data['age_group_65_100'] = 0
  encoded_data['balance_group_negatif'] = 0
  encoded_data['age_group_35_50'] = 0
  encoded_data['age_group_18_25'] = 0
    
  column_mapping = {
    "18_25": "age_group_18_25",
    "25_35": "age_group_25_35",
    "35_50": "age_group_35_50",
    "50_65": "age_group_50_65",
    "65_100": "age_group_65_100"
}
  age_options = {k: v for k, v in column_mapping.items()}
# Sélectionner la catégorie de "age_group" choisie par l'utilisateur
  selected_age_group = st.selectbox('Sélectionnez la catégorie de "age_group"', list(age_options.keys()))
    
# Récupérer le nom de la colonne encodée correspondant à la valeur sélectionnée
    #encoded_data[selected_age_group] = 0
  selected_age_column = column_mapping[selected_age_group]
  encoded_data[selected_age_column] = 1
# Répéter le processus pour la catégorie de "balance_group" choisie par l'utilisateur
  column_mapping_balance = {
    "eleve": "balance_group_eleve",
    "faible": "balance_group_faible",
    "moyen": "balance_group_moyen",
    "negatif": "balance_group_negatif",
    "tres_faible": "balance_group_tres_faible"
}
  balance_options = {k: v for k, v in column_mapping_balance.items()}
# Sélectionner la catégorie de "balance_group" choisie par l'utilisateur à partir des options inversées
    
  selected_balance_group = st.selectbox('Sélectionnez la catégorie de "balance_group"', list(balance_options.keys()))
# Créer une nouvelle colonne dans encoded_data pour la catégorie sélectionnée
  selected_balance_column = column_mapping_balance[selected_balance_group]
  encoded_data[selected_balance_column] = 1
  encoded_data['day'] = df['day'].median().astype(int)
  encoded_data['duration'] = df['duration'].mean().astype(int)
  encoded_data['pdays'] = df['pdays'].mean().astype(int)
  encoded_data['campaign'] = df['campaign'].mean().astype(int)
  encoded_data['previous'] = df['previous'].mean().astype(int)
  cols = [
    "day", "duration", "campaign", "pdays", "previous", "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid",
    "job_management", "job_retired", "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
    "job_unknown", "marital_divorced", "marital_married", "marital_single", "education_primary", "education_secondary",
    "education_tertiary", "education_unknown", "default_no", "default_yes", "housing_no", "housing_yes", "loan_no",
    "loan_yes", "contact_cellular", "contact_telephone", "contact_unknown", "month_apr", "month_aug", "month_dec",
    "month_feb", "month_jan", "month_jul", "month_jun", "month_mar", "month_may", "month_nov", "month_oct", "month_sep",
    "poutcome_failure", "poutcome_other", "poutcome_success", "poutcome_unknown", "age_group_18_25", "age_group_25_35",
    "age_group_35_50", "age_group_50_65", "age_group_65_100", "balance_group_eleve", "balance_group_faible",
    "balance_group_moyen", "balance_group_negatif", "balance_group_tres_faible"
]
  encoded_data = encoded_data[cols]
  with open('xgb_optimizedpickle', 'rb') as model_file:
      model = pickle.load(model_file)
  #model = xgboost.Booster()
  #model.load_model("xgb_optimizedbst.model")
  #dtest = xgboost.DMatrix(encoded_data)
  if st.button('Predictions'):
      prediction = model.predict(encoded_data)
      #st.write("Probabilités de prédiction :", prediction)

    # Créer un histogramme des probabilités
      #fig, ax = plt.subplots()
     #ax.hist(prediction, bins=10, range=(0,1))
      #ax.set_title("Distribution des Probabilités de Prédiction")
      #ax.set_xlabel("Probabilité")
      #ax.set_ylabel("Nombre de Prédictions")

    # Afficher l'histogramme dans Streamlit
      #st.pyplot(fig)

    # XGBoost donne des probabilités pour la classification binaire, donc vous devez définir un seuil
    # Par exemple, si la prédiction est supérieure à 0.5, on considère que la classe prédite est 1
      predicted_class = (prediction > 0.5).astype(int)
      if predicted_class[0] == 1:
          st.info("La prédiction est : Yes")
      else:
          st.info("La prédiction est : No")
        
   # prediction = model.predict(encoded_data)
    #if prediction == 1:
       # st.markdown("**La prédiction est : Yes**")
   # else:
       # st.mardown("**La prédiction est : No**")
if page == pages[4] :
  st.header("Utilisation professionnelle du projet")
  df = pd.read_csv('Banktest.csv')
  
  st.write("Partant du principe qu'il est essentiel pour un service marketing de pouvoir contacter les clients efficacement, \
  nous avons utilisé un nouveau jeu de données dérivé du nôtre.")
  st.write("Ce dataset a été complètement randomisé et enrichi de deux nouvelles colonnes : Prénom et Téléphone.")
  st.write("À partir de ce jeu de données, nous avons créé un script qui, en exploitant un fichier clients, \
  identifie les 50 clients les plus susceptibles de souscrire à un compte à terme.")
  
if page == pages[4]:
  if st.checkbox("Informations complémentaires"):
    st.write("-- Création et Modification d'un Dataset:")
    st.write("Nouvelles Colonnes : Ajout de deux colonnes, 'Prénom' et 'Téléphone', à notre dataset existant.")
    st.write("- Réaffectation des Données : Les données de chaque colonne sont redistribuées aléatoirement en utilisant \
    la méthode .sample 'frac=1' pour assurer un mélange complet.")
    st.write("-- Chargement et Préparation du Modèle XGBoost:")
    st.write("- Initialisation : Création d'une instance du modèle XGBoost.")
    st.write("- Chargement du Modèle : Importation du modèle pré-entraîné nommé xgb_optimizedbst.model.")
    st.write("-- Préparation des Données pour la Prédiction :")
    st.write("- Conversion des Données : Transformation des données encodées (dans 'encoded_df') en un format compatible avec XGBoost (DMatrix).")
    st.write("-- Réalisation des Prédictions avec XGBoost :")
    st.write("- Prédiction : Utilisation du modèle pour générer des prédictions sur l'ensemble de test 'dtest'.")
    st.write("- Ajout des Prédictions : Les prédictions (y_pred) sont ajoutées en tant que nouvelle colonne ('prediction') dans le DataFrame 'df'.")
    st.write("-- Concaténation, Tri et Affichage des Résultats :")
    st.write("- Concaténation : Fusion des colonnes 'Prénom' et 'Téléphone' avec 'df' pour intégrer ces informations.")
    st.write("- Colonne de Probabilité : Ajout d'une colonne 'probability' pour indiquer la probabilité de la classe positive, \
    étant donné que le modèle est un classificateur binaire.")
    st.write("- Tri : Organisation du DataFrame 'df' par ordre décroissant de probabilité.")
    st.write("- Sélection des Données : Réduction du DataFrame trié pour ne conserver que les colonnes 'Prénom', 'Téléphone' et 'Probability'.")
    st.write("- Affichage : Présentation des 50 premiers clients avec st.dataframe(df_sorted.head(50)), dans l'application Streamlit.")
  st.dataframe(df)
  def calculate_outlier_bounds(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
  def replace_outliers_with_mean(df, column, upper_bound):
      mean_value = df[column].mean()
      df.loc[df[column] > upper_bound, column] = mean_value
  def encode_categorical_features(df, categorical_columns):
      encoder = OneHotEncoder(drop=None, sparse=False)
      encoder.fit(df[categorical_columns])
      encoded_df_2 = encoder.transform(df[categorical_columns])
      encoded_df = pd.DataFrame(encoded_df_2, columns=encoder.get_feature_names_out(categorical_columns))
      return encoded_df
  columns_to_convert = ['day', 'duration', 'campaign', 'pdays', 'previous']
  for column in columns_to_convert:
      df[column] = df[column].astype(int)
    # Save and drop the columns "prénom" and "téléphone"
  df_prenom_telephone = df[['prénom', 'téléphone']]
  if 'pdays' not in df:
      st.write("XDLa colonne 'pdays' a disparu!")
  df = df.drop(columns=['prénom', 'téléphone'])
  if 'pdays' not in df:
      st.write("VLa colonne 'pdays' a disparu!")
    # Filter for pdays column
  pdays_filtered = df['pdays'][df['pdays'] != -1]
  if 'pdays' not in df:
      st.write("XLa colonne 'pdays' a disparu!")
    # Calculate outlier bounds for the respective columns
  _, upper_campaign = calculate_outlier_bounds(df, 'campaign')
  _, upper_pdays = calculate_outlier_bounds(df, 'pdays')
  _, upper_previous = calculate_outlier_bounds(df, 'previous')
  _, upper_duration = calculate_outlier_bounds(df, 'duration')
  if 'pdays' not in df:
       st.write("6La colonne 'pdays' a disparu!")
    # Replace outliers with mean
  replace_outliers_with_mean(df, 'pdays', upper_pdays)
  replace_outliers_with_mean(df, 'campaign', upper_campaign)
  replace_outliers_with_mean(df, 'previous', upper_previous)
  replace_outliers_with_mean(df, 'duration', upper_duration)
    
    # Bin 'age' and 'balance' columns
  age_bins = [18, 25, 35, 50, 65, 100]
  age_labels = ["18_25", "25_35", "35_50", "50_65", "65_100"]
  df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype('object')
    
  balance_bins = [-6848, 0, 122, 550, 1708, 81205]
  balance_labels = ["negatif", "tres_faible", "faible", "moyen", "eleve"]
  df['balance_group'] = pd.cut(df['balance'], bins=balance_bins, labels=balance_labels, right=False).astype('object')
    
    # Encode categorical columns
  categorical_columns = df.select_dtypes(include=['object']).columns
  encoded_df = encode_categorical_features(df, categorical_columns)
  columns_to_add = ['day', 'duration', 'campaign', 'pdays', 'previous']
  encoded_df = pd.concat([df[columns_to_add], encoded_df], axis=1)
    
  cols = [
    "day", "duration", "campaign", "pdays", "previous", "job_admin.", "job_blue-collar", "job_entrepreneur", "job_housemaid",
    "job_management", "job_retired", "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
    "job_unknown", "marital_divorced", "marital_married", "marital_single", "education_primary", "education_secondary",
    "education_tertiary", "education_unknown", "default_no", "default_yes", "housing_no", "housing_yes", "loan_no",
    "loan_yes", "contact_cellular", "contact_telephone", "contact_unknown", "month_apr", "month_aug", "month_dec",
    "month_feb", "month_jan", "month_jul", "month_jun", "month_mar", "month_may", "month_nov", "month_oct", "month_sep",
    "poutcome_failure", "poutcome_other", "poutcome_success", "poutcome_unknown", "age_group_18_25", "age_group_25_35",
    "age_group_35_50", "age_group_50_65", "age_group_65_100", "balance_group_eleve", "balance_group_faible",
    "balance_group_moyen", "balance_group_negatif", "balance_group_tres_faible"
]
  encoded_df = encoded_df[cols]
    # Load the trained model and predict
  with open('xgb_optimizedpickle', 'rb') as model_file:
      model = pickle.load(model_file)
  y_pred = model.predict(encoded_df)
  df['prediction'] = y_pred
    
    # Concatenate the columns "prénom" and "téléphone" and sort by prediction
  df = pd.concat([df_prenom_telephone, df], axis=1)
    
    #df_sorted = df.sort_values(by='prediction', ascending=False)
  y_proba = model.predict_proba(encoded_df)
  df['probability'] = y_proba[:,1]  # Pour une classification binaire, cela donnerait la probabilité de la classe 1
  df_sorted = df.sort_values(by='probability', ascending=False)
  df_sorted = df_sorted[['prénom', 'téléphone','probability']]
# Display the top 50 clients
  st.dataframe(df_sorted.head(50))
  prediction = model.predict(encoded_df)
# Créer un histogramme des probabilités
  fig, ax = plt.subplots()
  ax.hist(prediction, bins=10, range=(0,1))
  ax.set_title("Distribution des Probabilités de Prédiction")
  ax.set_xlabel("Probabilité")
  ax.set_ylabel("Nombre de Prédictions")

    # Afficher l'histogramme dans Streamlit
  st.pyplot(fig)
