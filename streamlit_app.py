import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.svm import SVC
import xgboost
from sklearn.model_selection import train_test_split #split
from sklearn.metrics import accuracy_score #metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import io
from io import StringIO
import plotly.graph_objs as go
import plotly.express as px
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from joblib import load
from sklearn.preprocessing import StandardScaler
import joblib
import dill
import statsmodels
import numpy as np
import itertools
import plotly.express as px
from statsmodels.formula.api import ols
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
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
  st.header("Présentation du projet")
  
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
  st.header('Exploration')
  st.header('Visualisation')
  st.subheader('Distribution des variables')

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


# Analyses des corrélations et tests statistiques
  st.subheader("Analyse des corrélations avec tests statistiques des variables explicatives")

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

  st.subheader("Analyse de la corrélation des variables explicatives et de la variable cible")

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


  st.subheader("validation Statistique")
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
  model = xgboost.Booster()
  model.load_model("xgb_optimizedbst.model")
  dtest = xgboost.DMatrix(encoded_data)
  if st.button('Predictions'):
      prediction = model.predict(dtest)
    # XGBoost donne des probabilités pour la classification binaire, donc vous devez définir un seuil
    # Par exemple, si la prédiction est supérieure à 0.5, on considère que la classe prédite est 1
      predicted_class = (prediction > 0.5).astype(int)
      if predicted_class[0] == 1:
          st.info("La prédiction est : Yes")
      else:
          st.warning("La prédiction est : No")
        
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
