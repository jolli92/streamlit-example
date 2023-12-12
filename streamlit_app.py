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
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('bank.csv')
st.title("Analyse de bank marketing")
st.sidebar.title("Sommaire")
pages=["DataVizualization","Pre-processing", "Prédictions", "Prédictions_2"]
page=st.sidebar.radio("Aller vers", pages)
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

if page == pages[0]:
    df = df
    st.title('EXPLORATION')
    st.title('VISUALISATION')
    st.header('Distribution des variables')
    
    # Widgets pour la sélection des variables et l'affichage des commentaires
    with st.container():
        selected_vars = st.multiselect('Sélectionnez les variables à visualiser:', df.columns)
        show_annotations = st.checkbox('Afficher les commentaires')

    if selected_vars:
        fig = create_visualisations(df, selected_vars)
        st.plotly_chart(fig)

        if show_annotations:
            st.header("Commentaire de la variable selectionnée")
            
            # Dictionnaire des commentaires pour chaque variable
            comments = {
    'job': "JOB est une variable qualitative ou catégorielle qui désigne le métier de chaque client. Il a 12 valeurs uniques à savoir : ['admin.' ,'technician' ,'services' ,'management' ,'retired' ,'blue-collar','unemployed' ,'entrepreneur' ,'housemaid' ,'unknown', 'self-employed','student']  , avec une majorité des clients qui travaillent dans le management soit un taux de 23%. dans la distribution suivi des ouvriers à 17,4%. et des techniciens à 16,3%. On observe aussi une valeur minime 'unknown' 0,6% qui peut signifier que certains clients préfèrent ne pas donner leur situation professionnelle.",
    'age': "Cette variable représente l'âge de chaque individu dans les données et est de type quantitatif. Il a 76 valeurs uniques avec un minimum de 18 ans et un maximum de 95 ans dont une moyenne de 41 ans avec 50% ayant plus ou moins de 39 ans. Suivant la visualisation, les clients sont majoritairement dans l’intervalle de 25 à 59 ans.",
    'marital': "Suivant la distribution de la variable “marital” qui est une variable qualitative, désigne le statut matrimonial de chaque client, il existe 3 valeurs uniques à savoir : 'married','divorced','single'. Avec 56,9% des clients mariés, 31,5% célibataire et 11,6% divorcé.",
    'education': "La variable “education” représente le niveau d'étude de chaque client. Etant une variable qualitative elle a comme valeur unique : 'unknown','secondary','primary','tertiary' avec une majorité de client qui sont aux secondaires soit 49,1%, suivi des clients ayant fait des études universitaires et ceux du primaire soit 33% et 13,4% et on termine avec une valeur « unknow » qui peut désigner des clients qui n’ont pas souhaité renseigner leur niveau d’étude ou même ceux qui n’en ont pas fait d’étude.",
    'balance': "La variable balance désigne le solde bancaire de chaque client prospecté ce qui signifie que c’est une variable quantitative avec 3 805 valeurs unique dont une moyenne de 1 528,538524 euros pour un minimum de -6 847 euros et un maximum de 81 204 euros. On observe aussi qu’il y a plus et moins 50% de client qui ont un solde de 550 avec la majorité des clients ayant un solde bancaire compris entre 122 et 1 708 euros d’où l’importance de faire attention aux valeur extrême de cette variable qui a un maximum plus élevé que la médiane.",
    'default': "La variable ‘default’ désigne le risque de solvabilité d’un client il permet de savoir si un client est en défaut de paiement ou pas. Etant une variable catégorielle de type booléen, on peut clairement deviner qu’elle n’a que 2 valeurs unique (Yes et No) on se rend compte que la majorité des clients soit 98,5% n’est pas en défaut de paiement.",
    'housing': "La variable ‘housing’ représente les clients qui ont un crédit immobilier ou non, c’est donc une variable qualitative de type booléen avec deux valeurs unique (Yes et No). On constate suivant le graphique ci-dessus que 52,7% des clients n’ont pas de crédit immobilier et 47,3% en ont.",
    'loan': "La variable ‘loan’ représente l’ensemble de client endetté. C’est une variable catégorielle de type booléen à deux valeurs uniques (Yes et No). Le graphique nous renseigne que 86,9% des clients n’ont pas de dette et 13,1% en ont.",
    'contact': "La variable ‘contact’ désigne la façon dont les clients ont été contacté pendant la campagne, il en ressort que 72% des clients ont été contacté par téléphone et on a 21% de client dont on ne sait comment ils ont été contactés, on peut supposer par mail ou en présentiel ou tout simplement inconnu. C’est une variable catégorielle avec 3 valeurs uniques (cellular, unknown et téléphone)",
    'day': "La variable « day » désigne le jour où le client a été contacté pour la dernière fois. On se rend compte que la moyenne et la médiane de cette variable est casi similaire à 15 ce qui signifie que la variable est bien répartie, avec plusieurs jours ou on a beaucoup appelé les clients. Le premier jour étant de 1 et le dernier jour le 31 du mois avec des faibles taux d’appel le 1, 10, 24 et 31. C’est une variable quantitative ave 31 valeurs uniques.",
    'month': "La variable ‘month’ correspond au dernier mois ou on a contacté le client pendant une, c’est une variable catégorielle à 12 valeurs uniques ( 'jan', 'feb', 'mar', ..., 'nov', 'dec'). Suivant le graphique on a contacté les clients beaucoup plus en mai avec 25,3% ensuite juillet, aout et juin, le reste de valeur est relativement en dessous de 10% ",
    'duration': "La variable ‘duration’ représente le temps d'appel de la dernière fois que le client a été contacté en seconds, c’est une variable quantitative avec une moyenne d’appel de 371,99 secondes, on observe des temps d’appel de plus d’une heure de temps qu’il faut quand même bien analyser pour la suite pour savoir comment les traiter.",
    'campaign': "La variable ‘campaign’ représente le nombre de fois que le client a été contacté lors de la campagne, c’est une variable quantitative avec 36 valeurs uniques. On constate selon le graphique que la majorité des clients ont été contactés une seule fois avec une variation allant jusqu’à 11 fois. Toute fois il faut qu’on prête attention au client contacté 63 fois pour voir comment le traiter.",
    'pdays': "La variable ‘pdays’ représente le nombre de jours qui se sont écoulés depuis qu'un client a été contacté lors de la campagne précédente sachant que -1 est une valeur qui signifie que le client n’a pas été contacté lors de la campagne précédente. C’est une variable quantitative de 472 valeurs uniques avec un temps écoulé moyen de 51.330407 jours mais une médiane de -1 ce qui signifie que 50% des clients n’avaient pas été contacté précédemment et 50% avaient déjà été contacté.",
    'previous': "La variable ‘previous’ représente le nombre de fois qu’un client a été contacté lors de la campagne précédente. C’est une variable quantitative de 34 valeurs uniques avec plus de 8000 clients qui n’ont pas été contacté lors de la précédente campagne. Le nombre moyen de fois est quasi nul avec une médiane et un premier quartil de 0 ce qui signifie que 50% des clients n’avaient pas été contacté lors de la précédente campagne. Le nombre de fois où les clients ont été contactés sur la campagne précédente a rarement dépassé trois fois, ce qui peut être compréhensible puisqu’un client trop sollicité pendant une campagne aura tendance à se désintéresser des dépôts à terme.",
    'poutcome': "La variable ‘poutcome’ est le resultat de la campagne marketing précédente c’est une variable catégorielle de 4 valeurs uniques ('unknown','other'','failure','success'). On observe qu’il y’a plus de 8000 clients soit 74,6% de clients qui sont dans la catégorie « unknow » mais ceci peut s’expliquer vu le nombre de client qui n’avaient jamais été contacté dans la variable « prévious » ainsi qu’au nombre de fois que le client avait été contacté suivant la variable « pdays ».",
    'deposit' : "La variable « deposit » est notre variable cible, elle indique si oui ou non un client a souscrit à un dépôt à terme c’est une variable catégorielle de type booléen avec 52,6% de client ayant refusé de souscrire et 47,4% ayant souscrit."
}

# Affichage des commentaires avec st.info
            for var in selected_vars:
                if var in comments:
                    st.info(f"{var}: {comments[var]}")
                else:
                    st.info(f"Aucun commentaire disponible pour {var}.")
    else:
        st.write("Veuillez sélectionner des variables pour afficher les graphiques et les commentaires associés.")


# Analyses des corrélations et tests statistiques

st.header("Analyse des corrélations avec tests statistiques des variables explicatives")

# Widget pour choisir les heatmaps à afficher
heatmap_choices = st.multiselect("Choisissez les heatmaps à afficher:", 
                                 ["Corr Numérique", "Corr Catégorielle", "Corr Num-Cat"])

# Boucle sur les choix de l'utilisateur et affichage des heatmaps correspondantes
for choice in heatmap_choices:
    if choice == "Corr Numérique":
        # Affichage de la heatmap numérique
        st.header("Analyse de la corrélation entre les variables numériques")
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
            st.header("Analyse de la corrélation entre les variables catégorielles et les variables numériques")
            import numpy as np
            import itertools
            import plotly.express as px
            from statsmodels.formula.api import ols
            import statsmodels.api as sm


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
Nous avons décidé de commun accord le maintien de la variable 'day' dans notre analyse.
""")
 
#visualisation des corrélations avec la variable cible déposit



    # Commentaires pour chaque corrélation
    correlation_comments = {
    'previous':"La majorité des clients n'avaient pas été contactés avant cette campagne. Cependant,  un  taux  de  souscription  plus  élevé  est  observé  chez  ceux  ayant  été  contactés  plusieurs  fois auparavant, suggérant que les efforts de marketing répétés peuvent construire une base de clients  fidèles et réceptifs. ",
    'pdays': "Un grand nombre de clients ont été contactés après une longue période (999 jours indiquant probablement une absence  de  contact  antérieur).  Les  clients  contactés  plus  récemment  sont  plus  susceptibles  de  souscrire, soulignant l'importance de maintenir une communication régulière. ",
    'default':"Les graphiques ci-dessus démontrent tout d’abord que la corrélation entre la variable default et la variable cible est en dessous de 0.5 mais existante. Les personnes en défaut de paiement sont moins intéressées par les dépôts à terme par rapport à ceux qui ne le sont pas du fait des difficultés que peuvent présenter leurs trésoreries",
    'campaign': "La plupart des souscriptions se produisent lorsque les clients sont contactés entre une et trois fois. Au-delà,  la  probabilité  de  souscription  diminue,  ce  qui  indique  un  point  de  saturation  dans  les  efforts  de communication.",
    'duration':"La durée de l'appel semble être un indicateur fort de la souscription, avec des appels plus longs qui impliquent une plus grande probabilité de souscription.",
    'day':"La distribution de la souscription est relativement uniforme à travers le mois, bien qu'il y  ait des variations mineures qui méritent une analyse plus approfondie pour optimiser le timing des contacts.",
    'poutcome':"Les  clients  ayant  eu  un  résultat  positif  (« success »)  lors  de  la  campagne  précédente  sont  nettement  plus susceptibles de souscrire à nouveau, soulignant l'importance de construire une relation positive continue avec les clients. ",
    'month':"Bien que le mois de Mai soit le mois le plus actif en termes de contacts, les mois de Mars, Décembre, Octobre et  Septembre  se  distinguent  par  une  réussite  de  souscription  plus  élevée,  suggérant  une  saisonnalité  dans l'efficacité de nos campagnes",
    'loan':"De même, les clients sans prêt personnel montrent une propension plus élevée à la souscription, renforçant l'idée qu'une moindre charge de dettes favorise l'engagement envers de nouveaux services financiers. ",
    'housing':"Il apparaît que les clients sans prêt immobilier sont plus enclins à souscrire, ce qui peut refléter une plus grande flexibilité financière ou une aversion moindre au risque. ",
    'education':"Nous constatons que les clients avec un niveau d'éducation tertiaire ont un taux de souscription plus élevé par rapport aux autres niveaux d'éducation. Cela indique que le niveau d'éducation peut influencer la propension à souscrire.",
    'marital':"Les célibataires affichent un taux de souscription légèrement supérieur comparé aux autres statuts maritaux, ce qui suggère que le célibat peut être un indicateur positif pour la souscription à nos services.",
    'balance': "Ce graphique nous indique que la majeure partie des clients qui souscrivent au dépôt à terme ont des soldes bancaires qui varient entre 0 et 10k euros.",
    'age' : "En analysant ce graphique, il est évident que les distributions d'âge pour les souscriptions au dépôt ('oui' et 'non') sont remarquablement proches. L'alignement étroit des deux distributions suggère que l'âge seul pourrait ne pas être un déterminant important pour prédire si un client souscrira au dépôt à terme",
    'job': "Le graphique présente une fréquence élevée de souscription pour les étudiants, les managers et les ouvriers, cela suppose que les personnes de ces corps de métiers ont plus de chance de souscrire à un dépôt à terme que les  autres  catégories.  Tout  en  notant  que  dans  la  majorité  des  métiers  il  y  a  une  fréquence  de  souscription plutôt bonne.",
  }

    # Variables explicatives à sélectionner pour la visualisation
    variables_to_choose = ['marital', 'education', 'default', 'housing', 'loan', 'month','previous', 
                       'poutcome', 'day', 'age', 'job', 'balance', 'contact', 'duration','campaign','pdays']

    st.header("Analyse de la corrélation des variables explicatives et de la variable cible")

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

    
    st.header("validation Statistique")

    # Checkbox pour la première partie (Test statistique du Chi Carré)
    if st.checkbox("Test statistique du Chi Carré"):
        st.markdown("""
    Afin de vérifier statistiquement l'influence des variables catégorielles sur la variable cible, nous avons utilisé le test statistique du chi carré qui permet de montrer s'il existe 
ou non une relation entre deux variables catégorielles. 
Nous  constatons  que  toutes  les  statistiques  de  test  des  variables  catégorielles  respectives  sont  toutes 
significativement  inférieures  à  5%.  Ce  qui  nous  permet  de  rejeter  l'hypothèse  nulle  d'indépendance  des 
variables catégorielles par rapport à la variable cible (deposit) et cela sous entend que toutes ces variables ont une influence sur la décision du client à souscrire ou pas au dépot à terme    
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
    if st.checkbox("Test de Student"):
        st.markdown("""
    
    Le test de Student est un test statistique utilisé pour confirmer la dépendance pertinente observée entre des variables numériques et une variable catégorielle.
    Nous constatons que toutes les statistiques de test (t-student) des variables numériques respectives sont toutes < 5%.
    Nous pouvons donc affirmer avec certitude que les caractéristiques numériques que nous avons étudiées sont liées à la décision du client de souscrire ou non au dépôt à terme.
    
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


    st.markdown("""
En bref, malgré les informations substantielles fournies par l'analyse exploratoire des variables, il est crucial de noter que la relation statistique ne garantit pas la causalité. Une investigation plus approfondie, telle qu'une modélisation prédictive, serait nécessaire pour comprendre comment ces variables influent réellement sur la souscription aux dépôts à terme.
Nous allons donc procéder à la modélisation de notre jeu de données pour faire de bonnes prédictions, en commençant par le Pre-processing.
""")

    

if page == pages[2] :
    st.write("Pre-processing")
    df = pd.read_csv('df_preprocessed.csv')
     
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
    X_train_encoded.info()
    buffer = io.StringIO()
    X_train_encoded.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
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
    X_train_normalised.info()

    model_choisi = st.selectbox(label = "Modèle", options = ['Regression Logistique', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost'])
        
    if model_choisi == 'Regression Logistique' :        
        LR = joblib.load('LogisticRegression')
        y_pred = LR.predict(X_test_normalised)
        st.text(classification_report(y_test, y_pred))
        train_sizes, train_scores, test_scores = learning_curve(LR, X_train_normalised, y_train, n_jobs=-1, 
                                                        train_sizes=np.linspace(.1, 1.0, 5))
    if model_choisi == 'KNN' :        
        knn = joblib.load("knn_ma")
        y_pred = knn.predict(X_test_normalised)
        st.text(classification_report(y_test, y_pred))
        train_sizes, train_scores, test_scores = learning_curve(knn, X_train_normalised, y_train, n_jobs=-1, 
                                                        train_sizes=np.linspace(.1, 1.0, 5))
    if model_choisi == 'Decision Tree' :       
       with open('clf_dt_gini.dill', 'rb') as f:
           clf_dt_ginis = dill.load(f)
       y_pred = clf_dt_ginis.predict(X_test_encoded)
       #st.text(classification_report(y_test, y_pred1))
       st.text(classification_report(y_test, y_pred))
       train_sizes, train_scores, test_scores = learning_curve(clf_dt_ginis, X_train_encoded, y_train, n_jobs=-1, 
                                                            train_sizes=np.linspace(.1, 1.0, 5))
    if model_choisi == 'Random Forest' :
        with open('random_forest_model.dill', 'rb') as f:
            clf_optimizedd = dill.load(f)
        y_pred = clf_optimizedd.predict(X_test_encoded)
        st.text(classification_report(y_test, y_pred))
        train_sizes, train_scores, test_scores = learning_curve(clf_optimizedd, X_train_encoded, y_train, n_jobs=-1, 
                                                            train_sizes=np.linspace(.1, 1.0, 5))
    if model_choisi == 'XGBoost' :
       #XGBoost = load('xgb_optimized')
       #XGBoost.fit(X_train_encoded, y_train)
       XGBoost = joblib.load("xgb_optimized")       
       X_test_encoded = xgboost.DMatrix(X_test_encoded)
       loaded_bst = xgboost.Booster()
       loaded_bst.load_model('xgb_optimizedbst.model')
       y_pred = loaded_bst.predict(X_test_encoded)
       #st.text(classification_report(y_test, y_pred))
       y_pred_labels = (y_pred > 0.5).astype(int)
       report = classification_report(y_test, y_pred_labels)
       st.text(report)
       train_sizes, train_scores, test_scores = learning_curve(XGBoost, X_train_encoded, y_train, n_jobs=-1, 
                                                        train_sizes=np.linspace(.1, 1.0, 5))

# Calcul des moyennes et des écarts-types des scores de formation et de test
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

# Création du graphique
    plt.figure()
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title('Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()

# Affichage du graphique dans Streamlit
    st.pyplot()

if page == pages[3] :
    st.write("Prédictions")
    df = pd.read_csv('bank.csv')
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
    st.write("Ce script démontre l'utilisation standard de Streamlit et XGBoost pour développer une application web interactive axée sur les prédictions, en utilisant des données fournies par l'utilisateur.")
        

    if st.checkbox('Informations complémentaires'):
        st.write("""
        Choix des Caractéristiques par l'Utilisateur :

        Le script emploie la fonction st.selectbox de Streamlit pour générer des menus déroulants. Ces menus permettent aux utilisateurs de sélectionner des options pour divers attributs tels que le métier, le mois, l'éducation, etc., à partir d'un DataFrame nommé df.
        Pour chaque attribut sélectionné, le script crée une colonne correspondante dans un autre DataFrame encoded_data. La catégorie choisie reçoit la valeur 1, tandis que toutes les autres catégories reçoivent la valeur 0.

        Préparation des Données Complémentaires :

        Le script assigne automatiquement des valeurs par défaut à certaines colonnes, telles que age_group et balance_group, basées sur des catégories préétablies.
        Il utilise également des statistiques descriptives telles que la médiane et la moyenne du DataFrame df pour compléter d'autres colonnes, notamment day, duration, pdays, campaign, et previous.

        Finalisation de la Préparation des Données :

        Avant la prédiction, le DataFrame encoded_data est réorganisé pour correspondre à la structure requise par le modèle prédictif.

        Processus de Prédiction :

        Lorsque l'utilisateur clique sur le bouton "Prédictions", le modèle génère une prédiction basée sur les données entrées.
        Le modèle XGBoost, utilisé pour la prédiction, fournit des probabilités pour une classification binaire. Un seuil spécifique, comme 0.5, est appliqué pour déterminer la classe prédite.
    """)

    
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
    st.write("Prédictions_2")
    st.write("Ajout d'une colonne prénom + téléphone(généré aleatoirement) /colonne déposit supprimée et redistribution compléte du dataset sur toutes les colonnes à l'aide de .sample")
    df = pd.read_csv('Banktest.csv')
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
    #with open('xgb_optimizedpickle', 'rb') as model_file:
        #model = pickle.load(model_file)
    #y_pred = model.predict(encoded_df)
    #df['prediction'] = y_pred
    
    # Concatenate the columns "prénom" and "téléphone" and sort by prediction
    #df = pd.concat([df_prenom_telephone, df], axis=1)
    
    #df_sorted = df.sort_values(by='prediction', ascending=False)
    #y_proba = model.predict_proba(encoded_df)
    #df['probability'] = y_proba[:,1]  # Pour une classification binaire, cela donnerait la probabilité de la classe 1
    #df_sorted = df.sort_values(by='probability', ascending=False)
    #df_sorted = df_sorted[['prénom', 'téléphone','probability']]
# Display the top 50 clients
    #st.dataframe(df_sorted.head(50))
 # Charger le modèle XGBoost
    model = xgboost.Booster()
    model.load_model("xgb_optimizedbst.model")

# Préparation des données pour la prédiction
    dtest = xgboost.DMatrix(encoded_df)

# Effectuer les prédictions
    y_pred = model.predict(dtest)
    df['prediction'] = y_pred

# Concaténation des colonnes "prénom" et "téléphone" et tri par probabilité
    df = pd.concat([df_prenom_telephone, df], axis=1)

# Calcul des probabilités (si votre modèle est un classificateur binaire)
# Note: XGBoost retourne directement les probabilités, donc pas besoin de 'predict_proba' comme dans scikit-learn
    df['probability'] = y_pred  # Si le modèle est binaire, cela donne directement la probabilité de la classe positive

# Trier le DataFrame par probabilité
    df_sorted = df.sort_values(by='probability', ascending=False)

# Garder uniquement les colonnes pertinentes
    df_sorted = df_sorted[['prénom', 'téléphone', 'probability']]

# Afficher les 50 premiers clients
    st.dataframe(df_sorted.head(50))

