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


df = pd.read_csv('bank.csv')
st.title("Analyse de bank marketing")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization","Pre-processing", "Prédictions", "Prédictions_2"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] :
    st.write("Exploration")
    st.header("Informations du DataFrame :")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("premier aperçu de la table, nous avons 11 162 enregistrements non null sur 17 variables, dont 7 variables sont de type integer et 10 de type objets.")
    st.write("des structures des données affichent que toutes les lignes sont remplies, tandis que les premières lignes de données renvoient des valeurs 'Unknown', l'impact de cette valeur à est un point d'attention à voir plus loin..")
    st.header("Affichage des valeurs uniques prisent par les variables")
    categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
    for column in categorical_columns:
        unique_values = df[column].unique()
        st.markdown(f"Valeurs uniques de la colonne '{column}': {unique_values}\n")
    st.header("Description statistique du DataFrame :")
    st.dataframe(df.describe())
    st.dataframe(df.describe(include=["object"]))
    st.write(" * Informations client :")
    st.write("L'aperçu des données numériques montre que l'échantillon de la population varie de 18 à 95 ans, avec un âge moyen de 41 ans et une grande majorité vers 49 ans.")
    st.write("Le solde de leur compte courant varie entre un déficit de -6 847 et un crédit de 81 204 euros, en moyenne les clients ont 1 528 euros sur leurs comptes.")
    st.write(" * Informations campagne étudiée :")
    st.write("Les clients sont contactés durant la campagne en moyenne 2 à 3 fois. La majorité des réponses sont obtenues à partir de la troisième prise de contact.")
    st.write("Un point d'attention est mis sur la valeur 63 affectée à la prise de contact durant la campagne... Peut-on déjà en déduire que c'est une valeur aberrante ? Pourtant, pas impossible...")
    st.write("Nous constatons aussi que le dernier contact avec le client se localise vers la fin du mois, le 22, avec une durée comprise entre 2 secondes et 1 heure.")
    st.write("La majorité des réponses sont obtenues entre 6 et 8 minutes. Peut-on dire que les clients qui nécessitent de rester 1 heure en ligne sont plus difficiles à convaincre ?")
    st.write(" * Informations campagne précédente :")
    st.write("On constate que 50% de l'échantillon n'a jamais été contacté (previous=0) avant la campagne, ce qui est cohérent avec le nombre de jours séparant le dernier contact (pdays = -1).")
    st.write("Peut-on dire que l'échantillon contient essentiellement de nouveaux contacts clients ?")
    st.write("De plus, 75% de ceux qui ont été contactés l'ont été au bout de 21 jours et n'ont eu en général qu'un seul contact.")
    st.write("Pour ceux qui ont été régulièrement contactés (previous=58), ils ne l'ont été qu'au bout de plus de deux ans... Ce qui attire notre attention sur la cohérence entre le nombre de fois que le client a été contacté (58) et la durée du dernier contact, plus de deux ans... Peut-on déduire que c'est un client en portefeuille depuis trop longtemps ?")
    st.write("Ou au contraire, le client n'a été très sollicité puisqu'il l'a été qu'au bout de plus de deux ans... Mais dans ce cas, le nombre de contacts, 58 fois avant la campagne, nous pose quelques questions... 58 fois")
    st.header("Description des variables")
    st.write("age (quantitative)")
    st.write("job: type de job (categorielle: \"admin.\",\"unknown\",\"unemployed\",\"management\",\"housemaid\",\"entrepreneur\",\"student\",\"blue-collar\",\"self-employed\",\"retired\",\"technician\",\"services\")")
    st.write("marital : Statut marital (categorielle: \"married\",\"divorced\",\"single\"; note: \"divorced\" meansdivorced or widowed)")
    st.write("education : (categorielle: \"unknown\",\"secondary\",\"primary\",\"tertiary\")")
    st.write("default : Le client a-t-il des crédits en défaut ? (binaire: \"yes\",\"no\")")
    st.write("balance : Solde annuel et moyen des clients, en euros (quantitative)")
    st.write("housing :Le client a-t-il un crédit immobilier ? (binaire: \"yes\",\"no\")")
    st.write("loan : Le client a-t-il des crédits personnels ? (binaire: \"yes\",\"no\")")
    st.write("contact : Type de moyen de communication utilisé pour contacter (categorielle: \"unknown\",\"telephone\",\"cellular\")")
    st.write("day : Dernier jour de contact du mois (quantitative)")
    st.write("month : Dernier mois de contact de l'année (categorielle: \"jan\", \"feb\", \"mar\", ..., \"nov\", \"dec\")")
    st.write("duration : Temps d'appel du dernier contact effectué, en secondes (quantitative)")
    st.write("campaign : Nombre de contacts effectués durant cette campagne et pour ce client (quantitative, includes last contact)")
    st.write("pdays : Nombre de jours qui se sont écoulés depuis qu'un client a été lors de la campagne précédente (quantitative, -1 signifie que le client n'a jamais été contacté)")
    st.write("previous : Nombre de contacts effectués lors de la campagne précédente et pour ce client (quantitative)")
    st.write("poutcome : Résultat de la campagne marketing précédente (categorielle: \"unknown\",\"other\",\"failure\",\"success\")")

if page == pages[1] :
    st.write("DataVizualization")
    st.header('_Visualisation de la distribution de la variable cible : deposit_')        

    fig1 = px.histogram(df, x="deposit")
    deposit_counts = df['deposit'].value_counts()
    labels = deposit_counts.index
    sizes = deposit_counts.values
    fig2 = go.Figure(data=go.Pie(labels=labels, values=sizes, textinfo='percent+label', insidetextorientation='radial'))
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    st.write('47.4% des clients de la banque ont souscrit un compte à terme')
    st.write("52.6% des clients de la banque n'ont pas souscrit un compte à terme")


    st.header("Analyse de toutes les variables spécifiques aux clients")
    fig1 = px.histogram(df, x="age", nbins=20, title="Visualisation de la distribution de l'âge",
               labels={'age': 'Âge'}, marginal='box')

    df['duration_minutes'] = df['duration'] / 60
    fig2 = px.histogram(df, x="duration_minutes", nbins=20, title="Visualisation de la durée de contact (appel tel)",
               labels={'duration_minutes': 'Durée (minutes)'}, marginal='box')
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    
    fig4 = go.Figure(data=go.Histogram(x=df['pdays'], nbinsx=20))
    fig4.update_layout(title="Distribution de pdays", xaxis_title="pdays")

    fig = go.Figure(data=go.Bar(x=df['job'].value_counts().index, y=df['job'].value_counts().values))
    fig.update_layout(title="'Distribution des jobs", xaxis_title="Emploi", yaxis_title="Nombre de clients")
    fig.update_xaxes(tickangle=45)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig4, use_container_width=True)
    col2.plotly_chart(fig, use_container_width=True)

    marital_counts = df['marital'].value_counts()
    labels = marital_counts.index
    sizes = marital_counts.values
    fig1 = go.Figure(data=go.Pie(labels=labels, values=sizes, hoverinfo='label+percent',
                        textinfo='percent', insidetextorientation='radial'))
    fig1.update_layout(title='Distribution des états matrimoniaux')

    
    fig2 = go.Figure(data=go.Bar(x=df['education'].value_counts().index, y=df['education'].value_counts().values))
    fig2.update_layout(title="Distribution du niveau d'étude", xaxis_title="Niveau d'éducation", yaxis_title="Décompte")
    fig2.update_xaxes(tickangle=45)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    st.write("Les clients ayant des emplois de gestion et des emplois d'ouvrier qualifié sont les plus nombreux dans la banque.")
    st.write("Il y a très peu d'étudiants parmi les clients de la banque.")
    st.write("Les métiers les plus représentés chez les clients de la banque sont le management et les blue-collar.")
    st.write("La majorité des clients de la banque sont mariés (56.9%) ou célibataires (31.5%).")
    st.write("La majorité des clients de la banque ont suivi un cursus de second cycle (49.1%) ou de troisième cycle (33%).")
    st.write("Seulement 11% des clients ont suivi un cursus de premier cycle.")


    variables = ["default", "housing", "loan"]
    for variable in variables:
        counts = df[variable].value_counts()
        labels = counts.index
        sizes = counts.values
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f"Répartition de la variable '{variable}'")
        st.pyplot(plt.gcf())
        plt.clf()

    contact_counts = df['contact'].value_counts()
    labels = contact_counts.index
    sizes = contact_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des types de contact')
    st.pyplot(plt.gcf())
    plt.clf()

    month_counts = df['month'].value_counts().sort_index()
    months_ordered = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_counts_ordered = month_counts.reindex(months_ordered)
    plt.bar(month_counts_ordered.index, month_counts_ordered.values)
    plt.title('Décompte des contacts par mois')
    plt.xlabel('Mois')
    plt.ylabel('Décompte')
    st.pyplot(plt.gcf())
    plt.clf()

    poutcome_counts = df['poutcome'].value_counts()
    labels = poutcome_counts.index
    counts = poutcome_counts.values
    plt.bar(labels, counts)
    plt.title('Décompte des résultats de la campagne précédente')
    plt.xlabel('Résultat de la campagne précédente')
    plt.ylabel('Décompte')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    poutcome_counts = df['poutcome'].value_counts()
    labels = poutcome_counts.index
    sizes = poutcome_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des résultats de la campagne précédente')
    st.pyplot(plt.gcf())
    plt.clf()

    job_balance_mean = df.groupby('job')['balance'].mean()
    job_balance_mean = job_balance_mean.sort_values(ascending=False)
    plt.bar(job_balance_mean.index, job_balance_mean.values)
    plt.title('Solde moyen par profession')
    plt.xlabel('Profession')
    plt.ylabel('Solde moyen')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    sns.boxplot(x="job", y="age", data=df)
    plt.title("Distribution de l'âge par type de job")
    plt.xlabel("Type de job")
    plt.ylabel("Âge")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    sns.boxplot(x="marital", y="age", data=df)
    plt.title("Distribution de l'âge par état matrimonial")
    plt.xlabel("État matrimonial")
    plt.ylabel("Âge")
    st.pyplot(plt.gcf())
    plt.clf()

    sns.boxplot(x="education", y="age", data=df)
    plt.title("Distribution de l'âge par niveau d'éducation")
    plt.xlabel("Niveau d'éducation")
    plt.ylabel("Âge")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(ax=ax, x="loan", y="age", data=df)
    ax.set_title("Distribution de l'âge selon les prêts personnels")
    ax.set_xlabel("Prêt personnel")
    ax.set_ylabel("Âge")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(ax=ax, x="housing", y="age", data=df)
    ax.set_title("Distribution de l'âge selon les prêts immobiliers")
    ax.set_xlabel("Prêt immobilier")
    ax.set_ylabel("Âge")
    st.pyplot(fig)

if page == pages[2] :
    st.write("Pre-processing")
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


#Transformation des colonnes age et balance pour creer un découpage dans le but d'attenuer les valeurs extrémes qui ne me semble pas abberante tout en les gardant.
#Création du bins et des étiquettes
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
    X_train = X_train_encoded.drop(columns=['balance', 'age'])
    X_test = X_test_encoded.drop(columns=['balance', 'age'])
    buffer = io.StringIO()
    X_train.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

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
    with open('xgb_optimized.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
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








    if st.button('Predictions'):
        prediction = model.predict(encoded_data)
        if prediction == 1:
        # Affichage du résultat sous forme d'un pop-up
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
    with open('xgb_optimized.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    y_pred = model.predict(encoded_df)
    df['prediction'] = y_pred
    if 'pdays' not in df:
        st.write("1La colonne 'pdays' a disparu!")
    # Concatenate the columns "prénom" and "téléphone" and sort by prediction
    df = pd.concat([df_prenom_telephone, df], axis=1)
    if 'pdays' not in df:
        st.write("0La colonne 'pdays' a disparu!")
    df_sorted = df.sort_values(by='prediction', ascending=False)
    df_sorted = df_sorted[['prénom', 'téléphone']  
    # Display the top 50 clients
    
    st.dataframe(df_sorted.head(50))
