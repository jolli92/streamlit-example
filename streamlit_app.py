import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


file_path = 'bank.csv'


df = pd.read_csv(file_path)

st.title('Analyse de bank marketing')
df = st.file_uploader("Upload a Dataset", type=['csv', 'txt'])
if df is not None:
    df = pd.read_csv(df)
    st.sidebar.dataframe(df.head())

    st.title('Etude statistique')

    st.header('Distribution de la variable cible')
   #distribution de la variable cible
   sns.countplot(x="deposit", data=df)
   st.pyplot(plt)
   #Souscription à un compte à terme
   deposit_counts = df['deposit'].value_counts()
   labels = deposit_counts.index
   sizes = deposit_counts.values

   plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
   plt.axis('equal')
   plt.title('Répartition des dépôts')
   st.pyplot(plt)
   #Exploration des relations entre les attributs numériques à l'aide d'une matrice de corrélation
   correlation_matrix = df.corr()
   sns.heatmap(correlation_matrix, annot=True)
   st.pyplot(plt)
   #Aucune corrélation linéaire : Si le coefficient est proche de 0, cela indique une absence de corrélation linéaire entre les variables.
   #Cela ne signifie pas nécessairement qu'il n'y a aucune relation entre les variables, mais plutôt qu'il n'y a pas de relation linéaire claire.
   #Distribution de l'age
   sns.histplot(x=df['age'],label='Age', kde=True);
   st.pyplot(plt)
   #Distribution de Duration
   df['duration_minutes'] = df['duration'] / 60
   sns.kdeplot(x=df['duration_minutes'],label='duration_minutes');
   st.pyplot(plt)
   #Distribution de pdays
   sns.kdeplot(x=df['pdays'],label='pdays');
   st.pyplot(plt)
   #Type de metier
   sns.countplot(x="job", data=df);
   plt.xticks(rotation=45);
   st.pyplot(plt)
   #Statuts marital
   marital_counts = df['marital'].value_counts()
   labels = marital_counts.index
   sizes = marital_counts.values

   plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
   plt.axis('equal')
   plt.title('Répartition des états matrimoniaux')
   st.pyplot(plt)
   #Niveau d'éducation des clients
   sns.countplot(x="education", data=df)
   plt.title("Répartition des niveaux d'éducation")
   plt.xlabel("Niveau d'éducation")
   plt.ylabel("Décompte")
   plt.xticks(rotation=45)
   st.pyplot(plt)

   education_counts = df['education'].value_counts()
   labels = education_counts.index
   sizes = education_counts.values

   plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
   plt.axis('equal')
   plt.title('Répartition des niveaux d\'éducation')
   st.pyplot(plt)
   variables = ["default", "housing", "loan"]

   # Parcours des variables
   for variable in variables:
       counts = df[variable].value_counts()
       labels = counts.index
       sizes = counts.values

       plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
       plt.axis('equal')
       plt.title(f"Répartition de la variable '{variable}'")
       st.pyplot(plt)
   #Répartition des types de contact
   contact_counts = df['contact'].value_counts()
   labels = contact_counts.index
   sizes = contact_counts.values

   plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
   plt.axis('equal')
   plt.title('Répartition des types de contact')
   st.pyplot(plt)
   #Décompte des contacts par mois
   month_counts = df['month'].value_counts().sort_index()
   months_ordered = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
   month_counts_ordered = month_counts.reindex(months_ordered)

   plt.bar(month_counts_ordered.index, month_counts_ordered.values)
   plt.title('Décompte des contacts par mois')
   plt.xlabel('Mois')
   plt.ylabel('Décompte')
   st.pyplot(plt)
   #Resultat de la campagne marketing précedente
   poutcome_counts = df['poutcome'].value_counts()
   labels = poutcome_counts.index
   counts = poutcome_counts.values

   plt.bar(labels, counts)
   plt.title('Décompte des résultats de la campagne précédente')
   plt.xlabel('Résultat de la campagne précédente')
   plt.ylabel('Décompte')
   plt.xticks(rotation=45)
   st.pyplot(plt)  

   poutcome_counts = df['poutcome'].value_counts()
   labels = poutcome_counts.index
   sizes = poutcome_counts.values

   plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
   plt.axis('equal')
   plt.title('Répartition des résultats de la campagne précédente')
   sst.pyplot(plt)
   #Balance par rapport au type de job
   job_balance_mean = df.groupby('job')['balance'].mean()
   job_balance_mean = job_balance_mean.sort_values(ascending=False)

   plt.bar(job_balance_mean.index, job_balance_mean.values)
   plt.title('Solde moyen par profession')
   plt.xlabel('Profession')
   plt.ylabel('Solde moyen')
   plt.xticks(rotation=45)
   st.pyplot(plt)
   #Distribution de l'age par rapport au type de job
   sns.boxplot(x="job", y="age", data=df)
   plt.title("Distribution de l'âge par type de job")
   plt.xlabel("Type de job")
   plt.ylabel("Âge")
   plt.xticks(rotation=45)
   st.pyplot(plt)
   #Distribution de l'age par rapport au statut marital
   sns.boxplot(x="marital", y="age", data=df)
   plt.title("Distribution de l'âge par état matrimonial")
   plt.xlabel("État matrimonial")
   plt.ylabel("Âge")
   st.pyplot(plt)
   #Distribution de l'age par rapport a l'éducation
   sns.boxplot(x="education", y="age", data=df)
   plt.title("Distribution de l'âge par niveau d'éducation")
   plt.xlabel("Niveau d'éducation")
   plt.ylabel("Âge")
   plt.xticks(rotation=45)
   st.pyplot(plt)

   #Distribution de l'age par rapport a loan et housing
   fig, axes = plt.subplots(1, 2, figsize=(12, 6))

   sns.boxplot(ax=axes[0], x="loan", y="age", data=df)
   axes[0].set_title("Distribution de l'âge selon les prêts personnels")
   axes[0].set_xlabel("Prêt personnel")
   axes[0].set_ylabel("Âge")
   st.pyplot(plt)
   sns.boxplot(ax=axes[1], x="housing", y="age", data=df)
   axes[1].set_title("Distribution de l'âge selon les prêts immobiliers")
   axes[1].set_xlabel("Prêt immobilier")
   axes[1].set_ylabel("Âge")
   st.pyplot(plt)


