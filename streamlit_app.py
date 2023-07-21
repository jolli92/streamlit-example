import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
option = st.sidebar.selectbox(
    'Quel menu voulez-vous voir ?',
     ('Étude Statistiques', 'Menu 2', 'Menu 3'))
st.title('Analyse de bank marketing')
df_file = st.sidebar.file_uploader("Upload a Dataset", type=['csv', 'txt'])
if option == 'Étude Statistiques':
    df = pd.read_csv(df_file)
    st.dataframe(df.head())
    st.header('Distribution de la variable cible')        
    sns.countplot(x="deposit", data=df)
    st.pyplot(plt.gcf())
    plt.clf()

    deposit_counts = df['deposit'].value_counts()
    labels = deposit_counts.index
    sizes = deposit_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des dépôts')
    st.pyplot(plt.gcf())
    plt.clf()

    numeric_columns = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()
    sns.heatmap(correlation_matrix, annot=True)
    st.pyplot(plt)
    plt.clf()

    sns.histplot(x=df['age'],label='Age', kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

    df['duration_minutes'] = df['duration'] / 60
    sns.kdeplot(x=df['duration_minutes'],label='duration_minutes')
    st.pyplot(plt.gcf())
    plt.clf()

    sns.kdeplot(x=df['pdays'],label='pdays')
    st.pyplot(plt.gcf())
    plt.clf()

    sns.countplot(x="job", data=df)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    marital_counts = df['marital'].value_counts()
    labels = marital_counts.index
    sizes = marital_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des états matrimoniaux')
    st.pyplot(plt.gcf())
    plt.clf()

    sns.countplot(x="education", data=df)
    plt.title("Répartition des niveaux d'éducation")
    plt.xlabel("Niveau d'éducation")
    plt.ylabel("Décompte")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

    education_counts = df['education'].value_counts()
    labels = education_counts.index
    sizes = education_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des niveaux d\'éducation')
    st.pyplot(plt.gcf())
    plt.clf()

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

        
elif option == 'Menu 2':
    st.write('soon')

elif option == 'Menu 3':
    st.write('soon')
