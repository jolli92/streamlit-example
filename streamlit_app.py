import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


option = st.sidebar.selectbox(
    'Quel menu voulez-vous voir ?',
     ('Étude Statistiques', 'Menu 2', 'Menu 3'))
st.title('Analyse de bank marketing')
df_file = st.sidebar.file_uploader("Upload a Dataset", type=['csv', 'txt'])
if option == 'Étude Statistiques':
    df = pd.read_csv(df_file)
    st.header('Visualisation de la distribution de la variable cible : deposit')        
    fig = px.histogram(df, x="deposit", title="Distribution de deposit")
    st.plotly_chart(fig)
    plt.clf()


    deposit_counts = df['deposit'].value_counts()
    labels = deposit_counts.index
    sizes = deposit_counts.values
    fig2 = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des dépôts')
    plt.clf()
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig)
    col1.plotly_chart(fig2)
    
    st.header("Exploration des relations entre les attributs numériques à l'aide d'une matrice de corrélation")
    numeric_columns = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr()
    sns.heatmap(correlation_matrix, annot=True)
    st.pyplot(plt)
    plt.clf()
    st.write("Aucune corrélation linéaire : Si le coefficient est proche de 0, cela indique une absence de corrélation linéaire entre les variables.")
    st.write("Cependant cela ne signifie pas nécessairement qu'il n'y a aucune relation entre les variables, mais plutôt qu'il n'y a pas de relation linéaire claire.")
    
    st.header('Visualisation de la distribution de l"âge')
    fig2 = px.histogram(df, x="age", nbins=20, title="Distribution de l'âge",
                   labels={'age': 'Âge'}, marginal='box')
    st.plotly_chart(fig2)
    plt.clf()

    st.header('Visualisation de la durée de contact (appel téléphonique)')
    df['duration_minutes'] = df['duration'] / 60
    fig3 = px.histogram(df, x="duration_minutes", nbins=20, title="Distribution de la durée en minutes",
                   labels={'duration_minutes': 'Durée (minutes)'}, marginal='box')
    st.plotly_chart(fig3)

    st.header('Distribution du nombre de jours passés entre deux contacts de campagnes différentes')
    fig4 = go.Figure(data=go.Histogram(x=df['pdays'], nbinsx=20))
    fig4.update_layout(title="Distribution de pdays", xaxis_title="pdays")
    st.plotly_chart(fig4)
    plt.clf()

    st.header('Distribution des jobs')
    fig = go.Figure(data=go.Bar(x=df['job'].value_counts().index, y=df['job'].value_counts().values))
    fig.update_layout(title="Répartition des emplois", xaxis_title="Emploi", yaxis_title="Nombre de clients")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    st.write(df['job'].value_counts())
    st.write("*Les clients ayant des emplois de gestion et des emplois d'ouvrier qualifié sont les plus nombreux dans la banque.")
    st.write("*Il y a très peu d'étudiants parmi les clients de la banque.")

    st.header('Distribution des états matrimoniaux')
    marital_counts = df['marital'].value_counts()
    labels = marital_counts.index
    sizes = marital_counts.values
    fig = go.Figure(data=go.Pie(labels=labels, values=sizes, hoverinfo='label+percent',
                            textinfo='percent', insidetextorientation='radial'))
    fig.update_layout(title='Répartition des états matrimoniaux')
    st.plotly_chart(fig)
    st.write(marital_counts)

    st.header("Distribution du niveau d'étude")
    fig = go.Figure(data=go.Bar(x=df['education'].value_counts().index, y=df['education'].value_counts().values))
    fig.update_layout(title="Répartition des niveaux d'éducation", xaxis_title="Niveau d'éducation", yaxis_title="Décompte")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    st.write(df['education'].value_counts())

    education_counts = df['education'].value_counts()
    labels = education_counts.index
    sizes = education_counts.values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.header("Distribution des variables Default, housing et Loan")
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
    print('soon')

elif option == 'Menu 3':
    print('soon')
