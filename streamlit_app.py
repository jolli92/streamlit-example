import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import io

st.title('Analyse de bank marketing')
df_file = st.sidebar.file_uploader("Upload a Dataset", type=['csv', 'txt'])
df = pd.read_csv(df_file)
option = st.sidebar.selectbox('Quel menu voulez-vous voir ?', ('Etude statistiques', 'Menu X', 'Menu XX'))

if option == 'Etude statistiques':
    option = st.sidebar.selectbox('Quel menu voulez-vous voir ?', ('Analyse des informations brutes', 'Etude des variables', 'Menu 3'))

    if option == 'Analyse des informations brutes':
        st.header("Informations du DataFrame :")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.header("Description statistique du DataFrame :")
        st.dataframe(df.describe())
        st.dataframe(df.describe(include=["object"]))
        st.header("Description des variables")
        st.markdown("<u>age</u> (quantitative)")
        st.write("<u>job:</u> type de job (categorielle: \"admin.\",\"unknown\",\"unemployed\",\"management\",\"housemaid\",\"entrepreneur\",\"student\",\"blue-collar\",\"self-employed\",\"retired\",\"technician\",\"services\")")
        st.write("<u>marital</u> : Statut marital (categorielle: \"married\",\"divorced\",\"single\"; note: \"divorced\" meansdivorced or widowed)")
        st.write("<u>education</u> (categorielle: \"unknown\",\"secondary\",\"primary\",\"tertiary\")")
        st.write("<u>default:</u> Le client a-t-il des crédits en défaut ? (binaire: \"yes\",\"no\")")
        st.write("<u>balance:</u> Solde annuel et moyen des clients, en euros (quantitative)")
        st.write("<u>housing:</u> Le client a-t-il un crédit immobilier ? (binaire: \"yes\",\"no\")")
        st.write("<u>loan:</u> Le client a-t-il des crédits personnels ? (binaire: \"yes\",\"no\")")
        st.write("<u>contact:</u> Type de moyen de communication utilisé pour contacter (categorielle: \"unknown\",\"telephone\",\"cellular\")")
        st.write("<u>day:</u> Dernier jour de contact du mois (quantitative)")
        st.write("<u>month:</u> Dernier mois de contact de l'année (categorielle: \"jan\", \"feb\", \"mar\", ..., \"nov\", \"dec\")")
        st.write("<u>duration:</u> Temps d'appel du dernier contact effectué, en secondes (quantitative)")
        st.write("<u>campaign:</u> Nombre de contacts effectués durant cette campagne et pour ce client (quantitative, includes last contact)")
        st.write("<u>pdays:</u> Nombre de jours qui se sont écoulés depuis qu'un client a été lors de la campagne précédente (quantitative, -1 signifie que le client n'a jamais été contacté)")
        st.write("<u>previous:</u> Nombre de contacts effectués lors de la campagne précédente et pour ce client (quantitative)")
        st.write("<u>poutcome:</u> Résultat de la campagne marketing précédente (categorielle: \"unknown\",\"other\",\"failure\",\"success\")")

    elif option == 'Etude des variables':
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

                
        
        st.write("Les clients ayant des emplois de gestion et des emplois d'ouvrier qualifié sont les plus nombreux dans la banque.")
        st.write("Il y a très peu d'étudiants parmi les clients de la banque.")
        st.write("Management et blue colar sont les métiers les plus représentés chez les clients de la banque.")
        st.write("La majorité des clients de la banques sont mariés (56.9%) ou célibataire (31.5%).")

        marital_counts = df['marital'].value_counts()
        labels = marital_counts.index
        sizes = marital_counts.values
        fig1 = go.Figure(data=go.Pie(labels=labels, values=sizes, hoverinfo='label+percent',
                            textinfo='percent', insidetextorientation='radial'))
        fig1.update_layout(title='Distribution des états matrimoniaux')

        st.header("Distribution du niveau d'étude")
        fig2 = go.Figure(data=go.Bar(x=df['education'].value_counts().index, y=df['education'].value_counts().values))
        fig2.update_layout(title="Distribution du niveau d'étude", xaxis_title="Niveau d'éducation", yaxis_title="Décompte")
        fig2.update_xaxes(tickangle=45)
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        # ...

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

    elif option == 'Menu 3':
        print('soon')

elif option == 'Menu X':
    print('soon')

elif option == 'Menu XX':
    print('soon')
