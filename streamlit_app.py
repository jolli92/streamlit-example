import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.title('Analyse de bank marketing')
df_file = st.sidebar.file_uploader("Upload a Dataset", type=['csv', 'txt'])
df = pd.read_csv(df_file)
option = st.sidebar.selectbox(
    'Quel menu voulez-vous voir ?',
     ('Étude Statistiques', 'Menu X', 'Menu XX'))
if option == 'Étude Statistiques':
    option = st.sidebar.selectbox(
        'Quel menu voulez-vous voir ?',
         ('Étude des variables', 'Menu 2', 'Menu 3'))
    
   
    if option == 'Étude des variables':
        st.header('_Visualisation de la distribution de la variable cible : deposit_')        
        
        
        fig1 = px.histogram(df, x="deposit")
    

        deposit_counts = df['deposit'].value_counts()
        labels = deposit_counts.index
        sizes = deposit_counts.values
        fig2 = go.Figure(data=go.Pie(labels=labels, values=sizes, textinfo='percent+label', insidetextorientation='radial'))
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)
        st.write('47.4% des clients de la banques ont souscrit un compte à terme')
        st.write("52.6% des clients de la banques n'ont pas souscrit un compte à terme")
      
    
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
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)
        

        education_counts = df['education'].value_counts()
        labels = education_counts.index
        sizes = education_counts.values
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
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
        print('soon')

    elif option == 'Menu 3':
        print('soon')
elif option == 'Menu X':
        print('soon')

elif option == 'Menu XX':
        print('soon')


    
    
        
