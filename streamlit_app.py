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
        st.write(" * Information client :")
        st.write("l'aperçu des données numériques montrent l'échantillon de la population varie de 18 à 95 ans avec un moyen d'âge de 41 ans et une grande majorité vers les 49 ans.")
        st.write("leur compte courant qui varie entre un déficit de -6 847 et un crédit de 81 204 (euro?), en moyen les clients a  1 528 (euro?) sur leurs comptes.")
        st.write(" * Information campagne étudiée :")
        st.write("Les clients sont contactés durant la campagne en moyen 2 et 3 fois. La mayorité des réponses est obtenue au de la troisème prise de contact.")
        st.write("Un point d'attention est mis sur la valeur 63 affectée à la prise de contact durant la campagne... peut-on déjà en déduire que c'est une valeur abérrante ?..pourtant pas impossible ..")
        st.write("Nous constatons aussi que le dernier contact avec le client se localise vers la fin du mois, le 22 avec une durée entre 2 secondes et 1h.")
        st.write("La majorité des réponses est obtenue entre 6 et 8 mins, pouvons-nous dire que les clients qui nécessitent de rester 1h en ligne sont plus difficile à convaincre ?")
        st.write(" * Information campagne précedente :")
        st.write("On constate que 50% de l'échantillon n'ont jamais été contacté (previous=0) avant la campagne, ce qui est en cohérence avec le nombre de jour séparant avec le dernier contact (pdays = -1).")
        st.write("Peut-on dire que l'échantillon conntient essentiellement de nouveaux contacts clients ?")
        st.write("et 75% de ceux qui ont été contacté, ont été contacté au bout de 21 jours et n'a eu en générale qu'un seul contact.")
        st.write("Pour ceux qui ont été régulièrement coontacté (previous=58), ils ne le sont que au bout de plus de deux ans...Ce qui attire notre attention sur la cohérence entre le nombre de fois que le client a été contacté (58) et \nla durée du dernier contact plus de ans... Peut-on déduire que c'est un client en portefeuille depuis trop longtemps??")
        st.write("ou au contraire le client n'a été très solicité puisque il l'a été que au bout de plus de deux ans ...mais dans ce cas le nombre de contact, 58 fois avant la campagne nous pose quelques questions...58 fois")
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
        st.write("Management et blue colar sont les métiers les plus représentés chez les clients de la banque.")
        st.write("La majorité des clients de la banques sont mariés (56.9%) ou célibataire (31.5%).")
        st.write("La majorité des clients de la banques sont issues d'un cursus de second cycle (49.1%) et de troisiéme cycle (33%)")
        st.write("Seulement 11% des clients sont issues d'un cursus de premier cycle")

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

    elif option == 'Menu 3':
        age_counts_yes = df[df['deposit'] == 'yes']['age'].value_counts().sort_index()
        age_counts_no = df[df['deposit'] == 'no']['age'].value_counts().sort_index()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
        x=age_counts_yes.index, 
        y=age_counts_yes.values, 
        mode='lines',
        name='Deposit Yes',
        line=dict(color='#66B3FF')
    ))

    fig.add_trace(go.Scatter(
        x=age_counts_no.index, 
        y=age_counts_no.values, 
        mode='lines',
        name='Deposit No',
        line=dict(color='#FF9999')
    ))

    fig.update_layout(
        title_text="Effet de l'âge sur deposit",
        xaxis_title="Âge",
        yaxis_title="Nombre de clients",
        hovermode="x",
        autosize=False,
        width=800,
        height=500,
    )

    st.plotly_chart(fig)

elif option == 'Menu X':
    print('soon')

elif option == 'Menu XX':
    print('soon')
