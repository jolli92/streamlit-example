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

st.title('Analyse de bank marketing')
df = pd.read_csv('bank.csv')
option = st.sidebar.selectbox('Quel menu voulez-vous voir ?', ('Etude statistiques 📈', 'Menu X', 'Menu XX'))

if option == 'Etude statistiques 📈':
    option = st.sidebar.selectbox('Quel menu voulez-vous voir ?', ('Analyse des informations brutes', 'Etude des variables', 'Menu 3'))

    if option == 'Analyse des informations brute':
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

    elif option == 'Menu 3':
        st.header("Effet de l'age sur Deposit")
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
        fig.update_layout(xaxis_title="Âge",
        yaxis_title="Nombre de clients",
        hovermode="x",
        autosize=False,
        width=800,
        height=500,
    )
        st.plotly_chart(fig)
        st.write("""
Selon notre analyse, même si une partie significative des clients d'âge moyen souscrit à des dépôts à terme, il est notable qu'une majorité d'entre eux n'y souscrit pas. 

L'analyse montre aussi que les clients les plus âgés sont plus enclins à souscrire à des dépôts à terme, avec moins d'entre eux qui choisissent de ne pas y souscrire. 

Par conséquent, il serait judicieux pour les banques de cibler davantage cette catégorie d'âge pour augmenter le nombre de souscriptions aux dépôts à terme.
""")



    
        st.header('Effet du mois sur deposit')
        deposit_yes = df[df['deposit'] == 'yes']
        deposit_no = df[df['deposit'] == 'no']
        count_yes = deposit_yes['month'].value_counts().sort_index()
        count_no = deposit_no['month'].value_counts().sort_index()
        bar_width = 0.35
        months = range(len(count_yes.index))
        fig, ax = plt.subplots(figsize=(10,6))
        bar1 = ax.bar(months, count_yes.values, bar_width, label='Deposit Yes', color='#66B3FF')
        bar2 = ax.bar([month + bar_width for month in months], count_no.values, bar_width, label='Deposit No', color='#FF9999')
        for i, value in enumerate(count_yes.values):
            ax.text(i, value, f"{value/df.shape[0]:.2%}", ha='center', va='bottom')
        for i, value in enumerate(count_no.values):
            ax.text(i + bar_width, value, f"{value/df.shape[0]:.2%}", ha='center', va='bottom')
        ax.set_xlabel('Mois')
        ax.set_ylabel('Nombre de clients')
        ax.set_xticks([month + bar_width / 2 for month in months])
        ax.set_xticklabels(count_yes.index)
        ax.legend()
        st.pyplot(fig)
        st.write("""
Les mois de mai, juin, juillet et août de l'année précédente ont été les plus actifs en termes de contacts avec les clients de la banque. C'est également pendant ces périodes que le nombre de souscriptions aux dépôts à terme a été le plus élevé.

Cependant, les mois de septembre, mars et décembre, malgré une moindre activité en matière de contacts, ont vu un taux de souscription aux dépôts à terme supérieur. Il serait donc judicieux de concentrer davantage d'efforts pour contacter les clients pendant ces périodes.
""")


    
        st.header('Effet de campaign sur deposit')
        campaign_counts_yes = deposit_yes['campaign'].value_counts().sort_index()
        campaign_counts_no = deposit_no['campaign'].value_counts().sort_index()
        counts_df_yes = pd.DataFrame(campaign_counts_yes).reset_index()
        counts_df_yes.columns = ['Campaign', 'Count']
        counts_df_yes['Deposit'] = 'Yes'
        counts_df_no = pd.DataFrame(campaign_counts_no).reset_index()
        counts_df_no.columns = ['Campaign', 'Count']
        counts_df_no['Deposit'] = 'No'
        counts_df = pd.concat([counts_df_yes, counts_df_no])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=counts_df['Campaign'][counts_df['Deposit'] == 'Yes'], y=counts_df['Count'][counts_df['Deposit'] == 'Yes'], mode='lines+markers', name='Deposit Yes', line=dict(color='#66B3FF')))
        fig.add_trace(go.Scatter(x=counts_df['Campaign'][counts_df['Deposit'] == 'No'], y=counts_df['Count'][counts_df['Deposit'] == 'No'], mode='lines+markers', name='Deposit No', line=dict(color='#FF9999')))
        fig.update_layout(xaxis_title='Campagne', yaxis_title='Nombre de clients', legend_title='Deposit', autosize=False, width=1000, height=600, margin=dict(l=50, r=50, b=100, t=100, pad=4))
        st.plotly_chart(fig)
        st.write("""
Selon notre analyse, plus nous multiplions les contacts avec les clients lors d'une campagne, plus il est probable qu'ils ne souscrivent pas aux dépôts à terme . 

Ainsi, pour augmenter les souscriptions aux dépôts à terme, il serait avantageux de limiter le nombre de contacts avec le client.
""")



    
        st.header('Effet de previous sur deposit')
        trace0 = go.Scatter(
         x=campaign_counts_yes.index,
         y=campaign_counts_yes.values,
         mode='lines',
         name='Deposit Yes',
         line=dict(color='#66B3FF')
)
        trace1 = go.Scatter(
        x=campaign_counts_no.index,
        y=campaign_counts_no.values,
        mode='lines',
        name='Deposit No',
        line=dict(color='#FF9999')
)
        data = [trace0, trace1]
        layout = go.Layout(xaxis=dict(title='Previous'),
        yaxis=dict(title='Nombre de clients'),
)
        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig)
        st.write("""
Selon nos observations, plus un client a été contacté avant cette campagne, plus il est susceptible de ne pas souscrire aux dépôt à terme. 
Pour optimiser les résultats, il serait judicieux de limiter le nombre de contacts à moins de 3.
""")

        st.header('Effet de Poutcome sur deposit')
        def plot_interactive(df):
            poutcome_unique = df['poutcome'].unique()
            deposit_yes = df[df['deposit'] == 'yes']
            deposit_no = df[df['deposit'] == 'no']

            deposit_yes_counts = deposit_yes['poutcome'].value_counts().reindex(poutcome_unique, fill_value=0)
            deposit_no_counts = deposit_no['poutcome'].value_counts().reindex(poutcome_unique, fill_value=0)

            fig = go.Figure(data=[
            go.Bar(name='Deposit Yes', x=poutcome_unique, y=deposit_yes_counts, marker_color='#66B3FF', text=[f"{(i / j) * 100:.1f}%" for i, j in zip(deposit_yes_counts, deposit_yes_counts + deposit_no_counts)], textposition='auto'),
            go.Bar(name='Deposit No', x=poutcome_unique, y=deposit_no_counts, marker_color='#FF9999', text=[f"{(i / j) * 100:.1f}%" for i, j in zip(deposit_no_counts, deposit_yes_counts + deposit_no_counts)], textposition='auto')
    ])
            fig.update_layout(barmode='group', xaxis_title='Poutcome', yaxis_title='Nombre de clients')
            return fig


        fig = plot_interactive(df)
        st.plotly_chart(fig)
        st.write("""
Selon les résultats de la campagne précédente, lorsque l'issue est un échec, il y a 50 % de chances que le client ne souscrive pas au dépôt à terme. Parmi tous les échecs, 50,3 % des clients décident de souscrire, tandis que 49,7 % choisissent de ne pas souscrire au dépôt à terme.

En revanche, si l'issue est un succès, il y a une forte probabilité que le client souscrive au dépôt à terme. Parmi tous les succès, 91,3 % des clients s'abonnent, tandis que 8,7 % ne s'abonnent pas au dépôt à terme.
""")

        st.header('Tests statistiques')
        st.write('Tests statistiques variable catégorielle : utilisation de chi²')
        cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']

        chi2_p_values = {}

        for feature in cat_features:
            if feature != 'deposit':
                contingency_table = pd.crosstab(df[feature], df['deposit'])
                _, p, _, _ = chi2_contingency(contingency_table)
                chi2_p_values[feature] = p
        st.write(chi2_p_values)

        st.write('Tests statistiques variable catégorielle : utilisation du test t de student')
        num_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        ttest_p_values = {}
        for feature in num_features:
            group1 = df[df['deposit'] == 'yes'][feature]
            group2 = df[df['deposit'] == 'no'][feature]
            _, p = ttest_ind(group1, group2)
            ttest_p_values[feature] = p
        st.write(ttest_p_values)
        st.write("""
Les valeurs de p des tests du Chi-carré pour les variables catégorielles et des tests t pour les variables numériques sont toutes significativement inférieures à 0,05. Cela signifie que nous pouvons rejeter l'hypothèse nulle pour ces variables. Par conséquent, il existe une différence statistiquement significative entre les groupes de dépôt (yes et no) pour chaque variable numérique.

En résumé, les tests du Chi-carré pour les variables catégorielles et les tests t pour les variables numériques suggèrent que toutes ces caractéristiques ont une relation statistiquement significative avec la variable de dépôt. Par conséquent, nous pouvons dire que toutes ces variables pourraient potentiellement avoir un effet sur la décision d'un client de faire un dépôt ou non. Cependant, il est important de se rappeler que la corrélation n'implique pas la causalité, et ces résultats ne nous indiquent pas comment ces variables influencent le résultat du dépôt. Pour cela, une investigation plus approfondie et éventuellement une modélisation prédictive seraient nécessaires.
""")


        st.header('Conclusion')
        st.write("""
L'année passée, la plupart des interactions avec les clients de la banque ont eu lieu entre les mois de mai et août. Cependant, le mois de mai, qui a connu le plus grand nombre de contacts, a également vu le moins d'adhésion aux dépôts à terme. Les mois de mars, septembre et décembre ont vu peu de contacts, et il serait bénéfique de privilégier ces périodes pour une meilleure communication.

Lorsqu'un client est sollicité par plusieurs campagnes ou est contacté plusieurs fois, il a tendance à se désintéresser des dépôts à terme. Il est donc recommandé de limiter les interactions à deux ou trois tentatives au maximum.

Au niveau des professions, les retraités, les étudiants et les aides ménagères semblent être les plus enclins à opter pour les dépôts à terme. Les retraités, qui dépensent généralement peu, sont plus disposés à investir leur argent dans une institution financière. Les étudiants forment également un groupe privilégié pour la souscription aux dépôts à terme.

Malgré une souscription notable aux dépôts à terme chez les clients d'âge moyen, ils sont plus nombreux à ne pas souscrire. Par contre, les clients âgés adhèrent davantage à ces produits et sont moins nombreux à refuser. Il serait donc profitable pour les banques de concentrer leurs efforts sur la clientèle âgée pour augmenter les souscriptions aux dépôts à terme.

En matière de résultats de campagnes antérieures, un échec conduit à une probabilité de 50% pour le client de ne pas souscrire au dépôt à terme. Par contre, si le résultat de la campagne précédente a été positif, les chances que le client souscrive sont élevées. Précisément, 91,3 % des succès ont abouti à une souscription, contre 8,7 % qui n'ont pas abouti.

Les clients qui n'ont pas d'intérêt pour les prêts immobiliers pourraient être intéressés par les dépôts à terme. 

De plus, un solde client supérieur à la moyenne est un indicateur positif de souscription à un dépôt à terme.
""")




elif option == 'Menu X':
    print('soon')
    if option == 'Test Prediction':
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
        encoder = OneHotEncoder(drop='first', sparse=False)

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
        st.dataframe(X_train.info())


    elif option == 'Menu XX':
        print('soon')
