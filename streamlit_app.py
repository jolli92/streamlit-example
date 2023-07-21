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
    plt.figure(figsize=(6,4))
    sns.countplot(x="deposit", data=df)
    st.pyplot()

    # Ajoutez ici les autres graphiques, en suivant le même format. Par exemple:

    st.header('Souscription à un compte à terme')
    deposit_counts = df['deposit'].value_counts()
    labels = deposit_counts.index
    sizes = deposit_counts.values
    plt.figure(figsize=(6,4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Répartition des dépôts')
    st.pyplot()

