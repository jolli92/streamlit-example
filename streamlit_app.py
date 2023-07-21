import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
%matplotlib


file_path = 'bank.csv'


df = pd.read_csv(file_path)

st.title('Analyse de bank marketing')
df = st.file_uploader("Upload a Dataset", type=['csv', 'txt'])
if df is not None:
    df = pd.read_csv(df)
    st.dataframe(df.head())
    if st.button("Show summary"):
        st.write(df.describe())
    if st.button("Show Histogram"):
        st.write(sns.histplot(df))
       

