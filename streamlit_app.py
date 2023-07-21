import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st



file_path = 'bank.csv'


df = pd.read_csv(file_path)

st.title('Analyse de bank marketing')
if df is not None:
    data = pd.read_csv(df)
    st.dataframe(df.head())
    if st.button("Show summary"):
        st.write(df.describe())
    if st.button("Show Histogram"):
        st.write(sns.histplot(df))
        st.pyplot()
#
