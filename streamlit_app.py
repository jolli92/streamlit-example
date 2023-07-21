import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st




df = pd.read_csv("/workspaces/streamlit-example/bank.csv")
st.title('Analyse de bank marketing')
df = st.file_uploader("Upload a Dataset", type=['csv', 'txt'])
if df is not None:
    data = pd.read_csv(df)
    st.dataframe(data.head())
    if st.button("Show summary"):
        st.write(data.describe())
    if st.button("Show Histogram"):
        st.write(sns.histplot(data))
        st.pyplot()
