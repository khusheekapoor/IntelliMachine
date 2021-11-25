# importing the basic libraries to manipulate data
import streamlit as st
import numpy as np
import pandas as pd


def app():

    # formatting the title
    st.title('Data Upload')

    # accepting uplaoded file from user
    st. write('Upload Dataset')
    uploaded_file = st.file_uploader('Please upload a file:', type = ['csv', 'xlsx'])
    global df
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)
    
    # making a local copy of the dataset for globalization purposes
    if st.button('Load Data'):
        
        st.dataframe(df)
        st.write('Number of Rows: ', df.shape[0])
        st.write('Number of Columns: ', df.shape[1])
        df.to_csv('data/main_data.csv', index=False)
 