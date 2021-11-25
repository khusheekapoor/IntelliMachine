# importing basic libraries for data preprocessing
import streamlit as st
import numpy as np
import pandas as pd
import os

def app():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    showWarningOnDirectExecution = False

    # formatting the title
    st.title('Pre-Processing')

    if 'main_data.csv' in os.listdir('data'):

        df = pd.read_csv('data/main_data.csv')

        # handling missing values
        with st.expander('Handling Missing Values'):

            # displaying total number of missing values
            missing_values = df.isnull().sum()
            st.write('Total Missing Values: ', missing_values)

            # menu to handle missing values
            method = st.selectbox('Select the method to handle missing values:',('Select', 'Remove Observations','Remove Columns','Fill with Values'), key='method', help='Remove Observations with missing values in any of the columns or Remove Columns with lots of missing values or Fill with Values.')

            # removing observations
            if method=='Remove Observations':
                df = df.dropna()

            # removing columns
            if method=='Remove Columns':
                columns = st.multiselect('Select columns to remove:',(df.columns), key='columns', help='Remove columns with many missing values.')
                df = df.drop(columns=columns)


            # fill missing values
            if method=='Fill with Values':

                # for numerical columns
                if st.checkbox('Numeric Columns'):

                    # selecting column
                    column = st.multiselect('Select column:', (df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()), key='column_missing', help='Select column from dataset to handle missing values.')

                    # calculating number of missing values
                    missing_values = df[column].isnull().sum()
                    st.write('Number of missing values: ', missing_values)

                    # menu to impute missing values
                    if(missing_values>0):
                        method_impute = st.radio('Select method to impute:', ('Mean','Median','Mode','Given Value','Random Value'), key='method', help='Select method to fill the missing values in the dataset with.')
                        
                        # impute with mean
                        if method_impute=='Mean':
                            df[column] = df[column].fillna(df[column].mean())

                        # impute with median
                        if method_impute=='Median':
                            df[column] = df[column].fillna(df[column].median())
                        
                        # impute with mode
                        if method_impute=='Mode':
                            df[column] = df[column].fillna(df[column].getmode())
                        
                        # impute with given value
                        if method_impute=='Given Value':
                            value = st.number_input('Enter value:', key='value', help='Enter value to fill the missing values in column with')
                            df[column] = df[column].fillna(value)
                        
                        # impute with random values
                        if method_impute=='Random Value':
                            df[column] = df[column].apply(lambda x: np.random.choice(df[column].dropna().values) if np.isnan(x) else x)

                    
                
                # for categorical columns
                if st.checkbox('Categorical Columns'):

                    # selecting column
                    column = st.selectbox('Select column:', (df.select_dtypes(include=['category', 'object']).columns.tolist()), key='column_missing', help='Select column from dataset to handle missing values.')

                    # calculating number of missing values
                    missing_values = df[column].isnull().sum()
                    st.write('Number of missing values: ', missing_values)

                    # menu to impute missing values
                    if(missing_values>0):
                        method_impute = st.radio('Select method to impute:', ('Mode','Given Value'), key='method', help='Select method to fill the missing values in the dataset with.')

                        # impute with mode
                        if method_impute=='Mode':
                            df[column] = df[column].fillna(df[column].getmode())
                        
                        # impute with given value
                        if method_impute=='Given Value':
                            value = st.text_input_input('Enter value:', key='value', help='Enter value to fill the missing values in column with.')
                            df[column] = df[column].fillna(value)

            # handling missing values  
            if st.button('Finish'):
                st.write('Missing Values Handled!')
                st.dataframe(df)



        # transforming and scaling values
        with st.expander('Feature Scaling and Transformation'):

            # selecting columns
            columns = st.multiselect('Select column:', (df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()), key='column_scale', help='Select column from dataset to scale and normalize values.')

            # scaling columns

            # MinMax scaler    
            if st.checkbox('Scale'):

                # selecting type of scaler
                scaler = st.selectbox('Select type of scaler:', ('Select', 'MinMax', 'Standard', 'MaxAbs', 'Robust'), key='scaler', help='Select the type of scaler to scale values.')
               
                # scaling values basis type of scaler
                if scaler=='MinMax':
                    
                    # user-help to minmax scaler
                    st.write( """
                    The MinMax scaler is one of the simplest scalers to understand.  
                    It just scales all the data between 0 and 1. The formula for calculating the scaled value is- x_scaled = (x – x_min)/(x_max – x_min)
                    """)

                    # scaling the values
                    if st.button('Scale'):
                        
                        # executing the code
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df[columns] = scaler.fit_transform(df[columns])
                        
                        st.write('Scaled Successfully!')
                        st.dataframe(df)


                # Standard scaler
                if scaler=='Standard':

                    # user-help to standard scaler
                    st.write( """
                    For each feature, the Standard Scaler scales the values such that the mean is 0 and the standard deviation is 1(or the variance).
                    The formula for calculating the scaled value is- x_scaled = x – mean/std_dev
                    However, Standard Scaler assumes that the distribution of the variable is normal. Thus, in case, the variables are not normally distributed, we 
                    either choose a different scaler or first, convert the variables to a normal distribution and then apply this scaler
                    """)

                    # scaling the values
                    if st.button('Scale'):
                    
                        # executing the code
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df[columns] = scaler.fit_transform(df[columns])

                        st.write('Scaled Successfully!')
                        st.dataframe(df)


                
                # MaxAbs scaler
                if scaler=='MaxAbs':

                    # user-help to maxabs scaler
                    st.write( """
                    n simplest terms, the MaxAbs scaler takes the absolute maximum value of each column and divides each value in the column by the maximum value. 
                    Thus, it first takes the absolute value of each value in the column and then takes the maximum value out of those. This operation scales the data 
                    between the range [-1, 1].
                    """)

                    # scaling the values
                    if st.button('Scale'):
                    
                        # executing the code
                        from sklearn.preprocessing import MaxAbsScaler
                        scaler = MaxAbsScaler()
                        df[columns] = scaler.fit_transform(df[columns])

                        st.write('Scaled Successfully!')
                        st.dataframe(df)



                # Robust scaler
                if scaler=='Robust':

                    # user-help to robust scaler
                    st.write( """
                    The Robust Scaler, as the name suggests is not sensitive to outliers. This scaler- removes the median from the data and scales the data by the 
                    InterQuartile Range(IQR). The formula for calculating the scaled value is- x_scaled = (x – Q1)/(Q3 – Q1).
                    """)
                    
                    # scaling the values
                    if st.button('Scale'):
                    
                        # executing the code
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df[columns] = scaler.fit_transform(df[columns])

                        st.write('Scaled Successfully!')
                        st.dataframe(df)



            # transforming columns
            if st.checkbox('Transform'):

                # selecting type of transformer
                transformer = st.selectbox('Select type of transformer:', ('Quantile', 'Log', 'Square Root', 'Cube Root', 'Power'), key='transformer', help='Select method to transform data to normal distribution.')

                # scaling values basis type of transformer

                # Quantile Transformer
                if transformer=='Quantile':
                    
                    # user-help to quantile transformer
                    st.write("""
                    the Quantile Transformer Scaler converts the variable distribution to a normal distribution. and scales it accordingly. 
                    Since it makes the variable normally distributed, it also deals with the outliers.
                    """)

                    # transforming the values
                    if st.button('Transform'):
                        
                        # executing the code
                        from sklearn.preprocessing import QuantileTransformer
                        transformer = QuantileTransformer()

                        df[columns] = transformer.fit_transform(df[columns])
                    
                        st.write('Transformed Successfully!')
                        st.dataframe(df)



                # log transformer
                if transformer=='Log':

                    # user-help to log transformer
                    st.write("""
                    Log transform primarily used to convert a skewed distribution to a normal distribution/less-skewed distribution. 
                    In this transform, we take the log of the values in a column and use these values as the column instead.
                    """)

                    # transforming the values
                    if st.button('Transform'):

                        # executing the code
                        df[columns] = np.log(df[columns])
                    
                        st.write('Transformed Successfully!')
                        st.dataframe(df)



                # square root transformer
                if transformer=='Square Root':

                    # user-help to square root transformer
                    st.write("""
                    Square Root transform primarily used to convert a skewed distribution to a normal distribution/less-skewed distribution. 
                    In this transform, we take the square root of the values in a column and use these values as the column instead.
                    """)

                    # transforming the values
                    if st.button('Transform'):

                        # executing the code
                        df[columns] = np.sqrt(df[columns])
                    
                        st.write('Transformed Successfully!')
                        st.dataframe(df)


                # cube root transformer
                if transformer=='Cube Root':

                    # user-help to cube root transformer
                    st.write("""
                    Cube Root transform primarily used to convert a skewed distribution to a normal distribution/less-skewed distribution. 
                    In this transform, we take the cube root of the values in a column and use these values as the column instead.
                    """)

                    # transforming the values
                    if st.button('Transform'):

                        # executing the code
                        df[columns] = np.cbrt(df[columns])
                    
                        st.write('Transformed Successfully!')
                        st.dataframe(df)



                # power transformer
                if transformer=='Power':

                    # user-help to power transformer
                    st.write("""
                    It is used when dealing with heteroskedasticity. The Power Transformer also changes the distribution of the variable, as in, it makes it more 
                    Gaussian(normal). It introduces a parameter called lambda. It decides on a generalized power transform by finding the best value of lambda using either the:
                    Box-Cox Transformation, or the Yeo-Johnson Transformation. Not going into deeper math, it's important for us to know that the Box-Cox method works only for
                    positive values, whereas, the Yeo-Johnson method works for both, positive and negative values.
                    """)

                    power = st.radio('Select type of power transformer:', ('Box-Cox', 'Yeo-Johnson'), key='power', help='Box-Cox works for only positive values, whereas, Yeo-Johnson works for both.')     

                    # transforming basis type of power transformer

                    # box-cox transformer
                    if power=='Box-Cox':
                        
                        # transforming the values
                        if st.button('Transform'):

                            # executing the code
                            from sklearn.preprocessing import PowerTransformer
                            transformer = PowerTransformer(method='box-cox')

                            df[columns] = transformer.fit_transform(df[columns])

                            st.write('Transformed Successfully!')
                            st.dataframe(df)


                    # yeo-johnson transformer
                    if power=='Yeo-Johnson':        

                        # transforming the values
                        if st.button('Transform'):

                            # executing the code
                            from sklearn.preprocessing import PowerTransformer
                            transformer = PowerTransformer(method='yeo-johnson')

                            df[columns] = transformer.fit_transform(df[columns])

                            st.write('Transformed Successfully!')
                            st.dataframe(df)


        # encoding categorical columns
        with st.expander('Encode Categorical Columns'):

            # selecting columns
            columns = st.multiselect('Select column:', (df.select_dtypes(include=['object']).columns.tolist()), key='column_encode', help='Select categorical columns from dataset to encode to numerical columns.')

            # user-help to one-hot encoding
            st.write("""
            One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction.
            """)

            # one hot encoding the columns
            if st.button('Encode'):

                # executing the code
                from sklearn.preprocessing import OneHotEncoder
                ohe = OneHotEncoder()
                ohe_results = ohe.fit_transform(df[columns])
                df = df.join(pd.DataFrame(ohe_results.toarray(), columns=ohe.categories_))
                df = df.drop(columns=columns)

                st.write('Encoded Successfully!')
                st.dataframe(df)


        df.to_csv('data/main_data.csv', index=False)
 

    else:
        st.write('Please upload the dataset through the main page!')













