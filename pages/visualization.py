# importing basic libraries for data visualization
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt 
import seaborn as sns
import os


def app():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    showWarningOnDirectExecution = False

    # formatting the title
    st.title('Data Visualization')

    if 'main_data.csv' in os.listdir('data'):

        df = pd.read_csv('data/main_data.csv')

        # user input for type of visualization
        plot_type = st.selectbox('Please select type of visualization:', ('Select','Scatter Plot','Box Plot','Line Plot','Bar Plot','Histogram',
        'Correlation Heatmap','Pair Plot','Kernel Density Estimate Plot','Violin Plot','Swarm Plot','Strip Plot'))

        # Scatter Plot
        if plot_type == 'Scatter Plot':

            # user help to scatter plot
            st.write("""
            The scatter plots graphs pairs of numerical data, with one variable on each axis, to look for a relationship 
            between them. If the variables are correlated, the points will fall along a line or curve. The better the correlation, 
            the tighter the points will hug the line.
            """)

            # variables to plot the chart
            x = st.selectbox('Select x-axis variable:', df.columns, key='x')
            y = st.selectbox('Select y-axis variable:', df.columns, key='y')

            # grouping variables to differentiate between points
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between points based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None
            size = st.checkbox('Add size variable', key='size', help='Grouping variable to differentiate between points based on size.')
            if size:
                size = st.selectbox('Select size variable:', (df.columns))
            else:
                size = None
            style = st.checkbox('Add size variable', key='style', help='Grouping variable to differentiate between points based on shape.')
            if style:
                style = st.selectbox('Select style variable:', (df.columns))
            else:
                style = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.scatterplot(x=x, y=y, data=df, hue=hue, size=size, style=style)
                    st.pyplot()


        # Box Plot
        if plot_type == 'Box Plot':

            # user help to box plot
            st.write("""
            A boxplot is a standardized way of displaying the dataset based on a five-number summary: 
            the minimum, the maximum, the sample median, and the first and third quartiles.
            """)

            # variables to plot the chart
            x = st.selectbox('Select x-axis variable:', df.columns, key='x')
            y = st.selectbox('Select y-axis variable:', df.columns, key='y')

            # grouping variable to differentiate between boxes
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between points based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.code():
                    sns.boxplot(x=x, y=y, data=df, hue=hue)
                    st.pyplot()


        # Line Plot
        if plot_type == 'Line Plot':

            # user help to line plot
            st.write("""
            A line plot is a way to display data along a line over a period of time.
            """)

            # variables to plot the chart
            x = st.selectbox('Select x-axis variable:', df.columns, key='x')
            y = st.selectbox('Select y-axis variable:', df.columns, key='y')

            # grouping variable to differentiate between lines
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between lines based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None
            size = st.checkbox('Add size variable', key='size', help='Grouping variable to differentiate between lines based on size.')
            if size:
                size = st.selectbox('Select size variable:', (df.columns))
            else:
                size = None
            style = st.checkbox('Add size variable', key='style', help='Grouping variable to differentiate between lines based on shape.')
            if style:
                style = st.selectbox('Select style variable:', (df.columns))
            else:
                style = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.lineplot(x=x, y=y, data=df, hue=hue, size=size, style=style)
                    st.pyplot()



        # Bar Plot
        if plot_type == 'Bar Plot':

            # user help to bar plot
            st.write("""
            A bar plot is a graph that represents the category of data with rectangular bars with lengths 
            and heights that is proportional to the values which they represent. 
            """)

            # variable to plot the chart
            x = st.selectbox('Select variable to plot:', df.columns, key='x')

            # grouping variable to differentiate between bars
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between bars based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.barplot(x=x, data=df, hue=hue)
                    st.pyplot()



        # Histogram        
        if plot_type == 'Histogram':

            # user help to histogram
            st.write("""
            A histogram is a bar graph-like representation of data that buckets a range of outcomes into columns along the x-axis.
            The y-axis represents the number count or percentage of occurrences in the data for each column and can be used to 
            visualize data distributions.
            """)

            # variable to plot the chart
            x = st.selectbox('Select variable to plot:', df.columns, key='x')

            # grouping variable to differentiate between bars
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between bars based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.histplot(x=x, data=df, hue=hue)
                    st.pyplot()

                

        # Correlation Heatmap        
        if plot_type == 'Correlation Heatmap':

            # user help to correlation heatmap
            st.write("""
            A correlation heatmap is a heatmap that shows a 2D correlation matrix between two discrete dimensions, using colored cells 
            to represent data from usually a monochromatic scale. The values of the first dimension appear as the rows of the table while 
            of the second dimension as a column. The color of the cell is proportional to the number of measurements that match the 
            dimensional value.
            """)
            
            # switch for annotation
            annot = st.checkbox('Annotate', key='annot', help='Annotate correlation values')
            
            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.heatmap(data=(df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])).corr(), annot=annot)
                    st.pyplot()



        # Pair Plot        
        if plot_type == 'Pair Plot':

            # user help to pair plot
            st.write("""
            Pair plots allow us to visualize the relationship between all the variables in a dataset taken two at a time.
            """)

            # grouping variable to differentiate between points
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between points based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None
            
            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.pariplot(data=df, hue=hue)
                    st.pyplot()



        # Kernel Density Estimate Plot        
        if plot_type == 'Kernel Density Estimate Plot':

            # user help to kde plot
            st.write("""
            Kernel Density Estimate plot represents the data using a continuous probability density curve in one or more dimensions. 
            KDE plot smooths the observations with a Gaussian kernel, producing a continuous density estimate.
            """)

            # variables to plot the chart
            x = st.selectbox('Select x-axis variable:', df.columns, key='x')
            y = st.selectbox('Select y-axis variable:', df.columns, key='y')

            # grouping variable to differentiate between lines
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between lines based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None

            # switch to shade the area beneath the curve
            fill = st.checkbox('Fill', key='fill', help='Switch to highlight the area beneath the curve')

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.kdeplot(x=x, y=y, data=df, hue=hue, fill=fill)
                    st.pyplot()



        # Violin Plot        
        if plot_type == 'Violin Plot':

            # user help to violin plot
            st.write("""
            In general, violin plots are a method of plotting numeric data and can be considered a combination of the 
            Box Plot with a Kernel Density Estimate Plot. It gives us the five number summary of the data while plotting a 
            continuous density estimate after smoothening it with a Gaussian kernel.
            """)

            # variables to plot the chart
            x = st.selectbox('Select x-axis variable:', df.columns, key='x')
            y = st.selectbox('Select y-axis variable:', df.columns, key='y')

            # grouping variable to differentiate between lines
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between lines based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.violinplot(x=x, y=y, data=df, hue=hue)
                    st.pyplot()



        # Swarm Plot        
        if plot_type == 'Swarm Plot':

            # user help to swarm plot
            st.write("""
            The swarm plots graphs pairs of numerical data and cateogircal data, with one variable on each axis, to look for a 
            relationship between them.
            """)

            # variables to plot the chart
            x = st.selectbox('Select x-axis variable:', df.columns, key='x')
            y = st.selectbox('Select y-axis variable:', df.columns, key='y')

            # grouping variable to differentiate between points
            hue = st.checkbox('Add hue variable', key='hue', help='Grouping variable to differentiate between points based on colour.')
            if hue:
                hue = st.selectbox('Select hue variable:', (df.columns))
            else:
                hue = None

            # plotting the chart
            if(st.button('Visualize')):

                # printing and executing the code
                with st.echo():
                    sns.swarmplot(x=x, y=y, data=df, hue=hue)
                    st.pyplot()


    else:
        st.write('Please upload the dataset through the main page!')
    