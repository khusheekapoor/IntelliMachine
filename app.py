# importing streamlit
import streamlit as st

# configuring the page
st.set_page_config(page_title='IntelliMachine', layout='centered', initial_sidebar_state='auto')

# importing basic libraries and pages
import numpy as np
import pandas as pd
from pages import upload, processing, visualization, models
from multipage import MultiPage

# formatting the title
st.title('IntelliMachine')

# creating instance of the app
app = MultiPage()

# adding pages
app.add_page('Data Upload', upload.app)
app.add_page('Pre-Processing', processing.app)
app.add_page('Data Visualization', visualization.app)
app.add_page('Model Building', models.app)

# the main application
app.run()
