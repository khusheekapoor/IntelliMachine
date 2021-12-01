# IntelliMachine

Link to the application: https://share.streamlit.io/khusheekapoor/intellimachine/main/app.py

## Idea

IntelliMachine is a beginner-friendly Machine Learning Portal inspired from the concept of AutoML where users can deploy Machine Learning Algorithms, perform
EDA techniques, and make Visualizations on datasets with the help of some buttons and no-code with explanations all throughout.

## Application Description

The Home Page of the application doubles as the [Data Upload Page](/pages/upload.py). The user has the option to upload a csv or an excel file.

<img src="https://user-images.githubusercontent.com/74901388/144238415-ebb207b5-b765-4602-9295-8ce9df5f9632.png" width="700" height="300" alt="Home Page">

After uploading the dataset, on clicking the Load Data button, the user can see the dataset loaded into a Pandas dataframe, alongwith the number of rows and columns in the dataframe.

<img src="https://user-images.githubusercontent.com/74901388/144238816-d1057c02-f7bc-447b-8f63-9311f243d6c5.png" width="700" height="300" alt="Load Data">

To proceed with the analysis, the user can use the Navigation Menu on the left pane, which connects the user with the Pre-Processing, Visualization, and the Model Building Pages.
The [Navigation Menu](/multipage.py) uses OOPs principles to connect the pages together. 

<img src="https://user-images.githubusercontent.com/74901388/144239134-27becb54-a50b-4101-9e13-3cf57b2506ab.png" width="700" height="300" alt="Navigation">

The [Pre-Processing Page](/pages/processing.py) presents three strategies - Handling Missing Values, Feature Scaling and Transformation, and Encoding Categorical Columns. 

<img src="https://user-images.githubusercontent.com/74901388/144239430-27536660-df19-485d-a285-d207f2fa70b9.png" width="700" height="300" alt="Pre-Processing Home">

Let's explore the first option - Handling Missing Values. On expanding the option, the user is provided with a summary of the missing values per column. To handle them, three options
are available - Removing Columns, Removing Observations, and Filling with Values (mean/ median/ mode/ user-defined constant).

<img src="https://user-images.githubusercontent.com/74901388/144240881-e935e7e9-4ae1-4d6d-a40c-2a6853a24371.png" width="700" height="300" alt="Missing Values">

To make the application more beginner-friendly, a tool-tip with a small explanation is provided.

<img src="https://user-images.githubusercontent.com/74901388/144241118-d65e402c-a86f-4d8e-8826-5a48bd1f1a77.png" width="700" height="300" alt="Tool Tip">

Venturing into the second option - Feature Scaling and Transformation. 

<img src="https://user-images.githubusercontent.com/74901388/144241512-cce08f0c-6c9a-4fb8-b7b9-9d6ab4feb900.png" width="700" height="300" alt="Scaling and Transformation">

A variety of Feature Scaling and Transformation options have been given to the user - including Standard, MinMax, MaxAbs and Robust scalers and Log, Square Root, and Cube Root 
transformers.

<img src="https://user-images.githubusercontent.com/74901388/144241663-69ebae5f-980e-4ca0-8256-94d9b356aae3.png" width="700" height="300" alt="Scaling">

<img src="https://user-images.githubusercontent.com/74901388/144241735-aa6fe9c0-406b-4801-892f-eef8a9fe1d06.png" width="700" height="300" alt="Transformation">

A success message is posted and the changed dataframe is displayed at the end of the function.

<img src="https://user-images.githubusercontent.com/74901388/144242071-69309a7e-0702-42d8-b24d-66c093cb09e8.png" width="700" height="300" alt="Success">

The user can also encode categorical columns by exploring the Encode Categorical Columns option.

<img src="https://user-images.githubusercontent.com/74901388/144242303-a8c0253d-168d-41ef-a32e-08f62e563d46.png" width="700" height="300" alt="Encoding">

Navigating to the [Data Visualization Page](/pages/visualization.py). The user has a number of charts to choose from for plotting - including Scatter, Line, Bar,
Violin, KDE, and Pair plots, Heatmap, and Histogram.

<img src="https://user-images.githubusercontent.com/74901388/144243093-446cf99d-282e-4d2f-acc7-aaf07ca07e80.png" width="700" height="300" alt="Visualization">

The user can also add hue/ shape/ and size variables. 

<img src="https://user-images.githubusercontent.com/74901388/144244720-d6cadadd-d877-436e-8da0-3154486eb969.png" width="700" height="300" alt="Differentiators">

On clicking Visualize, the chart is plotted. The code to replicate the chart is also printed.

<img src="https://user-images.githubusercontent.com/74901388/144244478-71294528-b0aa-4a77-8da8-e137c2db0b72.png" width="700" height="300" alt="Plot">

Coming to [Model Building](/pages/models.py), one of the most exciting features. The user can solve two types of Supervised Machine Learning problems - Regression and Classification.

<img src="https://user-images.githubusercontent.com/74901388/144245763-46054f7d-0be2-4ec6-ac95-07600545a5d9.png" width="700" height="300" alt="Model Building">

The user has numerous algorithms to try out - including Regression, Descision Trees, Random Forests, AdaBoost and XGBoost.

<img src="https://user-images.githubusercontent.com/74901388/144245962-de819c58-83ad-4c30-bf5f-12aa22e78e5a.png" width="700" height="300" alt="Options">

The user can select features and set the training size suited to their data.

<img src="https://user-images.githubusercontent.com/74901388/144246153-4eb9444c-6936-4631-93c4-564828e4dda8.png" width="700" height="300" alt="Details">

At the end of running the algorithm, the user is provided with a number of evaluation metrics - including MSE, RMSE, ROC Curve, and Confusion Matrix, and the code to replicate the same.

<img src="https://user-images.githubusercontent.com/74901388/144246388-0306ea60-f8ea-474d-bc4a-d9a73bee9161.png" width="700" height="300" alt="Regression Results">

<img src="https://user-images.githubusercontent.com/74901388/144246490-61e7e099-2a71-475f-8706-1177eacdf523.png" width="700" height="300" alt="Classification Results">

## Tech Stack

### For Data Preprocessing:

[NumPy](https://numpy.org/) &nbsp; [Pandas](https://pandas.pydata.org/) &nbsp; [ScikitLearn](https://scikit-learn.org/stable/)

### For Data Visualization:

[MatPlotLib](https://matplotlib.org/) &nbsp; [Seaborn](https://seaborn.pydata.org/)

### For Model Building:

[ScikitLearn](https://scikit-learn.org/stable/) &nbsp; [XGBoost](https://xgboost.readthedocs.io/en/stable/)

### For Deployment:

[Streamlit](https://streamlit.io/)


