# importing basic libraries for building the model
import streamlit as st
import numpy as np
import pandas as pd
import os

# function to plot evaluation metrics based on user input
def plot_metrics(model,metrics_list,X_test,Y_test):

    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

    # confusion matrix
    if 'Confusion Matrix' in metrics_list:
        st.subheader('Confusion Matrix')

        # printing and executing code
        with st.echo():
            plot_confusion_matrix(model,X_test,Y_test)
            st.pyplot()
    
    # ROC curve
    if 'ROC Curve' in metrics_list:
        st.subheader('ROC Curve')

        # printing and executing code
        with st.echo():
            plot_roc_curve(model,X_test,Y_test)
            st.pyplot()    

    # precision-recall curve
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-recall Curve')

        # printing and executing code
        with st.echo():
            plot_precision_recall_curve(model,X_test,Y_test)
            st.pyplot() 


def app():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    showWarningOnDirectExecution = False

    # formatting the title
    st.title('Model Building')


    if 'main_data.csv' in os.listdir('data'):

        df = pd.read_csv('data/main_data.csv')

        # user input for type of problem
        option = st.selectbox('Please select the type of problem:',('Select','Classification','Regression')) 
            

        if option!='Select':
            
            # splitting the dataset for training and testing
            def split(df):

                # selecting type of problem
                target = st.selectbox('Select the target variable for prediction:', df.columns, key='target', help='Select varaible to predict.')
                columns = st.multiselect('Select independent variables for model building:', (df.columns), key='columns', help='You may drop identifier columns or highly correlated features.')

                from sklearn.model_selection import train_test_split

                # segregating the independent and dependent variables
                Y = df[target]    
                X = df[columns]

                # splitting the dataset for training and testing
                train_size = st.slider('Training Set Size:', min_value=0.05, max_value=1.0, step=0.05, key='train_size', help='Select the training data size.')
                
                X_train,X_test,Y_train,Y_test = train_test_split(X, Y, random_state=0, train_size=train_size)
                return X_train,X_test,Y_train,Y_test

            X_train,X_test,Y_train,Y_test = split(df)



        # code for classfication type of problem
        if option == 'Classification':

            # importing libraries for model evaluation
            from sklearn.metrics import precision_score, recall_score

            # options for different types of models
            classifier = st.selectbox('Classification Algorithm:',('Select','Logistic Regression','Support Vector Machine','K-Nearest Neighbours','Gaussian Naive Bayes',
            'Decision Trees','Random Forests','AdaBoost','XGBoost','XGBoost with Random Forests'))

            # Logistic Regression
            if classifier == 'Logistic Regression':

                # user help to logistic regression
                st.write("""
                Logistic Regression is a Supervised Machine Learning' algorithm that can be used to model the probability 
                of a certain class or event. It is used when the data is linearly separable and the outcome is binary or dichotomous in nature.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                C = st.number_input('C (Regularization Parameter):', min_value=0.01, max_value=10.0, step=0.01, key='C', help='Inverse of regularization strength.')
                max_iter = st.number_input('Number of Iterations:', min_value=100, max_value=1000, step=10, key='max_iter', help='Maximum number of iterations taken for the solvers to converge.')
                penalty = st.radio('Select type of Penalty:',('none','l1','l2'), key='penalty', help='none: no penalty added\nl1:add an L1 penalty term\nl2: add an L2 penalty term')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')

                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tuned hyperparameters
                if st.button('Classify'):

                    from sklearn.linear_model import LogisticRegression
                    st.subheader('Logistic Regression Results')
                    st.spinner(text='Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, random_state=random_state)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)

                    # printing results and plotting metrics
                    st.write('Accuracy: ', accuracy)
                    st.write('Precision: ', precision)
                    st.write('Recall: ', recall)
                    plot_metrics(model,metrics,X_test,Y_test)      



            # Support Vector Machine
            if classifier == 'Support Vector Machine':

                # user help to support vector machine
                st.write("""
                Support Vector Machine is a Supervised Machine Learning algorithm where we plot each data item as a point in n-dimensional 
                space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. 
                Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                C = st.number_input('C (Regularization Parameter):', min_value=0.01, max_value=10.0, step=0.01, key='C', help='Inverse of regularization strength.')
                kernel = st.selectbox('Kernel:', ('rbf', 'linear', 'poly', 'sigmoid'), key='kernel', help='Specifies the type of kernel to use to transform the dimensions so that they are easily separable.')
                gamma = st.radio('Gamma (Kernel Coefficient):', ('auto', 'scale'), key='gamma', help='Kernel Coefficent.')
                probability = st.radio('Probability Estimates:',('True','False'), key='probability', help='To determine probability estimates using 5 fold cross validation.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tuned hyperparameters
                if st.button('Classify'):

                    from sklearn.svm import SVC
                    st.subheader('Support Vector Classification Results')
                    st.spinner(text='Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability, random_state=random_state)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall =recall_score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Accuracy: ', accuracy)
                    st.write('Precision: ', precision)
                    st.write('Recall: ', recall)
                    plot_metrics(model,metrics,X_test,Y_test) 



            # K-Nearest Neighbors
            if classifier == 'K-Nearest Neighbours':

                # user help to k-nearest neighbours
                st.write("""
                K Nearest Neighbour is a Supervised Machine Learning algorithm that stores all the available cases and classifies 
                the new data or case based on a similarity measure. It is mostly used to classifies a data point based on how its 
                neighbours are classified.
                """)
                
                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_neighbors = st.number_input('Number of Neighbors:', min_value=1, max_value=20, step=1, key='n_neighbors', help='Number of neighbours used to classify.')
                weights = st.radio('Weights:', ('uniform', 'distance'), key='weights', help='uniform: all neighbours are weighted equally\ndistance: neighbours are weighted inversely proportional to their distance from the data point')
                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tuned hyperparameters
                if st.button('Classify'):
                    from sklearn.neighbors import KNeighborsClassifier
                    st.subheader('K-Nearest Neighbors Results')
                    st.spinner(text='Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Accuracy: ', accuracy)
                    st.write('Precision: ', precision)
                    st.write('Recall: ', recall)
                    plot_metrics(model,metrics,X_test,Y_test)      



            # Guassian Naive Bayes
            if classifier == 'Gaussian Naive Bayes':

                # user help to gaussian naive bayes
                st.write("""
                Gaussian Navie Bayes is a Supervise Machine Learning model that uses the Baye's theorem to calculate 
                probabilites for events to occur with the “naive” assumption of conditional independence between every pair 
                of features given the value of the class variable and classifies the target variable accordingly.
                """)
                                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model
                if st.button('Classify'):
                    from sklearn.naive_bayes import GaussianNB
                    st.subheader('Gaussian Naive Bayes Results')
                    st.spinner(text='Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = GaussianNB()
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)
                    
                    # evaluating the model
                    st.write('Accuracy: ',accuracy)
                    st.write('Precision: ',precision)
                    st.write('Recall: ',recall)
                    plot_metrics(model,metrics,X_test,Y_test)    



            # Decision Trees
            if classifier == 'Decision Trees':

                # user help to decision trees
                st.write("""
                Decision Trees is a Supervised Learning Machine Learning algorithm. The goal is to create a model that predicts the value 
                of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise 
                constant approximation.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                criterion = st.radio('Criteria for Split:', ('gini', 'entropy'), key='criterion', help='The criteria for split of node\bgini: based on gini imppurity index\nentropy: based on information gain')
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=1, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tunes hyperparameters
                if st.button('Classify'):
                    from sklearn.tree import DecisionTreeClassifier
                    st.subheader('Decision Tree Classifier Results')
                    st.spinner(text='Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test, Y_pred)
                        recall = recall_score(Y_test, Y_pred)

                    # evaluating the model
                    st.write('Accuracy: ',accuracy)
                    st.write('Precision: ',precision)
                    st.write('Recall: ',recall)
                    plot_metrics(model,metrics,X_test,Y_test)          



            # Random Forests
            if classifier == 'Random Forests':

                # user help to random forests
                st.write("""
                A random forest is a Supervised Machine Learning algorithm that fits a number of decision tree classifiers on various 
                sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Bagged:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The number of trees in the forest.')              
                criterion = st.radio('Criteria for Split:', ('gini', 'entropy'), key='criterion', help='The criteria for split of node\bgini: based on gini imppurity index\nentropy: based on information gain')
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=1, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                bootstrap = st.radio('Bootstrap Samples:', ('True','False'), key='bootstrap', help='Bootstrapping to obtain samples to build trees.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')

                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tunes hyperparameters
                if st.button('Classify'):
                    from sklearn.ensemble import RandomForestClassifier
                    st.subheader('Random Forest Classifier Results')
                    st.spinner(text='Loading results..')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state, bootstrap=bootstrap)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)
                        
                    # evaluating the model
                    st.write('Accuracy: ',accuracy)
                    st.write('Precision: ',precision)
                    st.write('Recall: ',recall)
                    plot_metrics(model,metrics,X_test,Y_test)



            # AdaBoost 
            if classifier == 'AdaBoost':

                # user help to adaboost
                st.write("""
                AdaBoost is a Supervised Machine Learning algorithm that fits additional copies of the classifier on the same dataset 
                but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on 
                difficult cases.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Boosted:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.')
                learning_rate = st.number_input('Learning Rate:', min_value=0.001, max_value=10.0, step=0.001, key='learning_rate', help='Weight applied to each classifier at each boosting iteration.')           
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tunes hyperparameters
                if st.button('Classify'):
                    from sklearn.ensemble import AdaBoostClassifier
                    st.subheader('AdaBoost Classifier Results')
                    st.spinner('Loading results..')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state, learning_rate=learning_rate)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)
                    
                    # evaluating the model
                    st.write('Accuracy: ',accuracy)
                    st.write('Precision: ',precision)
                    st.write('Recall: ',recall)
                    plot_metrics(model,metrics,X_test,Y_test)



            # XGBoost
            if classifier == 'XGBoost':

                # user help to xgboost
                st.write("""
                XGBoost is a Supervised Machine Learning algorithm that minimize the loss function by adding weak learners 
                using a gradient descent optimization algorithm.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Boosted:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The number of trees in the forest.')              
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=1, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                eta = st.number_input('ETA (Learning Rate)', min_value=0.2, max_value=0.3, step=0.01, key='eta', help='Step size shrinkage used in update to prevents overfitting.')
                gamma = st.number_input('Gamma:', min_value=0.01, max_value=10.0, step=0.01, key='gamma', help='Minimum loss reduction required to make a further partition on a leaf node of the tree.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tunes hyperparameters
                if st.button('Classify'):
                    from xgboost import XGBClassifier
                    st.subheader('XGBClassifier Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, eta=eta, gamma=gamma)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)
                    
                    # evaluating the model
                    st.write('Accuracy: ',accuracy)
                    st.write('Precision: ',precision)
                    st.write('Recall: ',recall)
                    plot_metrics(model,metrics,X_test,Y_test)



            # XGBoost with Random Forests
            if classifier == 'XGBoost with Random Forests':

                # user help to xgboost with random forests
                st.write("""
                XGBoost is a Supervised Machine Learning algorithm that minimize the loss function by adding weak learners 
                using a gradient descent optimization algorithm. Here, the learners are random forests.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Boosted:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The number of estimators in the forest.')              
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=1, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                learning_rate = st.number_input('Learning Rate', min_value=0.01, max_value=10.0, step=0.01, key='learning_rate', help='Step size shrinkage used in update to prevents overfitting.')
                gamma = st.number_input('Gamma:', min_value=0.01, max_value=10.0, step=0.01, key='gamma', help='Minimum loss reduction required to make a further partition on a leaf node of the tree.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                
                # plotting visual metrics
                metrics = st.multiselect('Visual Metrics:',('Confusion Matrix','ROC Curve', 'Precision-Recall Curve'))

                # building model with tunes hyperparameters
                if st.button('Classify'):
                    from xgboost import XGBRFClassifier
                    st.subheader('XGBRFClassifier Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = XGBRFClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, learning_rate=learning_rate, gamma=gamma)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        accuracy = model.score(X_test,Y_test)
                        precision = precision_score(Y_test,Y_pred)
                        recall = recall_score(Y_test,Y_pred)
                        
                    # evaluating the model
                    st.write('Accuracy: ',accuracy)
                    st.write('Precision: ',precision)
                    st.write('Recall: ',recall)
                    plot_metrics(model,metrics,X_test,Y_test)




        # code for regression type of problem
        elif option== 'Regression':

            # importing libraries for model evaluation
            from sklearn.metrics import mean_absolute_error
            from sklearn.metrics import mean_squared_error
            import math

            # options for different types of models
            regressor = st.selectbox('Regression Algorithm:',('Select','Linear Regression','Ridge','Lasso','ElasticNet','Decision Trees','Random Forests','AdaBoost',
            'XGBoost','XGBoost with Random Forests'))

            # Linear Regression
            if regressor == 'Linear Regression':

                # user help to linear regression
                st.write("""
                Linear Regression is a Supervised Mahcine Learning algorithm that predicts the value of a continuous variable by
                fitting a linear equation to the observed data.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                fit_intercept = st.radio('Intercept',('True','False'), key='fit_intercept', help='If set to true, the intercept will be calculated.')
                normalize = st.radio('Normalize Data',('True','False'), key='normalize', help='If set to true, the data will be scaled before fitting.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.linear_model import LinearRegression
                    st.subheader('Linear Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test, Y_pred)
                        mean_squared = math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(X_test, Y_test)

                    # evaluating the model
                    st.write('Mean Absolute Error: ',  mean_absolute)
                    st.write('Root Mean Square Error: ', mean_squared)
                    st.write('R-Squared Error: ', r_squared)
                    


                
            # Ridge Regression
            if regressor == 'Ridge':

                # user help to ridge regression
                st.write("""
                Ridge Regression is a Supervised Mahcine Learning algorithm that predicts the value of a continuous variable by
                fitting a linear equation to the observed data and applies L2 type penalty to prevent overfitting.
                """)

                # tuning the hyperparameters
                st.ssubheader('Model Hyperparameters:')
                alpha = st.number_input('Regularization Strength:', min_value=0.01, max_value=10.0, step=0.01, key='alpha', help='Strength of regularization.')
                fit_intercept = st.radio('Intercept',('True','False'), key='fit_intercept', help='If set to true, the intercept will be calculated.')
                normalize = st.radio('Normalize Data',('True','False'), key='normalize', help='If set to true, the data will be scaled before fitting.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.linear_model import Ridge
                    st.subheader('Ridge Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # Lasso Regression
            if regressor == 'Lasso':

                # user help to lasso regression
                st.write("""
                Lasso Regression is a Supervised Mahcine Learning algorithm that predicts the value of a continuous variable by
                fitting a linear equation to the observed data and applies L1 type penalty to prevent overfitting.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                alpha = st.number_input('Regularization Strength:', min_value=0.01, max_value=10.0, step=0.01, key='alpha', help='Strength of regularization.')
                fit_intercept = st.radio('Intercept',('True','False'), key='fit_intercept', help='If set to true, the intercept will be calculated.')
                normalize = st.radio('Normalize Data',('True','False'), key='normalize', help='If set to true, the data will be scaled before fitting.')
                positive = st.radio('Positive Coefficients',('True','False'),key='positive', help='If set to true, the coefficients are only positive.')
                max_iter = st.number_input('Maximum Iterations:', min_value=100, max_value=10000, step=10, key='max_iter', help='Maximum number of iterations for conjugate gradient solver.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.linear_model import Lasso
                    st.subheader('Lasso Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, positive=positive, max_iter=max_iter)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # ElasticNet Regression
            if regressor == 'ElasticNet':

                # user help to elastic net regression
                st.write("""
                ElasticNet Regression is a Supervised Mahcine Learning algorithm that predicts the value of a continuous variable by
                fitting a linear equation to the observed data and applies both L1 type and L2 type penalties to prevent overfitting.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                alpha = st.number_input('Regularization Strength:', min_value=0.01, max_value=10.0, step=0.01, key='alpha', help='Strength of regularization.')
                l1_ratio = st.slider('L1 Regularization Ratio:', min_value=0.0, max_value=1.0, step=0.05, key='l1_ratio', help='Percentage of L1 type penalty. L2 type will be 1 - L1 type')
                fit_intercept = st.radio('Intercept',('True','False'), key='fit_intercept', help='If set to true, the intercept will be calculated.')
                normalize = st.radio('Normalize Data',('True','False'), key='normalize', help='If set to true, the data will be scaled before fitting.')
                positive = st.radio('Positive Coefficients',('True','False'),key='positive', help='If set to true, the coefficients are only positive.')
                max_iter = st.number_input('Maximum Iterations:', min_value=100, max_value=10000, step=10, key='max_iter', help='Maximum number of iterations for conjugate gradient solver.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.linear_model import ElasticNet
                    st.subheader('ElasticNet Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = ElasticNet(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, positive=positive, max_iter=max_iter, l1_ratio=l1_ratio)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # Decision Trees Regression
            if regressor == 'Decision Trees':

                # user help to decision trees
                st.write("""
                Decision Trees is a Supervised Learning Machine Learning algorithm. The goal is to create a model that predicts the value 
                of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise 
                constant approximation.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                max_depth = st.sidebar.number_input('Maximum Depth of Trees:', min_value=2, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                random_state = st.sidebar.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.trees import DecisionTreeRegressor
                    st.subheader('Decision Tree Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # Random Forests Regression
            if regressor == 'Random Forests':

                # user help to random forests
                st.write("""
                A random forest is a Supervised Machine Learning algorithm that fits a number of decision tree classifiers on various 
                sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Bagged:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The number of trees in the forest.')
                criterion = st.radio('Criteria for Split:', ('mse', 'mae'), key='criterion', help='The criteria for split of node\bgini: based on gini imppurity index\nentropy: based on information gain')
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=2, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                bootstrap = st.radio('Bootstrap Samples:', ('True','False'), key='bootstrap', help='Bootstrapping to obtain samples to build trees.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.ensemble import RandomForestRegressor
                    st.subheader('Random Forests Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = RandomForestRegressor(max_depth=max_depth, random_state=random_state, criterion=criterion, n_estimators=n_estimators, bootstrap=bootstrap)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # AdaBoost
            if regressor == 'AdaBoost':

                # user help to adaboost
                st.write("""
                AdaBoost is a Supervised Machine Learning algorithm that fits additional copies of the regressor on the same dataset but 
                where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent 
                regressors focus more on difficult cases.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Boosted:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.')
                learning_rate = st.number_input('Learning Rate:', min_value=0.001, max_value=10.0, step=0.001, key='learning_rate', help='Weight applied to each classifier at each boosting iteration.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')

                # building model with tuned hyperparameters
                if st.button('Predict'):
                    from sklearn.ensemble import AdaBoostRegressor
                    st.subheader('AdaBoost Regression Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = AdaBoostRegressor(random_state=random_state, n_estimators=n_estimators, learning_rate=learning_rate)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)


                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # XGBoost
            if regressor == 'XGBoost':

                # user help to xgboost
                st.write("""
                XGBoost is a Supervised Machine Learning algorithm that minimize the loss function by adding weak learners 
                using a gradient descent optimization algorithm.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Boosted:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The number of trees in the forest.')              
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=1, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                eta = st.number_input('ETA (Learning Rate)', min_value=0.2, max_value=0.3, step=0.01, key='eta', help='Step size shrinkage used in update to prevents overfitting.')
                gamma = st.number_input('Gamma:', min_value=0.01, max_value=10.0, step=0.01, key='gamma', help='Minimum loss reduction required to make a further partition on a leaf node of the tree.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                

                # building model with tunes hyperparameters
                if st.button('Predict'):
                    from xgboost import XGBRegressor
                    st.subheader('XGBRegressor Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, eta=eta, gamma=gamma)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)

                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)



            # XGBoost with Random Forests
            if regressor == 'XGBoost with Random Forests':

                # user help to xgboost with random forests
                st.write("""
                XGBoost is a Supervised Machine Learning algorithm that minimize the loss function by adding weak learners 
                using a gradient descent optimization algorithm. Here, the learners are random forests.
                """)

                # tuning the hyperparameters
                st.subheader('Model Hyperparameters:')
                n_estimators = st.number_input('Number of Trees Boosted:', min_value=100, max_value=5000, step=10, key='n_estimators', help='The number of estimators in the forest.')              
                max_depth = st.number_input('Maximum Depth of Trees:', min_value=1, max_value=20, step=1, key='max_depth', help='The maximum depth of trees used to predict the outcome.')
                learning_rate = st.number_input('Learning Rate', min_value=0.01, max_value=10.0, step=0.01, key='learning_rate', help='Step size shrinkage used in update to prevents overfitting.')
                gamma = st.number_input('Gamma:', min_value=0.01, max_value=10.0, step=0.01, key='gamma', help='Minimum loss reduction required to make a further partition on a leaf node of the tree.')
                random_state = st.number_input('Random State:', min_value=0, step=1, key='random_state', help='Fix the random state to obtain consistent results.')
                        
                # building model with tunes hyperparameters
                if st.button('Predict'):
                    from xgboost import XGBRFRegressor
                    st.subheader('XGBRFRegressor Results')
                    st.spinner('Loading results...')

                    # printing and exceuting the code to train the model and to evalaute metrics
                    with st.echo():
                        model = XGBRFRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, learning_rate=learning_rate, gamma=gamma)
                        model.fit(X_train,Y_train)
                        Y_pred = model.predict(X_test)
                        mean_absolute = mean_absolute_error(Y_test,Y_pred)
                        mean_squared =  math.sqrt(mean_squared_error(Y_test, Y_pred))
                        r_squared = model.score(Y_test,Y_pred)


                    # evaluating the model
                    st.write('Mean Absolute Error: ', mean_absolute)
                    st.write('Root Mean Square Error: ',mean_squared)
                    st.write('R-Squared Error: ', r_squared)

    else:
        st.write('Please upload the dataset through the main page!')
            