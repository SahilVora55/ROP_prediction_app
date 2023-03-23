import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from tpot import TPOTRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Define the app
def app():
    # Set the page title and description
    st.set_page_config(page_title='ROP Prediction App', page_icon=':chart_with_upwards_trend:', layout='wide')
    st.title('ROP Prediction App')
    st.markdown('This app predicts the Rate of Penetration (ROP) using real-time drilling data.')
    
    st.write("-----")
    st.write("")
    st.write("")

    if 'df_pred' not in st.session_state:
        st.session_state['df_pred'] = pd.DataFrame()

    # Upload the drilling data
    st.subheader('Make a Machine Learning model to predict the Rate of Penetration (ROP).')
    uploaded_file = st.file_uploader('Upload your drilling data (CSV file)', type=['csv'])
    if uploaded_file is not None:
        # Load the drilling data into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the drilling data
        st.write('Drilling Data:')
        st.dataframe(df, height=200)

        # Select the prediction model
        st.subheader('Selecting Prediction Model and Features')
        model_name = st.selectbox('Select the prediction model', ['Random Forest Regression', 'Gradient Boosting Regression', 'XGBoost Regression', 'Decision Tree Regression'])

        col1, col2 = st.columns(2)

        with col1:
            # Select the Rate of Penetration column
            target_column = st.selectbox('Select the Rate of Penetration column', list(df.columns), key='target_column')
        with col2:
            # Select the input features for the ROP prediction model
            selected_features = st.multiselect('Select the input features', list(df.drop('Rate of Penetration m/h',axis=1).columns))

        # Set the model parameters based on the selected model
        st.write(f'<h3 style="font-size:16px;">Adjust model parameters for {model_name}</h3>', unsafe_allow_html=True)

        if model_name == 'Random Forest Regression':
            # Create three columns with equal width
            col1, col2, col3 = st.columns(3)
            # Add sliders to each column
            with col1:
                n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)
            with col2:
                max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)
            with col3:
                min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)
            # Create a dictionary with the slider values
            model_params = {'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split}
            model = RandomForestRegressor(**model_params)

        elif model_name == 'Gradient Boosting Regression':
            # Create three columns with equal width
            col1, col2, col3 = st.columns(3)

            # Add sliders to each column
            with col1:
                n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)

            with col2:
                max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

            with col3:
                min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)

            # Create a dictionary with the slider values
            model_params = {'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split}

            model = GradientBoostingRegressor(**model_params)
        elif model_name == 'XGBoost Regression':
            # Create three columns with equal width
            col1, col2, col3 = st.columns(3)

            # Add sliders to each column
            with col1:
                n_estimators = st.slider('Number of Trees', min_value=10, max_value=500, value=100, step=10)

            with col2:
                max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

            with col3:
                learning_rate = st.slider('Learning Rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01)

            # Create a dictionary with the slider values
            model_params = {'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate}

            model = XGBRegressor(**model_params)
        elif model_name == 'Decision Tree Regression':
            # Create two columns with equal width
            col1, col2 = st.columns(2)

            # Add sliders to each column
            with col1:
                max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=10)

            with col2:
                min_samples_split = st.slider('Min Samples Split', min_value=2, max_value=20, value=5)

            # Create a dictionary with the slider values
            model_params = {'max_depth': max_depth,
                            'min_samples_split': min_samples_split}

            model = DecisionTreeRegressor(**model_params)
#        else:
#            # Create two columns with equal width
#            col1, col2 = st.columns(2)
#
#            # Add sliders to each column
#            with col1:
#                generations = st.slider('Number of generations', min_value=5, max_value=50, value=10, step=5)
#
#            with col2:
#                population_size = st.slider('Population size', min_value=10, max_value=100, value=50, step=10)
#
#            # Create a dictionary with the slider values
#            model_params = {'generations': generations,
#                            'population_size': population_size}
#
#            model = TPOTRegressor(generations=model_params['generations'], population_size=model_params['population_size'], verbosity=0, random_state=42)


        # Select test size
        st.write('<h3 style="font-size:16px;">Train-test split</h3>', unsafe_allow_html=True)
        test_size = st.slider('Select Test Size', min_value=0.1, max_value=0.5, value=0.2, step=0.01)

        # Make the ROP prediction
        button_html = '<button style="background-color: lightgreen; color: white; font-size: 16px; padding: 0.5em 1em; border-radius: 5px; border: none;">Make Prediction</button>'
        if st.button('Make Prediction',use_container_width=True):
            st.text('Prediction in progress...')  # display message while prediction is happening
            # Check if the target column exists in the input data
            if target_column not in df.columns:
                st.warning(f'The input data does not have a column named "{target_column}". Please upload valid drilling data.')
            else:
                # Preprocess the input data
                X = df[selected_features]
                y = df[target_column]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Fit the model to the data
                model.fit(X_train, y_train)

                # Predict the ROP
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                st.success('Prediction successful!')  # display success message after prediction

                # Save the model to a file
                filename = 'model.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(model, file)

                # Load the saved model
                with open(filename, 'rb') as file:
                    model = pickle.load(file)

                # Encode the model file to base64
                with open(filename, 'rb') as f:
                    bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()

                # Create a download link for the model file
                href = f'<a href="data:file/model.pkl;base64,{b64}" download="model.pkl">Download Trained Model (.pkl)</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Calculate MAE & R2 Score
                MAE_train = mean_absolute_error(y_train, y_pred_train)
                MAE_test = mean_absolute_error(y_test, y_pred_test)
                R2_train = r2_score(y_train, y_pred_train)
                R2_test = r2_score(y_test, y_pred_test)

                st.subheader('Result')
                col1, col2, col3 = st.columns([1,1,2])
                with col1:    
                    st.write('for training data\n- R2-score: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>\n- MAE: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>'.format(R2_train, MAE_train), unsafe_allow_html=True)
                with col2:    
                    st.write('for testing data\n- R2-score: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>\n- MAE: <span style="color:#007D5C;font-weight:bold">{:.3f}</span>'.format(R2_test, MAE_test), unsafe_allow_html=True)
                with col3:
                    # Display the ROP prediction
                       
                    df_pred1 = pd.DataFrame({'ROP_actual':y_test,'ROP_pred': y_pred_test})
                    X_test['ROP_actual'] = df_pred1['ROP_actual']
                    X_test['ROP_pred'] = df_pred1['ROP_pred']
                    st.session_state['df_pred'] = pd.concat([st.session_state['df_pred'], X_test], axis=0)

                    # Add a download button to download the dataframe as a CSV file
                    csv = st.session_state['df_pred'].to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # some strings
                    href = f'<a href="data:file/csv;base64,{b64}" download="st.session_state[\'df_pred\'].csv">Download predicted ROP data</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    # Display the dataframe in Streamlit
                    st.write('ROP Predicted:')
                    st.dataframe(st.session_state['df_pred'][['ROP_actual','ROP_pred']], height=200,width=400)

    # Add a blank line between the buttons
    st.write("")
    st.write("")
    st.write("")
    st.write("________")

    # Create a file uploader widget
    st.subheader('Calculate ROP using created machine learning model.')
    model_file = st.file_uploader("Upload a saved ML model (pkl)", type=["pkl"])

    # If a file has been uploaded, load the model from the file
    if model_file is not None:
        with model_file:
            model = pickle.load(model_file)

        # Display the loaded model
        if 'model' in locals():
            st.write("<span style='color:green; font-weight:bold'>Model loaded successfully!</span>", unsafe_allow_html=True)
            st.write(model)

        # Get the list of column names
        columns = st.session_state['df_pred'].drop(['ROP_actual','ROP_pred'],axis=1).columns.tolist()

        # Create a row of input boxes using beta_columns
        input_cols = st.columns(min(len(columns), 5))
        input_array = np.zeros(len(columns))
        for i, input_col in enumerate(input_cols):
            input_value = input_col.number_input(label=columns[i], step=0.1, value=0.0, min_value=0.0, max_value=1000000.0, key=columns[i])
            input_array[i] = input_value
            
        # Create additional rows of input boxes if there are more than 5 columns
        if len(columns) > 5:
            for j in range(5, len(columns), 5):
                input_cols = st.columns(min(len(columns)-j, 5))
                for i, input_col in enumerate(input_cols):
                    input_value = input_col.number_input(label=columns[j+i], step=0.1, value=0.0, min_value=0.0, max_value=1000000.0, key=columns[j+i])
                    input_array[j+i] = input_value

        # Define colors and font sizes
        HIGHLIGHT_COLOR = '#22c1c3'
        HEADER_FONT_SIZE = '20px'
        RESULT_FONT_SIZE = '36px'

        if st.button('Calculate ROP'):
            input_array = input_array.reshape(1, -1)
            rop = model.predict(input_array)
            st.success('Calculated successful!')
            # Format the output message
            result_text = f"Calculated Rate of Penetration (ROP): {rop[0]:.2f} ft/hr"
            result_html = f'<div style="font-size:{RESULT_FONT_SIZE}; color:{HIGHLIGHT_COLOR};">{result_text}</div>'
            st.markdown(result_html, unsafe_allow_html=True)


    # Add a blank line between the buttons
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("__________")

    col1, col2 = st.columns(2)
    with col1:
        if st.button('HELP',use_container_width=True):
            st.write('**Welcome to the ROP Prediction App!**')
            st.write('This app helps you predict the **Rate of Penetration (ROP)** using **real-time drilling data**. Here are a few guidelines to use this app:')
            st.write('1. Upload your **drilling data** in **CSV** format using the file uploader.')
            st.write('2. Select the **prediction model** from the sidebar.')
            st.write('3. Select the **Rate of Penetration (ROP)** column that you want to predict.')
            st.write('4. Select the **input features** that you want to use for the prediction.')
            st.write('5. Split your data into **training** and **testing** sets.')
            st.write('6. Adjust the **model parameters** as per your requirements.')
            st.write('7. Click on the **"Predict"** button to see the **predicted ROP values**.')
            st.write('Note: This app uses **Random Forest Regression, Gradient Boosting Regression, XGBoost Regression, Decision Tree Regression,** and **TPOT Regression** to predict ROP value.')
    with col2:
        # Add a contact us button
        email_icon = Image.open('email.png')
        if st.button('Contact Us',use_container_width=True):
            st.image(email_icon,width=150)
            st.write('Please email us at <span style="font-size:20px">sahilvoraa@gmail.com</span>', unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()
