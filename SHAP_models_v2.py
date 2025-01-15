import streamlit as st
import pandas as pd
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, classification_report, r2_score,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder  # <-- Correct import
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("SHAP Analysis (version 2)")

    # Upload CSV or Excel file
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        # Display uploaded data
        st.subheader("Uploaded Data")
        st.write(df.head())

        # Select target variable
        target_variable = st.selectbox("Select the target variable", df.columns)

        # Provide instructional messages
        st.info(
            f"The selected target variable '{target_variable}' is crucial for determining the type of model. "
            "Consider the nature of your target variable and choose the model type accordingly. "
            "For example, if you are predicting categories or labels (e.g., 'Red', 'Blue', 'Green'), choose a classifier model. "
            "If you are predicting numeric values (e.g., 'Sales', 'Temperature', 'Price'), choose a regressor model using the box below."
        )

        # Allow the user to choose the model type (Classifier or Regressor)
        model_type = st.radio("Select Model Type", ["Classifier", "Regressor"])

        # Select feature variables
        feature_variables = st.multiselect("Select feature variables", df.columns)

        # Set adjustable parameters
        random_state = st.slider("Random State", 0, 100, 42)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.01)

        # Handle missing values
        missing_values_option = st.selectbox("Handle Missing Values:", ["Exclude", "Impute (Mean)"])
        if missing_values_option == "Impute (Mean)":
            # Impute missing values with mean
            df = df.fillna(df.mean())
        elif missing_values_option == "Exclude":
            # Exclude rows with missing values
            df = df.dropna()

        # Allow user to choose cross-validation folds (radio)
        st.subheader("Cross-Validation")
        cv_folds = st.radio(
            "Select Cross-Validation Folds (optional)", 
            [None, 2, 5, 10], 
            index=0,
            help="Choose 2-, 5-, or 10-fold cross-validation. Select 'None' to skip."
        )

        if st.button("Generate SHAP Plots"):
            # Train model
            if model_type == "Classifier":
                # Encode labels
                label_encoder = LabelEncoder()
                df[target_variable] = label_encoder.fit_transform(df[target_variable])

                # Check if we have more than one unique class
                unique_labels = df[target_variable].unique()
                st.write("Unique labels in target:", unique_labels)
                if len(unique_labels) <= 1:
                    st.error("Your target variable has only one unique value. SHAP cannot run multi-class analysis.")
                    st.stop()

                model = train_classification_model(df, target_variable, feature_variables, random_state, test_size)
            else:
                model = train_regression_model(df, target_variable, feature_variables, random_state, test_size)

            # Perform cross validation if user selected an option (2, 5, 10)
            if cv_folds:
                perform_cross_validation(df, target_variable, feature_variables, model_type, cv_folds)

            # ===== SHAP Analysis =====
            shap_values, expected_value, feature_names = shap_analysis(model, df[feature_variables])

            # Display results
            if model_type == "Classifier":
                display_classification_results(model, df, target_variable, feature_variables)
                # Show SHAP per-class summary plots (only meaningful if multi-class)
                show_shap_per_class(shap_values, df[feature_variables])
            else:
                display_regression_results(model, df, target_variable, feature_variables)
                # Display SHAP Dependence Plots only for regressor models
                st.subheader("SHAP Dependence Plots")
                for feature in feature_names:
                    shap.dependence_plot(feature, shap_values, df[feature_variables], show=False)
                    st.pyplot()

            # Single summary plot (aggregated or average)
            st.subheader("SHAP Summary Plot")
            plot_summary(shap_values, feature_names)

            # Feature Importances
            st.subheader("Feature Importances")
            plot_feature_importances(model, feature_variables)

def load_data(uploaded_file):
    """Load CSV or Excel data into a Pandas DataFrame."""
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('xls', 'xlsx')):
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("Invalid file format. Please upload a CSV or Excel file.")
        st.stop()
    return df

def train_classification_model(df, target_variable, feature_variables, random_state, test_size):
    X = df[feature_variables]
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_regression_model(df, target_variable, feature_variables, random_state, test_size):
    X = df[feature_variables]
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

def perform_cross_validation(df, target_variable, feature_variables, model_type, cv_folds):
    """Perform cross validation and display per-fold results + mean ± std."""
    X = df[feature_variables]
    y = df[target_variable]

    st.write(f"Performing {cv_folds}-fold cross-validation...")

    if model_type == "Classifier":
        temp_model = DecisionTreeClassifier()
        cv_scores = cross_val_score(temp_model, X, y, cv=cv_folds, scoring='accuracy')
        # Display per-fold table
        cv_df = pd.DataFrame({
            'Fold': np.arange(1, cv_folds + 1),
            'Accuracy': cv_scores
        })
        st.table(cv_df)
        # Display summary (mean ± std)
        st.write(
            f"**Mean Accuracy:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )
    else:  # Regressor
        temp_model = DecisionTreeRegressor()
        cv_scores = cross_val_score(temp_model, X, y, cv=cv_folds, scoring='r2')
        # Display per-fold table
        cv_df = pd.DataFrame({
            'Fold': np.arange(1, cv_folds + 1),
            'R²': cv_scores
        })
        st.table(cv_df)
        # Display summary (mean ± std)
        st.write(
            f"**Mean R²:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

def shap_analysis(model, features):
    """Return SHAP values, expected value, and feature names."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    return shap_values, explainer.expected_value, features.columns

def show_shap_per_class(shap_values, features):
    """Display SHAP summary plot per class (only relevant for multi-class)."""
    # For binary classification, shap_values is often an array of shape (n, features).
    # For multi-class, shap_values is a list of arrays [class0, class1, ...].
    if isinstance(shap_values, list) and len(shap_values) > 1:
        st.subheader("Class-Specific SHAP Summary Plots")
        for class_idx, shap_val_for_class in enumerate(shap_values):
            st.write(f"**Class {class_idx}**")
            shap.summary_plot(shap_val_for_class, features, show=False)
            st.pyplot()

def plot_summary(shap_values, feature_names):
    """Aggregate SHAP summary plot with safe handling for single-class vs. multi-class."""
    # If shap_values is a list of arrays, we have multi-class or an edge binary scenario.
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            # Single-class or edge-case binary scenario
            shap.summary_plot(shap_values[0], feature_names=feature_names, show=False)
        else:
            # True multi-class
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    else:
        # Normal single array (binary classification or regression)
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)

    plt.gcf().set_size_inches(10, 6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot()

def mean_absolute_percentage_error(y_true, y_pred):
    """Compute mean absolute percentage error, ignoring zero-values in y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return np.nan  # If all y_true are zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def display_classification_results(model, df, target_variable, feature_variables):
    """Display classification metrics: classification report, confusion matrix, additional metrics."""
    X_test = df[feature_variables]
    y_test = df[target_variable]
    y_pred = model.predict(X_test)

    st.subheader("Classification Results")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Additional metrics (precision, recall, f1 -- macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    st.write(f"**Macro Precision:** {precision:.4f}")
    st.write(f"**Macro Recall:** {recall:.4f}")
    st.write(f"**Macro F1-Score:** {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm))

def display_regression_results(model, df, target_variable, feature_variables):
    """Display regression metrics: MAE, MSE, RMSE, MAPE, R²."""
    X_test = df[feature_variables]
    y_test = df[target_variable]
    y_pred = model.predict(X_test)

    st.subheader("Regression Results")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("Mean Absolute Error (MAE):", round(mae, 2))
    st.write("Mean Squared Error (MSE):", round(mse, 2))
    st.write("Root Mean Squared Error (RMSE):", round(rmse, 2))
    st.write("Mean Absolute Percentage Error (MAPE):", round(mape, 2))
    st.write("R² Score:", round(r2, 4))

def plot_feature_importances(model, feature_names):
    """Display feature importance table and bar plot."""
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Show the dataframe of feature importances
        st.write("Feature Importance Values:")
        st.dataframe(importance_df)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        if isinstance(model, DecisionTreeClassifier):
            plt.title('Feature Importances (Classifier)')
        else:
            plt.title('Feature Importances (Regressor)')

        plt.xticks(rotation=45, ha='right')
        st.pyplot()
    else:
        st.write("No feature_importances_ attribute found for this model.")

if __name__ == "__main__":
    main()
