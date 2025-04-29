import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D  # for 3d plotting

app = Flask(__name__)

# Load dataset
DATA_PATH = "lung_cancer_data.csv"
df = pd.read_csv(DATA_PATH)

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

df_processed, label_encoders = preprocess_data(df)

# Prepare features and target for prediction
# Assuming 'Stage' is the target variable
X = df_processed.drop(columns=['Patient_ID', 'Stage'])
y = df_processed['Stage']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model for stage prediction
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Routes

@app.route('/')
def home():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Prepare categorical options for dropdowns
    categorical_features = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment', 'Ethnicity', 'Insurance_Type']
    categorical_options = {}
    for feature in categorical_features:
        if feature in df.columns:
            categorical_options[feature] = sorted(df[feature].dropna().unique().tolist())

    # Prepare numerical ranges for sliders
    numerical_features = [col for col in X.columns if col not in categorical_features and not col.startswith('Comorbidity') and col not in ['Family_History', 'Performance_Status']]
    numerical_ranges = {}
    for feature in numerical_features:
        if feature in df.columns:
            numerical_ranges[feature] = {
                'min': float(df[feature].min()),
                'max': float(df[feature].max())
            }

    if request.method == 'POST':
        # Extract form data
        input_data = {}
        for feature in X.columns:
            val = request.form.get(feature)
            if val is None:
                return "Missing input for " + feature, 400
            input_data[feature] = val

        # Convert input data to dataframe
        input_df = pd.DataFrame([input_data])

        # Convert types and encode categorical features
        yes_no_columns = [col for col in input_df.columns if col.startswith('Comorbidity') or col in ['Family_History', 'Performance_Status']]
        for col in input_df.columns:
            if col in yes_no_columns:
                # Map Yes/No to 1/0
                input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})
            elif col in label_encoders:
                le = label_encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    # If unseen label, add it to encoder classes
                    le.classes_ = np.append(le.classes_, input_df[col].iloc[0])
                    input_df[col] = le.transform(input_df[col].astype(str))
            else:
                # Convert to float
                input_df[col] = pd.to_numeric(input_df[col])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict stage
        pred = model.predict(input_scaled)[0]
        # Decode predicted stage
        stage = label_encoders['Stage'].inverse_transform([pred])[0]

        return render_template('predict.html', prediction=stage, input_data=request.form, features=X.columns, categorical_options=categorical_options, numerical_ranges=numerical_ranges)
    else:
        return render_template('predict.html', prediction=None, features=X.columns, categorical_options=categorical_options, numerical_ranges=numerical_ranges)

@app.route('/clusters')
def clusters_view():
    return render_template('clusters.html')

@app.route('/clusters/image')
def clusters_image():
    # Add cluster labels to original df
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters

    # Generate a simple cluster plot (2D PCA or first two features)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df_clustered.iloc[:,1], y=df_clustered.iloc[:,4], hue=df_clustered['Cluster'], palette='Set1')
    plt.title('Patient Clusters')
    plt.xlabel(df_clustered.columns[1])
    plt.ylabel(df_clustered.columns[4])
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)

    return send_file(img, mimetype='image/png')

@app.route('/visualize')
def visualize():
    # For simplicity, show a page with links to different plots
    return render_template('visualize.html')

@app.route('/visualize/plot/<plot_name>')
def visualize_plot(plot_name):
    plt.figure(figsize=(8,6))
    if plot_name == 'hist_age':
        sns.histplot(df['Age'], kde=True)
        plt.title('Age Distribution')
    elif plot_name == 'box_tumor_size':
        sns.boxplot(x=df['Stage'], y=df['Tumor_Size_mm'])
        plt.title('Tumor Size by Stage')
    elif plot_name == '3d_scatter':
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['Age'], df['Tumor_Size_mm'], df['Survival_Months'], c='b', marker='o')
        ax.set_xlabel('Age')
        ax.set_ylabel('Tumor Size (mm)')
        ax.set_zlabel('Survival Months')
        plt.title('3D Scatter Plot')
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return send_file(img, mimetype='image/png')
    elif plot_name == 'correlation_heatmap':
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap')
    elif plot_name == 'pairplot':
        sns.pairplot(df.select_dtypes(include=['float64', 'int64']).drop(columns=['Patient_ID']))
        plt.title('Pairplot of Numeric Features')
    elif plot_name == 'violin_survival_stage':
        sns.violinplot(x=df['Stage'], y=df['Survival_Months'])
        plt.title('Survival Months by Stage')
    elif plot_name == 'count_gender':
        sns.countplot(x=df['Gender'])
        plt.title('Count of Patients by Gender')
    elif plot_name == 'count_smoking':
        sns.countplot(x=df['Smoking_History'])
        plt.title('Count of Patients by Smoking History')
    elif plot_name == '3d_scatter_alt':
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['Blood_Pressure_Systolic'], df['Blood_Pressure_Diastolic'], df['Hemoglobin_Level'], c='r', marker='^')
        ax.set_xlabel('Systolic BP')
        ax.set_ylabel('Diastolic BP')
        ax.set_zlabel('Hemoglobin Level')
        plt.title('3D Scatter Plot (BP and Hemoglobin)')
    else:
        return "Plot not found", 404

    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
