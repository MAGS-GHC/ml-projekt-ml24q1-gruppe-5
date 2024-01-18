import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt

# This if used for google collab when trying to run the test url again, it runs into a error without this line
tf.data.experimental.enable_debug_mode()

# Creating global variabels
data = None
model = None


def train_and_evaluate():

    global data
    global model
    # Loading dataset
    data = pd.read_csv('dataset_B_05_2020.csv')
    test_data = data.copy()
    # Extract target variable
    y = data['status']

    # Dropping our non-numeric and target columns
    exclude_columns = ['url', 'status']
    X_numeric = data.drop(columns=exclude_columns).select_dtypes(include=[np.number])

    # Seperating data into test and training
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # Converting status column to 0 and 1
    y_train = y_train.map({'legitimate': 0, 'phishing': 1})
    y_test = y_test.map({'legitimate': 0, 'phishing': 1})

    # Data scaling
    scaler = StandardScaler()
    # transform anvender normaliseringen ved at trække middelværdien fra og dele med standardafvigelsen
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model building
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model training
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Model evaluation
    y_pred = model.predict(X_test_scaled)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Save the model
    model.save('phishing_detection_model.keras')
    
    print("Accuracy:", accuracy_score(y_test, y_pred_binary))
    print("Classification Report:\n", classification_report(y_test, y_pred_binary))
    accuracy_label.config(text=f'Classification Report:\n {classification_report(y_test, y_pred_binary)}')
    modelevaluation(X_test_scaled=X_test_scaled, y_test=y_test)



def preprocess_input_row(input_row, categorical_columns=[]):

    # Dropping our non-numeric and target columns
    input_features = input_row.drop(columns=['url', 'status'])

    input_features = pd.get_dummies(input_features, columns=categorical_columns, drop_first=True)

    # Ensure that the input data has the same structure as the training data
    return input_features


def predict_url_status():
    global data

    try:
        loaded_model = tf.keras.models.load_model('phishing_detection_model.keras')
    
    except Exception:
        print(f"Could not find the loaded data!")
        result_label.config(text="Couldnt find the trained model, try to train the model again")
        return
    
    # Retrieving the user
    test_url = test_url_entry.get()

    if data is None:
        data = pd.read_csv('dataset_B_05_2020.csv')

    input_row = data.iloc[int(test_url)]  # Selecting the 4th row due to this is the row we'd like to test
    categorical_columns = ['status']  # Addressing the column we'd like to predict
    input_features_scaled = preprocess_input_row(input_row, categorical_columns)
    # Make predictions using the trained model
    input_features_scaled = input_features_scaled.values.astype('float32').reshape(1, -1)[:,:87]  # Take only the first 87 features
    prediction_proba = loaded_model.predict(input_features_scaled)

    prediction_binary = (prediction_proba > 0.5).astype(int)
    if prediction_binary[0] == 0:
        result_label.config(text=f"The URL '{input_row['url']}' is predicted as Legitimate.")
    else:
        result_label.config(text=f"The URL '{input_row['url']}' is predicted as Phishing.")






def modelevaluation(X_test_scaled, y_test):

    # Model evaluation
    y_pred_proba = model.predict(X_test_scaled)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Legitimate', 'Phishing']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    plt.show()



# Creating our UI using TKinter
    # Create the main window
root = tk.Tk()
root.title('Phishing Link Detection UI')

# Create and configure the notebook
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Create tab1
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Model Evaluation')

# Add components to the first tab
train_button = ttk.Button(tab1, text='Train and Evaluate', command=train_and_evaluate)
train_button.pack(pady=10)

accuracy_label = ttk.Label(tab1, text='')
accuracy_label.pack()

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Test Single URL')

test_url_label = ttk.Label(tab2, text='Enter a row number from the dataset')
test_url_label.pack(pady=10)

test_url_entry = ttk.Entry(tab2, width=50)
test_url_entry.pack()

test_button = ttk.Button(tab2, text='Test URL', command=predict_url_status)
test_button.pack(pady=10)

result_label = ttk.Label(tab2, text='')
result_label.pack()


root.mainloop()
