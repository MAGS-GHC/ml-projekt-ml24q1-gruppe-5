import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# This if used for google collab when trying to run the test url again, it runs into a error without this line
tf.data.experimental.enable_debug_mode()
# Loading dataset
data = pd.read_csv('dataset_B_05_2020.csv')
test_data = data.copy()
# Extract target variable
y = data['status']

# Dropping our non-numeric and target columns
exclude_columns = ['url', 'status']
X_numeric = data.drop(columns=exclude_columns).select_dtypes(include=[np.number])

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Converting status column to 0 and 1
y_train = y_train.map({'legitimate': 0, 'phishing': 1})
y_test = y_test.map({'legitimate': 0, 'phishing': 1})

# Data scaling
scaler = StandardScaler()
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

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

# Save the model
# This could be useful for saving the model, instead of training it on every run
#model.save('phishing_detection_model.h5')


def preprocess_input_row(input_row, categorical_columns=[]):

    # Dropping our non-numeric and target columns
    input_features = input_row.drop(columns=['url', 'status'])

    input_features = pd.get_dummies(input_features, columns=categorical_columns, drop_first=True)

    # Ensure that the input data has the same structure as the training data
    return input_features


input_row = data.iloc[4]  # Selecting the 4th row due to this is the row we'd like to test
categorical_columns = ['status']  # Addressing the column we'd like to predict
input_features_scaled = preprocess_input_row(input_row, categorical_columns)

# Make predictions using the trained model
input_features_scaled = input_features_scaled.values.astype('float32').reshape(1, -1)[:,:87]  # Take only the first 87 features
prediction_proba = model.predict(input_features_scaled)
prediction_binary = (prediction_proba > 0.5).astype(int)
if prediction_binary[0] == 0:
    print(f"The URL '{input_row['url']}' is predicted as Legitimate.")
else:
    print(f"The URL '{input_row['url']}' is predicted as Phishing.")




from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

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

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()