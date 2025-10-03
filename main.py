import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Step 1: Load dataset
# -------------------------
if os.path.exists("heart.csv"):
    df = pd.read_csv("heart.csv")
    st.sidebar.success("Loaded dataset: heart.csv")
else:
    st.sidebar.warning("No heart.csv found. Using synthetic dataset.")
    np.random.seed(42)
    num_samples = 1000
    data = {
        'age': np.random.randint(20, 80, num_samples),
        'sex': np.random.randint(0, 2, num_samples),
        'chest_pain_type': np.random.randint(0, 4, num_samples),
        'resting_bp': np.random.randint(90, 200, num_samples),
        'cholesterol': np.random.randint(100, 300, num_samples),
        'fasting_blood_sugar': np.random.randint(0, 2, num_samples),
        'max_heart_rate': np.random.randint(60, 200, num_samples),
        'exercise_angina': np.random.randint(0, 2, num_samples),
    }
    diseases = ((data['age'] > 50) & (data['cholesterol'] > 200) & 
                (data['resting_bp'] > 140) | (data['exercise_angina'] == 1)).astype(int)
    diseases = diseases + np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
    data['disease'] = np.clip(diseases, 0, 1)
    df = pd.DataFrame(data)

# -------------------------
# Step 2: Prepare data
# -------------------------
X = df.drop('disease', axis=1).values.astype(np.float32)
y = df['disease'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------
# Step 3: Define model
# -------------------------
class DiseasePredictor(nn.Module):
    def __init__(self, input_size):
        super(DiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# -------------------------
# Streamlit UI
# -------------------------
st.title("🩺 Disease Diagnosis Prediction Dashboard")
st.markdown("""
This dashboard trains a neural network on heart disease data,
evaluates its performance, and predicts for new patients.
""")

# Sidebar
st.sidebar.header("⚙️ Model Training Parameters")
epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50, 10)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
train_button = st.sidebar.button("Train Model")

# -------------------------
# Step 4: Train Model
# -------------------------
@st.cache_resource
def train_model(epochs, lr):
    model = DiseasePredictor(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))

    # Evaluation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            y_true.extend(labels.squeeze().tolist())
            y_pred.extend(predicted.squeeze().tolist())
            y_prob.extend(outputs.squeeze().tolist())

    acc = accuracy_score(y_true, y_pred)
    return model, losses, acc, y_true, y_pred, y_prob

if train_button:
    with st.spinner("Training the model..."):
        model, losses, acc, y_true, y_pred, y_prob = train_model(epochs, learning_rate)
        st.session_state.update({
            'model': model,
            'losses': losses,
            'accuracy': acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        })
    st.success("✅ Model trained successfully!")

# -------------------------
# Step 5: Results
# -------------------------
if 'model' in st.session_state:
    st.header("📊 Training Results")
    st.write(f"**Test Accuracy:** {st.session_state['accuracy']:.4f}")

    # Loss curve
    st.subheader("Loss Curve")
    fig, ax = plt.subplots()
    ax.plot(st.session_state['losses'])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(st.session_state['y_true'], st.session_state['y_pred'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(st.session_state['y_true'], st.session_state['y_prob'])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='orange', lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color='navy', linestyle="--")
    ax.legend()
    st.pyplot(fig)

# -------------------------
# Step 6: New Patient Prediction
# -------------------------
st.header("🧑‍⚕️ Predict for a New Patient")
with st.form("prediction_form"):
    age = st.number_input("Age", 0, 120, 55)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    resting_bp = st.number_input("Resting BP", 0, 300, 140)
    chol = st.number_input("Cholesterol", 0, 600, 200)
    sugar = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    hr = st.number_input("Max Heart Rate", 0, 300, 150)
    angina = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    submit = st.form_submit_button("Predict")

if submit:
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first!")
    else:
        patient = np.array([[age, sex, chest_pain, resting_bp, chol, sugar, hr, angina]], dtype=np.float32)
        patient_tensor = torch.tensor(patient)
        st.session_state['model'].eval()
        with torch.no_grad():
            prob = st.session_state['model'](patient_tensor).item()
        st.write(f"**Disease Probability:** {prob:.4f}")
        st.write(f"**Prediction:** {'Yes' if prob > 0.5 else 'No'}")
