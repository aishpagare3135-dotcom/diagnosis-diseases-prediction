# diagnosis-diseases-prediction
This Streamlit app loads a heart disease dataset (or generates synthetic data if unavailable), trains a neural network to predict disease, shows loss, accuracy, confusion matrix and ROC plots, and lets you input a new patient’s features to predict disease probability in an interactive dashboard.
# 🩺 Disease Diagnosis Prediction Dashboard

## Overview
This Streamlit application provides an interactive dashboard for training a neural network model to predict heart disease based on patient features. It includes data loading (real or synthetic), model training with customizable parameters, performance evaluation (loss curve, confusion matrix, ROC curve), and real-time predictions for new patients.

The model uses a simple feedforward neural network implemented in PyTorch, trained on features like age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, maximum heart rate, and exercise angina.

## Features
- **Data Handling**: Loads from `heart.csv` or generates synthetic data.
- **Model Training**: Adjustable epochs and learning rate via sidebar.
- **Visualization**: Plots for training loss, confusion matrix, and ROC curve.
- **Prediction Interface**: Form for inputting patient data and getting probability-based predictions.
- **Caching**: Uses `@st.cache_resource` for efficient model training.

## Prerequisites
- Python 3.8+ (tested with 3.12)
- Streamlit for the web interface
- PyTorch for the neural network
- Scikit-learn for data splitting and metrics
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualizations

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/aishwaryapagare/disease-diagnosis-dashboard.git
   cd disease-diagnosis-dashboard
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Place your `heart.csv` dataset in the root directory for real data loading. If absent, synthetic data will be used.

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
   This will open the dashboard in your browser (default: http://localhost:8501).

2. In the sidebar:
   - Adjust training parameters (epochs, learning rate).
   - Click **Train Model** to start training.

3. View results:
   - Training loss curve.
   - Confusion matrix and accuracy.
   - ROC curve with AUC score.

4. Predict for a new patient:
   - Fill the form under "Predict for a New Patient".
   - Submit to get disease probability and binary prediction.

## Project Structure
```
disease-diagnosis-dashboard/
├── app.py                 # Main Streamlit application
├── heart.csv              # Optional: Real dataset (UCI Heart Disease format)
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Dataset
- **Real Data**: Assumes `heart.csv` with columns: `age`, `sex`, `chest_pain_type`, `resting_bp`, `cholesterol`, `fasting_blood_sugar`, `max_heart_rate`, `exercise_angina`, `disease` (binary target).
- **Synthetic Data**: Generated if no file found, based on realistic distributions and correlations for heart disease risk.

## Model Architecture
- Input: 8 features.
- Hidden Layers: 64 → ReLU → 32 → ReLU.
- Output: Sigmoid for binary probability.
- Loss: Binary Cross-Entropy.
- Optimizer: Adam.

## Contributing
Contributions are welcome! Please fork the repo, make changes, and submit a pull request.

## License
MIT License - feel free to use and modify.

## Acknowledgments
- Created by [Aishwarya Pagare](https://github.com/aishwaryapagare) as a demonstration project.
- Supported by [New Gen Tech](https://newgentech.com) – Empowering innovative AI and ML solutions.

---

*For issues or feedback, open a GitHub issue.*

---

**To upload to GitHub:**

1. **Create a new repository on GitHub:**
   - Go to [github.com](https://github.com) and sign in (or create an account).
   - Click the "+" icon > "New repository".
   - Name it `disease-diagnosis-dashboard` (or your preferred name).
   - Make it public/private as desired.
   - **Do not** initialize with README, .gitignore, or license (we'll push them).
   - Click "Create repository".

2. **Prepare your local files:**
   - Save the provided code as `app.py` in a new folder named `disease-diagnosis-dashboard`.
   - Create `requirements.txt` with the following content:
     ```
     streamlit==1.28.1
     torch==2.0.1
     torchvision==0.15.2
     torchaudio==2.0.2
     scikit-learn==1.3.0
     pandas==2.0.3
     numpy==1.24.3
     matplotlib==3.7.2
     seaborn==0.12.2
     ```
   - (Optional) Download a sample `heart.csv` from [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) and place it in the folder.
   - Save the above as `README.md`.

3. **Initialize and push to GitHub:**
   - Open terminal in the folder.
   - Run:
     ```
     git init
     git add .
     git commit -m "Initial commit: Disease Diagnosis Dashboard by Aishwarya Pagare"
     git branch -M main
     git remote add origin https://github.com/YOUR_USERNAME/disease-diagnosis-dashboard.git  # Replace YOUR_USERNAME
     git push -u origin main
     ```

4. **Update repository description (optional):**
   - On GitHub, edit the repo: Add description like "Streamlit dashboard for heart disease prediction using PyTorch."
   - Add topics: `streamlit`, `pytorch`, `machine-learning`, `healthcare-ai`.

If you encounter issues, share the error for troubleshooting! 🚀
