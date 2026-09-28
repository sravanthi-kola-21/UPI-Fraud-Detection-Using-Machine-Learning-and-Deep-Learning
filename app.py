import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Global dataset and model storage
data = None
X_train = X_test = y_train = y_test = None
accuracies = {}
fnn_model = None  # Store trained FNN model
dt_model = None   # Store trained Decision Tree model
scaler = None  # Store fitted scaler
train_columns = None  # Store feature columns used in training

# GUI Root Setup (hidden initially)
root = tk.Tk()
root.title("Fraud Detection System")
root.geometry("1200x700")
root.withdraw()  # Hide initially, show after login

# Background for Main Page
main_bg_img = Image.open("images/img2.jpg").resize((1500, 800))
main_bg_photo = ImageTk.PhotoImage(main_bg_img)
main_bg_label = tk.Label(root, image=main_bg_photo)
main_bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Animated Title
title_var = tk.StringVar()
title_label = tk.Label(root, textvariable=title_var, font=("Helvetica", 24, "bold"), fg="white", bg="#004466")
title_label.pack(pady=10)

def animate_title():
    title = "Fraud Detection System Using ML & DL"
    for i in range(len(title) + 1):
        def closure(i=i): title_var.set(title[:i])
        root.after(i * 100, closure)
animate_title()

# Frame Layout
left_frame = tk.Frame(root, width=700, height=600, bg="#f0f0f0")
left_frame.pack(side=tk.LEFT, padx=40, pady=20)
right_frame = tk.Frame(root, width=400, height=600, bg="#002244")
right_frame.pack(side=tk.RIGHT, padx=30, pady=20)

text_area = tk.Text(left_frame, wrap=tk.WORD, width=80, height=35, font=("Helvetica", 14, "bold"))
text_area.pack()

# Login Window
def login_window():
    login = tk.Toplevel()
    login.title("Login")
    login.geometry("400x300")
    login.resizable(False, False)

    bg_img = ImageTk.PhotoImage(Image.open("images/img1.jpg").resize((400, 300)))
    bg_label = tk.Label(login, image=bg_img)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    login.bg_img = bg_img

    tk.Label(login, text="Username:", bg='lightblue', font=('Arial', 12)).pack(pady=10)
    user_entry = tk.Entry(login)
    user_entry.pack()

    tk.Label(login, text="Password:", bg='lightblue', font=('Arial', 12)).pack(pady=10)
    pass_entry = tk.Entry(login, show="*")
    pass_entry.pack()

    def check_login():
        if user_entry.get() == 'admin' and pass_entry.get() == 'admin':
            login.destroy()
            root.deiconify()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    tk.Button(login, text="Login", command=check_login, bg="green", fg="white", font=("Arial", 10, "bold")).pack(pady=20)

# Core Functions
def upload_dataset():
    global data
    file_path = filedialog.askopenfilename(initialdir='dataset',filetypes=[["CSV Files", "*.csv"]])
    if file_path:
        data = pd.read_csv(file_path)
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, str(data.head()))

def preprocess_data():
    global X_train, X_test, y_train, y_test, data, scaler, train_columns
    if data is None:
        messagebox.showerror("Error", "Please upload the dataset first.")
        return
    try:
        df = data.copy()
        drop_cols = ['TransactionID', 'UserID', 'DeviceID', 'IPAddress', 'PhoneNumber', 'BankName', 'Timestamp']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)
        df = pd.get_dummies(df, drop_first=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        X = df.drop('FraudFlag', axis=1)
        y = df['FraudFlag'].astype(int)
        X = X.astype(np.float64)

        train_columns = X.columns

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"✅ Preprocessing Done\nTrain Samples: {len(X_train)}\nTest Samples: {len(X_test)}")

    except Exception as e:
        messagebox.showerror("Preprocessing Error", str(e))

def run_model(model_type):
    global fnn_model, dt_model
    if X_train is None or X_test is None:
        messagebox.showerror("Error", "Please preprocess the data first.")
        return

    model = None
    name = model_type

    if model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "L1 Regression":
        model = LogisticRegression(penalty='l1', solver='liblinear')
    elif model_type == "L2 Regression":
        model = Ridge()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_binary = (preds >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds_binary)
        accuracies[name] = round(acc * 100, 2)
        text_area.delete(1.0, tk.END)
        return text_area.insert(tk.END, f"✅ {name} Accuracy: {acc * 100:.2f}%")

    elif model_type == "Naive Bayes":
        model = GaussianNB()
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = round(acc * 100, 2)
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"✅ {name} Accuracy: {acc * 100:.2f}%")
        dt_model = model
        return
    elif model_type == "Feedforward Neural Network":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies[name] = round(accuracy * 100, 2)
        fnn_model = model
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"✅ {name} Accuracy: {accuracy * 100:.2f}%")
        return
    else:
        messagebox.showerror("Error", f"Model '{model_type}' not implemented.")
        return

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = round(acc * 100, 2)
    text_area.delete(1.0, tk.END)
    text_area.insert(tk.END, f"✅ {name} Accuracy: {acc * 100:.2f}%")

def predict_from_file():
    global scaler, train_columns, dt_model
    file_path = filedialog.askopenfilename(initialdir='dataset',filetypes=[["CSV Files", "*.csv"]])
    if not file_path:
        return

    try:
        test_data = pd.read_csv(file_path)

        # Fill missing columns with 0.0
        for col in train_columns:
            if col not in test_data.columns:
                test_data[col] = 0.0

        # Ensure correct column order
        test_data = test_data[train_columns]

        # Apply same scaling as training
        test_scaled = scaler.transform(test_data)

        if dt_model:
            preds = dt_model.predict(test_scaled)
            labeled_preds = ["Fraudulent" if p == 1 else "Not Fraudulent" for p in preds]

            text_area.delete(1.0, tk.END)
            text_area.insert(tk.END, "Decision Tree Predictions:\n\n")
            for i, label in enumerate(labeled_preds, start=1):
                text_area.insert(tk.END, f"Transaction {i}: {label}\n")

        else:
            messagebox.showerror("Error", "No trained Decision Tree model found. Please run it first.")

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


def logout():
    root.quit()

# Graph display
def show_graphs():
    if not accuracies:
        messagebox.showerror("Error", "No model accuracies to display.")
        return
    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Buttons with full names and styling
button_colors = [
    "#007ACC", "#4CAF50", "#FF9800", "#9C27B0", "#E91E63",
    "#3F51B5", "#009688", "#F44336", "#795548", "#FF5722",
    "#8BC34A"
]

buttons = [
    ("Upload Dataset", upload_dataset),
    ("Preprocess Data", preprocess_data),
    ("Run K-Nearest Neighbors", lambda: run_model("KNN")),
    ("Run L1 Logistic Regression", lambda: run_model("L1 Regression")),
    ("Run L2 Ridge Regression", lambda: run_model("L2 Regression")),
    ("Run Naive Bayes Classifier", lambda: run_model("Naive Bayes")),
    ("Run Decision Tree Classifier", lambda: run_model("Decision Tree")),
    ("Run Feedforward Neural Network", lambda: run_model("Feedforward Neural Network")),
    ("Show Accuracy Graphs", show_graphs),
    ("Predict from CSV File", predict_from_file),
    ("Logout and Exit", logout)
]

for (text, cmd), color in zip(buttons, button_colors):
    tk.Button(right_frame, text=text, command=cmd,
              bg=color, fg='white', font=("Arial", 10, "bold"),
              relief=tk.RAISED, bd=2, width=25, height=2).pack(pady=6)

# Launch login window
login_window()
root.mainloop()