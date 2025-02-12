# TITLE:
## **SpamShield**
# BRIEF EXPLANATION ABOUT THE WORKING MODEL:
This machine learning-based spam filter predicts whether an email is **spam or not** using a **pre-trained model**. It processes email content with **NLP techniques**, extracts features, and classifies emails using algorithms like **Naïve Bayes, SVM, or Deep Learning**. The system is optimized for **accuracy, real-time predictions, and potential deployment** in email services or web apps. 🚀

# ML ALGORITHM USED:  
This spam filter model uses the **Naïve Bayes algorithm**, a probabilistic classifier based on **Bayes' Theorem**. It is well-suited for text classification tasks like spam detection because it assumes feature independence, making it computationally efficient and effective.  

#### **How the Model Works:**  
1. **Text Preprocessing:** The email content is cleaned by removing stopwords, punctuation, and special characters, then converted into numerical features using techniques like **TF-IDF (Term Frequency-Inverse Document Frequency)** or **Bag-of-Words (BoW)**.  
2. **Probability Calculation:** The Naïve Bayes classifier calculates the probability of an email being spam or ham based on word occurrences.  
3. **Classification:** It applies **Bayes' Theorem** to predict whether the email belongs to the **spam** or **ham** category by comparing likelihood probabilities.  
4. **Prediction & Evaluation:** The model is trained on a labeled dataset and tested using metrics like **accuracy, precision, recall, and F1-score** to ensure reliable predictions.  

# TECHNOLOGY USED FOR BUILDING THE FRONT-END:
The front-end for this spam filter model is built using **Tkinter**, Python’s built-in GUI toolkit.  

#### **Short Description:**  
🔹 **Tkinter** – A standard Python library for creating desktop applications with a simple and interactive graphical user interface (GUI).  
🔹 It provides widgets like **buttons, labels, text boxes, and frames** to build a user-friendly interface.  
🔹 The application takes user input (email text), processes it, and displays the classification result (Spam or Not Spam) in real-time.  

# STEP-BY-STEP EXPLANATION OF THE CODE:
## Backend: spam_filter_backend.py (Machine Learning Model)
This file trains the spam classifier and provides a function to classify messages.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

### Load dataset
data = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')
data = data[['label', 'text']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

### Split data
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Vectorization
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

### Train Model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

### Function to classify a message
def classify_text(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    return "Spam" if prediction == 1 else "Ham"
## How It Works?
The backend of the Spam Message Classifier handles **data processing, model training, and message classification**. It loads a **spam dataset**, encodes labels, and splits the data into **training and testing sets**. A **CountVectorizer** converts text into numerical features, and a **Naïve Bayes classifier** is trained on the processed data. When a user enters a message, it is **vectorized and passed to the trained model**, which predicts whether it is **spam or ham**. The prediction is then sent back to the front-end for display. This ensures **efficient and accurate spam detection** in real-time.
    
## Frontend: spam_filter_gui.py (User Interface with Tkinter)
This file builds the GUI and interacts with the backend.

import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk
from spam_filter_backend import classify_text  # Importing the backend function

### Function to classify message using backend
def classify_message():
    message = entry.get("1.0", tk.END).strip()
    if message == "":
        messagebox.showwarning("Input Error", "Please enter a message to classify.")
        return
    
    prediction = classify_text(message)  # Calling backend function
    result_label.config(text=f"The message is classified as: {prediction}",
                        fg="#FF5733" if prediction == "Spam" else "#33FF57",
                        font=("Helvetica", 14, "bold"))

### Function to exit the app
def exit_app():
    root.destroy()

### GUI Setup
root = tk.Tk()
root.title("Spam Filter")
root.state("zoomed")  # Maximized Window

### Load background image
bg_image = Image.open("unnamed.webp")
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_image)

### Create Canvas
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

### Header
header = tk.Label(root, text="Spam Message Classifier", font=("Helvetica", 20, "bold"), bg="#000000", fg="#00FFFF")
canvas.create_window(root.winfo_screenwidth() // 2, 50, window=header)

### Text Input
entry = scrolledtext.ScrolledText(root, width=80, height=5, font=("Courier New", 12, "italic"))
canvas.create_window(root.winfo_screenwidth() // 2, 150, window=entry)

### Button Frame
button_frame = tk.Frame(root, bg="#000000")
canvas.create_window(root.winfo_screenwidth() // 2, 250, window=button_frame)

classify_button = tk.Button(button_frame, text="Classify", command=classify_message,
                            font=("Helvetica", 12, "bold"), bg="#1E90FF", fg="white", padx=10, pady=5)
classify_button.grid(row=0, column=0, padx=10, pady=10)

exit_button = tk.Button(button_frame, text="Exit", command=exit_app,
                        font=("Helvetica", 12, "bold"), bg="#DC143C", fg="white", padx=10, pady=5)
exit_button.grid(row=0, column=1, padx=10, pady=10)

### Output Label
result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#000000")
canvas.create_window(root.winfo_screenwidth() // 2, 350, window=result_label)

### Run GUI
root.mainloop()
## How It Works?
The front-end of the Spam Message Classifier uses **Tkinter** to create a user-friendly interface. It features a **fullscreen window** with a background image, a **header label**, and a **scrollable text box** for message input. A **button frame** holds the **"Classify"** and **"Exit"** buttons. When the user enters a message and clicks **"Classify"**, it sends the input to the backend for prediction, displaying the result in **red (spam) or green (ham)**. If no text is entered, a **warning message** appears. The **"Exit" button** closes the app smoothly, ensuring a visually appealing and functional experience.


# OUTPUT:

(include the screenshots of accuracy score, confusion matrix & classification report)
(include the screenshot of the working User Interface)

# RESULT:
