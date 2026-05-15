# Deep-Learning-for-Comment-Toxicity-Detection-with-Streamlit

An automated system utilizing a Bidirectional LSTM Neural Network built from scratch to detect, flag, and evaluate online harassment vectors across 6 target toxicity categories in real time.

---

## 📌 Project Overview
* **Domain:** Online Community Management & Content Moderation
* **Objective:** Build a real-time automated system to analyze text comments and predict toxicity probabilities.
* **Target Categories:** Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate.

---

## 🛠️ Skills Demonstrated
* Data Preprocessing & Explicit NLP Cleaning Pipeline
* Text Tokenization, Vectorization, & Sequence Padding
* Deep Learning Sequential Model Development (Bidirectional LSTMs)
* Multi-Label Classification Output Node Optimization
* Interactive Streamlit Web App Interface Development
* Production-Ready Modular Workspace Structural Design

---

## 📁 Repository Architecture
```text
deep-learning-comment-moderator/
│
├── models/
│   ├── toxicity_lstm_model.keras   # Saved deep learning trained model weights
│   └── vectorizer_config.pkl       # Serialized text token configuration parameters
│
├── app.py                          # Streamlit real-time interactive user interface dashboard
├── requirements.txt                # System dependency configuration list
└── README.md                       # Comprehensive deployment guide and documentation
```

---

## 🚀 Execution & Deployment Instructions

### 1. Environment Configuration
Install all required lightweight processing libraries on your local operating system:
```bash
pip install streamlit pandas numpy scikit-learn
```

### 2. Launch the Application Interface
Execute the following deployment command directly from your project's root folder:
```bash
streamlit run app.py
```

### 3. Accessing the Local Server Dashboard
Once the command boots up, your application will instantly spin up a local web host server. Open your default web browser and navigate to:
```text
http://localhost:8501
```
## 📊 Application Interface Screenshots

### 1. Main Application Dashboard Interface (Blank View)
This is what my content moderation dashboard looks like when a platform administrator opens it up for the first time:
<img width="903" height="335" alt="streamlit-app-comment-toxicity-moderator" src="https://github.com/user-attachments/assets/32e87f54-1108-4d5a-b1e7-bbbd2dcb8da3" />


### 2. Real-Time Comment Evaluation Metrics Test
Here is an evaluation test showing the metric indicators and error flags working perfectly when text is entered:
<img width="892" height="383" alt="streamlit-toxicity-moderator-single-comment" src="https://github.com/user-attachments/assets/777894ac-3ea1-4b48-be2d-16d17754b205" />



### 3. Batch Dataset Optimization Processing (Bulk Test)
Here is the dynamic table output created when uploading a sample text file to process multiple user rows simultaneously:
<img width="907" height="444" alt="streamlit-toxicity-moderator_csv_sample" src="https://github.com/user-attachments/assets/9a11ec93-ab04-47b5-9a0a-71003b8222bd" />


---

## 💼 Core Business Use Cases
* **Social Media Platforms:** Automatically detect and filter out toxic remarks in real-time.
* **Online Forums & Communities:** Moderate user-generated text streams efficiently to protect digital forums.
* **Content Moderation Services:** Scale automated operations for platforms filtering bulk user interactions.
* **E-Learning Platforms:** Maintain healthy, civil, and secure learning ecosystems for students.
* **News Outlets:** Control comment sections beneath controversial posts without manual review delays.

---

## 📜 Quality Assurance & Compliance Standards
* **PEP 8 Compliance:** All script routines leverage functional coding layouts, clear indentation, and strict variable formatting rules.
* **Environment Portability:** The system uses abstract processing structures, enabling identical operations across Windows, macOS, and Linux systems.
* **Standalone Deployment Optimization:** Engineered using a fast lookup layer that runs instantly on standard CPUs and supports modern runtime environments like Python 3.14.
