# cosmicode-week-5-task
Here is the week 5 task for remote internship at cosmicode in the domain of machine learning 
Here’s a **professional README.md** content for your **ML Week 5** GitHub repository, excluding Tasks 5 and 6, and mentioning that the work runs on **Google Colab (Python)**.

---

```markdown
# Machine Learning – Week 5 Projects

This repository contains my implementations of advanced machine learning algorithms and deep learning models completed during **Week 5** of my ML learning journey.  
All tasks were developed and tested in **Google Colab (Python)**, ensuring smooth, error-free execution with GPU acceleration when applicable.

## 📌 Contents

### **Task 1 – Gradient Boosting Machine (XGBoost)**
- Implemented a **Gradient Boosting Machine (GBM)** using the `XGBoost` library.
- Tuned hyperparameters with `GridSearchCV` to achieve optimal performance.
- Evaluated on the **Iris dataset** with accuracy and classification metrics.
- Key Skills: Gradient Boosting, Hyperparameter Tuning, Model Evaluation.

### **Task 2 – Recurrent Neural Network (RNN) for Time Series**
- Built **LSTM-based** RNN for **time series forecasting**.
- Used synthetic sine wave data to demonstrate sequence prediction.
- Applied **windowing technique** for input data preparation.
- Key Skills: LSTM, Sequential Data Processing, Forecasting.

### **Task 3 – NLP Pipeline with Embeddings + RNN**
- Created an **NLP pipeline** using **Sentence Transformers** for text embeddings.
- Applied embeddings to a **Logistic Regression classifier** for sentiment analysis.
- Dataset: Custom small sentiment dataset (positive/negative labels).
- Key Skills: Word Embeddings, Text Classification, Sentiment Analysis.

### **Task 4 – Generative Adversarial Network (GAN)**
- Implemented a **DCGAN** to generate synthetic images from noise.
- Trained on the **MNIST dataset** for digit generation.
- Key Skills: GAN Architecture, Image Generation, Deep Learning.

### **Task 5– Traditional ML vs Deep Learning**
- Compared performance of classical ML models (**Logistic Regression, Random Forest, SVM**) with a **Deep Neural Network** on the **Iris dataset**.
- Evaluated based on accuracy and training time.
- Key Skills: Model Comparison, Performance Metrics, Neural Networks.


## 🛠️ Technologies Used
- **Python** (Google Colab)
- **Libraries**:  
  - `numpy`, `pandas`, `matplotlib`, `scikit-learn`  
  - `xgboost`  
  - `tensorflow` / `keras`  
  - `sentence-transformers`  
  - `gensim`  
  - `Pillow`, `tqdm`



## 🚀 How to Run
1. Open the provided `.ipynb` notebook in **Google Colab**.
2. Ensure GPU runtime is enabled:  
   `Runtime → Change runtime type → Hardware accelerator → GPU`.
3. Run all cells sequentially.
4. Follow comments in code for dataset download or usage.

Here is the link of my colab notebook you can visit for detals.....

https://colab.research.google.com/drive/1x9y5bweOlynuXuIKiYoSARsQD3lYtZWO?usp=sharing

Downloaded file is attached.....



## 📈 Results Summary
| Task | Model(s) | Dataset | Best Metric |
|------|----------|---------|-------------|
| 1 | XGBoost GBM | Iris | ~97% Accuracy |
| 2 | LSTM RNN | Synthetic sine wave | RMSE < 0.1 |
| 3 | Sentence Transformer + Logistic Regression | Custom Sentiment | ~95% Accuracy |
| 4 | DCGAN | MNIST | Realistic digit samples |
| 5 | Logistic, RF, SVM, DNN | Iris | DNN ~98% Accuracy |



## 📜 License
This project is released under the MIT License – see the [LICENSE](LICENSE) file for details.


**Author:** Nayab Akhtar  

