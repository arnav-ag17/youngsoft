
# Exercise Recommendation System

## Description

This project is designed to recommend exercises to patients based on their profile, including factors like medical conditions, affected areas, age, gender, and BMI. The system uses machine learning techniques, including BioBERT embeddings for text data and a custom tokenizer for exercise recommendations.

## Table of Contents

- [Exercise Recommendation System](#exercise-recommendation-system)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
    - [📁 Project Structure](#-project-structure)
  - [🔍 Module Details](#-module-details)
    - [1. Data Loading and Preprocessing Module](#1-data-loading-and-preprocessing-module)
    - [2. Utility and Helper Functions Module](#2-utility-and-helper-functions-module)
    - [3. Feature Engineering and Model Input Preparation Module](#3-feature-engineering-and-model-input-preparation-module)
    - [4. Model Definition and Training Module](#4-model-definition-and-training-module)
    - [5. Prediction and Post-Processing Module](#5-prediction-and-post-processing-module)
  - [Examples](#examples)
  - [Contributing](#contributing)
  - [License](#license)

---

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/exercise-recommendation-system.git
   cd exercise-recommendation-system
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

### 📁 Project Structure

```plaintext
├── data_preprocessing.py
├── feature_engineering.py
├── model_training.py
├── prediction.py
└── utils.py
```

---

## 🔍 Module Details

### 1. Data Loading and Preprocessing Module

The `data_preprocessing.py` module is responsible for loading the data, preprocessing it, and generating BioBERT embeddings.

**Key Components:**

- **BioBERT Tokenizer:**
  ```python
  tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
  biobert_model = TFBertModel.from_pretrained('dmis-lab/biobert-v1.1', from_pt=True)
  ```

  BioBERT is a variant of the BERT model specifically fine-tuned for biomedical text. It helps generate embeddings for medical conditions, affected areas, and exercise names, making it easier for the model to understand the text-based features.

- **Generating BioBERT Embeddings:**
  ```python
  def get_biobert_embedding(text):
      inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
      outputs = biobert_model(inputs)
      return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()

  df_ex['Exercise Embedding'] = df_ex['Exercise name'].apply(lambda x: get_biobert_embedding(x))
  ```

  This function generates BioBERT embeddings for the exercise names and other text features, which are then used as input for the machine learning model.

### 2. Utility and Helper Functions Module

The `utils.py` module contains helper functions and custom classes that support the main functionalities of the project.

**Key Components:**

- **Custom Tokenizer:**
  ```python
  class CustomTokenizer:
      def __init__(self):
          self.word_index = {}
          self.index_word = {}
          self.next_index = 1

      def fit_on_texts(self, texts):
          for text in texts:
              if text not in self.word_index:
                  self.word_index[text] = self.next_index
                  self.index_word[self.next_index] = text
                  self.next_index += 1

      def texts_to_sequences(self, texts):
          sequences = []
          for text in texts:
              sequence = []
              for word in text.split(', '):
                  word = word.strip()
                  if word in self.word_index:
                      sequence.append(self.word_index[word])
              sequences.append(sequence)
          return sequences
  ```

  The `CustomTokenizer` is designed to convert exercise names into sequences of indices. Unlike standard tokenizers, this custom version allows for more control over the tokenization process, specifically tailored to the structure of the exercise data.

### 3. Feature Engineering and Model Input Preparation Module

The `feature_engineering.py` module prepares the data for model training with the following steps:

- **Embedding Extraction:**
  - Extract **768-dimensional BioBERT embeddings** for both **medical conditions** and **affected areas**.

- **Numeric Feature Preparation:**
  - Normalize numeric features: **age**, **gender**, and **BMI**.

- **Input Construction:**
  - Combine embeddings and numeric features into a **single input tensor** for the model.

This module ensures that all features are properly formatted and scaled, ready for input into the neural network.


### 4. Model Definition and Training Module

The `model_train.py` module defines and trains a neural network model with the following architecture:

- **Inputs:** The model takes five inputs:
  - 768-dimensional embeddings for **medical conditions** and **affected areas** (generated using BioBERT).
  - Numeric features: **age**, **gender**, and **BMI**.

- **Layer Structure:**
  - **Concatenation Layer:** All inputs are concatenated into a single tensor.
  - **Dense Layers:**
    - First dense layer: **128 neurons** with ReLU activation.
    - Second dense layer: **64 neurons** with ReLU activation.
  - **Output Layer:** A dense layer with **sigmoid activation** outputs probabilities for each exercise.

- **Compilation:**
  - **Optimizer:** Adam
  - **Loss Function:** Binary cross-entropy
  - **Evaluation Metric:** Binary accuracy


### 5. Prediction and Post-Processing Module

The `prediction.py` module is responsible for generating exercise recommendations for new patient profiles using the trained model.

**Key Components:**

- **Making Predictions:**
  ```python
  predicted_probs = model.predict([test_condition_embedding, test_area_embedding, test_age, test_gender, test_bmi])
  ```

  The model generates probability scores for each exercise based on the patient's profile. These predicted probabilities are then converted to binary labels using a threshold, and the labels are mapped back to exercise names using the `CustomTokenizer`.

---

## Examples

You can find examples of how to use the project in the provided Jupyter Notebook (`exercise_recommendation_system.ipynb`). This notebook includes exploratory data analysis and demonstrates how to use the different components of the project.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. **Fork the repository.**
2. **Create a new branch** (`git checkout -b feature-branch`).
3. **Make your changes and commit them** (`git commit -m 'Add new feature'`).
4. **Push to the branch** (`git push origin feature-branch`).
5. **Open a Pull Request.**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
