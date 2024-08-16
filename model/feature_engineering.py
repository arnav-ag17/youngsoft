import numpy as np
import pandas as pd
from data_preprocessing import df_recs, df_ex  # Assuming df_recs and df_ex are loaded in data_preprocessing.py
from utils import CustomTokenizer
import tensorflow as tf

# Load previously saved embeddings
exercise_embeddings = {row['ExerciseID']: row['Exercise Embedding'] for index, row in df_ex.iterrows()}

# Tokenize Exercise Recommendations
exercise_tokenizer = CustomTokenizer()
exercise_tokenizer.fit_on_texts(df_ex['Exercise name'])

# Prepare input arrays
X_conditions = np.stack(df_recs['Medical Condition Embedding'].values)
X_areas = np.stack(df_recs['Affected Area Embedding'].values)
X_age = df_recs['Age'].values
X_gender = df_recs['Gender'].values
X_bmi = df_recs['BMI'].values

# Combine the inputs into a single array for the model
X_combined = np.hstack([
    X_conditions, 
    X_areas, 
    X_age.reshape(-1, 1), 
    X_gender.reshape(-1, 1), 
    X_bmi.reshape(-1, 1)
])

# Tokenize and pad the output sequences
y = exercise_tokenizer.texts_to_sequences(df_recs['Exercise Recommendations'])
y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post')
y = np.array(y)
