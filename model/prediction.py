import numpy as np
from model_train import model  
from utils import CustomTokenizer 

# Example patient profile for prediction
test_condition_embedding = np.random.rand(1, 768)  # Simulated condition embedding
test_area_embedding = np.random.rand(1, 768)       # Simulated area embedding
test_age = np.array([[45]])                        # Example age
test_gender = np.array([[0]])                      # Example gender (0 = male, 1 = female)
test_bmi = np.array([[23.5]])                      # Example BMI

# Predict using the model
predicted_probs = model.predict([test_condition_embedding, test_area_embedding, test_age, test_gender, test_bmi])

# Convert probabilities to binary predictions
threshold = 0.85  # You can adjust this threshold
predicted_labels = (predicted_probs > threshold).astype(int)

# Get indices of predicted exercises
predicted_exercises = np.where(predicted_labels[0] == 1)[0] + 1  # Add 1 to match the exercise indices

# Map indices back to exercise names using the tokenizer
exercise_tokenizer = CustomTokenizer()  # Assuming it's initialized and fitted previously
predicted_exercise_names = [exercise_tokenizer.index_word[idx] for idx in predicted_exercises]

print("Predicted exercises (names):")
print(predicted_exercise_names)
