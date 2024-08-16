import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Initialize BioBert
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
biobert_model = TFBertModel.from_pretrained('dmis-lab/biobert-v1.1', from_pt=True)

# Load the Exercise Recommendations CSV (Training Data)
ex_rec = './ex_rec.csv'
df_recs = pd.read_csv(ex_rec)

# Load the Exercise Dataset
ex_ds = './ex_ds.xlsx'
df_ex = pd.read_excel(ex_ds, sheet_name='output')

def get_biobert_embedding(text):
    """Generate BioBERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = biobert_model(inputs)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()

# Generate Exercise Embeddings
df_ex['Exercise Embedding'] = df_ex['Exercise name'].apply(lambda x: get_biobert_embedding(x))

# Save all Exercise Embeddings in a dictionary
exercise_embeddings = {row['ExerciseID']: row['Exercise Embedding'] for index, row in df_ex.iterrows()}

# Preprocess the patient profile data
le_gender = LabelEncoder()
df_recs['Gender'] = le_gender.fit_transform(df_recs['Gender'])

# Normalize Age and BMI
scaler = StandardScaler()
df_recs[['Age', 'BMI']] = scaler.fit_transform(df_recs[['Age', 'BMI']])

# Handle missing values in 'Affected Area'
df_recs['Affected Area'] = df_recs['Affected Area'].fillna('Unknown')

# Create embeddings for Medical Condition and Affected Area
df_recs['Medical Condition Embedding'] = df_recs['Medical Condition'].apply(lambda x: get_biobert_embedding(x))
df_recs['Affected Area Embedding'] = df_recs['Affected Area'].apply(lambda x: get_biobert_embedding(x))
