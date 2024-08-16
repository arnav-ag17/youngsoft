import tensorflow as tf
from sklearn.model_selection import train_test_split
from feature_engineering import X_combined, y  # Assuming feature engineering outputs are used here

# Define model inputs
input_condition = tf.keras.Input(shape=(768,), name='condition_embedding')
input_area = tf.keras.Input(shape=(768,), name='area_embedding')
input_age = tf.keras.Input(shape=(1,), name='age')
input_gender = tf.keras.Input(shape=(1,), name='gender')
input_bmi = tf.keras.Input(shape=(1,), name='bmi')

# Concatenate inputs and define model layers
concat = tf.keras.layers.Concatenate()([input_condition, input_area, input_gender, input_bmi])
dense_1 = tf.keras.layers.Dense(128, activation='relu')(concat)
dense_2 = tf.keras.layers.Dense(64, activation='relu')(dense_1)

num_classes = y.shape[1]
output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(dense_2)

# Compile model
model = tf.keras.Model(inputs=[input_condition, input_area, input_age, input_gender, input_bmi], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

# Perform train-test split
X_train_combined, X_test_combined, y_train, y_test = train_test_split(
    X_combined, 
    y, 
    test_size=0.2, 
    random_state=42
)

# Train the model
model.fit(
    [X_train_combined[:, :768], X_train_combined[:, 768:1536], X_train_combined[:, 1536:1537], 
     X_train_combined[:, 1537:1538], X_train_combined[:, 1538:]],
    y_train,
    validation_data=(
        [X_test_combined[:, :768], X_test_combined[:, 768:1536], X_test_combined[:, 1536:1537], 
         X_test_combined[:, 1537:1538], X_test_combined[:, 1538:]],
        y_test
    ),
    epochs=10,
    batch_size=32
)
