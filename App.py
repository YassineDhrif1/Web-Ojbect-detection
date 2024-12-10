import os
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tqdm import tqdm
from PIL import Image, ImageDraw

# Set environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and resize image function using TensorFlow
def load_and_resize_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (416, 416))
    return img.numpy()

# Y4:0 model architecture
def y4_0_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(4 * 3)(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Load CSV file
df = pd.read_csv('train/_annotations.csv')

# Select first 10k rows
df = df.head(10000)

# Create custom data generator
def data_generator(df, batch_size):
    while True:
        for i in range(0, len(df), batch_size):
            batch_images = np.array([load_and_resize_image(f"train/{row['filename']}") for row in df.iloc[i:i+batch_size]])
            batch_annotations = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']] for row in df.iloc[i:i+batch_size]])
            
            yield batch_images, batch_annotations

# Create data generators
train_gen = data_generator(df, batch_size=32)

# Print filenames
print("Training filenames:")
for filename in df['filename']:
    print(filename)

# Set up GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU Memory: {tf.memory.get_gpu_memory()}")
else:
    print("No GPU found")

# Create Y4:0 model
model = y4_0_model((416, 416, 3))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    train_gen,
    epochs=50,
    verbose=1,
    validation_split=0.1,
    callbacks=[tqdm()]
)

# Save the model
model.save('website_element_detector.h5')

print("Model saved successfully.")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(train_gen)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Visualize results
def visualize_results(image_path, predictions):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    for pred in predictions:
        class_prob = pred['probability']
        xmin, ymin, xmax, ymax = pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']
        
        # Draw bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red')
        
        # Add class label and probability
        draw.text((xmin, ymin - 10), f"{pred['class']}: {class_prob:.2f}", fill="white")
    
    img.show()

# Example usage
def predict_and_draw(image_path):
    # Assuming you have a pre-trained model
    model = tf.keras.models.load_model('website_element_detector.h5')
    
    # Load and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (416, 416))
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(416, 416), color_mode='rgb')
    
    # Make predictions
    predictions = model.predict(np.array([img]))
    
    return post_process_predictions(predictions)

new_image_path = "path/to/new/image.jpg"
predictions = predict_and_draw(new_image_path)
visualize_results(new_image_path, predictions)

# Helper functions
def post_process_predictions(predictions):
    batch_size = predictions.shape[0] // 12
    processed_predictions = []
    
    for i in range(batch_size):
        start_idx = i * 12
        end_idx = (i + 1) * 12
        
        img_predictions = predictions[start_idx:end_idx]
        
        class_probs = img_predictions[:4]
        coords = img_predictions[4:]
        
        max_prob_class = np.argmax(class_probs)
        
        if max_prob_class == 0:
            adjusted_coords = coords
        elif max_prob_class == 1:
            adjusted_coords = coords * 1.5
        else:
            adjusted_coords = coords * 0.8
        
        processed_predictions.append({
            'class': ['image', 'heading', 'link'][max_prob_class],
            'probability': max(class_probs),
            'xmin': adjusted_coords[0],
            'ymin': adjusted_coords[1],
            'xmax': adjusted_coords[2],
            'ymax': adjusted_coords[3]
        })
    
    return processed_predictions

def check_file_paths(df):
    for _, row in df.iterrows():
        full_path = f"train/{row['filename']}"
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
    return df

# Check file paths
df = check_file_paths(df)
