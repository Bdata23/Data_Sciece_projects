import joblib
model = joblib.load("C:\Users\white\Documents\Project P\potato-disease-classification-model.ipynb")
class_labelS=['Pepper__bell___Bacterial_spot...','Pepper__bell___healthy...','Potato___Early_blight...','Potato___Late_blight...','Potato___healthy...','Processing Tomato_Bacterial_spot...','Processing Tomato_Early_blight...','Processing Tomato_Late_blight...','Processing Tomato_Leaf_Mold...','Tomato_Septoria_leaf_spot...','Tomato_Spider_mites_Two_spotted_spider_mite...','Tomato__Target_Spot...','Tomato__Tomato_YellowLeaf__Curl_Virus...','Tomato__Tomato_mosaic_virus...','Tomato_healthy...']
import cv2
import numpy as np

# Define IMG_SIZE based on the input size expected by your model
IMG_SIZE = (100, 100)  # Example size, replace with the appropriate size for your model
# ,khgkjhgkh
# Load and preprocess the test image
def preprocess_test_image(img_path):
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        # Resize image to desired size
        img = cv2.resize(img, IMG_SIZE)
        # Normalize pixel values to range [0, 1]
        img = img / 255.00
        return img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Path to the test image (use double backslashes or a raw string)
test_image_path = r'/content/00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL 7886.JPG'

# Preprocess the test image
test_image = preprocess_test_image(test_image_path)

# Make predictions
if test_image is not None:
    # Reshape the image to match the model's input shape
    test_image = np.expand_dims(test_image, axis=0)
    # Make predictions
    predictions = model.predict(test_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    # Decode the predicted class index to get the disease class
    predicted_disease_class = class_labelS[predicted_class_index]  # Assuming class_labels is defined somewhere
    # Print the predicted disease class
    print(f"Predicted Disease: {predicted_disease_class}")
    print(predictions)
    print(predicted_class_index)
