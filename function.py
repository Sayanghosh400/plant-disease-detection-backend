from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Correctly formatted 'classes' list with the missing comma added
classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
           'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
           'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
           'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
           'Tomato_healthy']

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            print("Error: Image not loaded properly.")
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None

def evaluate(image_location):
    loaded_model = load_model("./model.h5")
    # image_location = r"F:\Programs\plant\backend\leaf-disease-detection-backend\wheat.jpg"
    im = convert_image_to_array(image_location)
    if im.size == 0:
        print("Error: Failed to process image.")
        return None
    
    np_image_li = np.array(im, dtype=np.float16) / 255.0
    npp_image = np.expand_dims(np_image_li, axis=0)

    result = loaded_model.predict(npp_image)
    itemindex = np.argmax(result)
    return classes[itemindex]
