import function

image_path = r"F:\Programs\plant\backend\leaf-disease-detection-backend\wheat.jpg"
a = function.evaluate(image_path)
if a:
    print(f"Detected disease: {a}")
else:
    print("Failed to evaluate the image.")
