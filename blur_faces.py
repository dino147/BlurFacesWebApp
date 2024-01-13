import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

n_uploaded_img = 0

def load_data(folder_path, num_images=None):
    images = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (64, 64))  # Resize to a consistent size
            images.append(img)
            count += 1
            if num_images is not None and count >= num_images:
                break

    return np.array(images)

def load_data_and_train():
    data_folder = r'C:\Users\Alex\Desktop\Scoala\master\anul I\PPC\proiectPPC\uploads'
    X_train = load_data(data_folder, num_images=100000)
    print(len(X_train)) 
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    labels = np.ones((X_train.shape[0], 1))  # Assuming all images contain faces
    
    model.fit(X_train, labels, epochs=10, validation_split=0.2)
    
    model.save("saved_models/sequential_model.h5")


def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

# new_image_path = r'C:\Users\User\Desktop\Scoala\master\anul I\PPC\proiectPPC\uploads\poza-profil.jpeg'
# # new_image_path = '/Users/vladcalomfirescu/Desktop/MyFiles/FAC/master/an 1/sem 1/PPC/BlurFacesML/data/img_align_celeba/201922.jpg'

# model = load_model("saved_models/sequential_model.h5")
# result = predict_image(model, new_image_path)
# print(result)

# if result >= 1:
#     print("The image contains a cropped face.")
# else:
#     print("No cropped face detected in the image.")
    
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  
    return img

def extract_patches(image, patch_size, stride):
    patches = []
    height, width, _ = image.shape

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)

    return patches

def predict_patches(model, patches):
    predictions = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)
        prediction = model.predict(patch)
        predictions.append(prediction[0][0])
    return predictions

def detect_faces(predictions, threshold=0.5):
    average_score = np.mean(predictions)
    if average_score > threshold:
        return True
    else:
        return False
    
def blur_faces(image_path):
    detector = MTCNN()
    img = cv2.imread(image_path)

    faces = detector.detect_faces(img)

    for face in faces:
        x, y, width, height = face['box']
        roi = img[y:y + height, x:x + width]
        
        roi = cv2.GaussianBlur(roi, (99, 99), 30)

        img[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

    # Display the result
    #cv2.imshow("Blurred Faces", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return img

def check_and_blur(larger_image_path):    
    #larger_image_path = r'C:\Users\Alex\Desktop\Scoala\master\anul I\PPC\proiectPPC\uploads\piscotii.jpeg'
    larger_image = preprocess_image(larger_image_path)
    
    patch_size = 64  
    stride = 16  
    
    os.chdir(r'C:\Users\Alex\Desktop\Scoala\master\anul I\PPC\proiectPPC') 
    global n_uploaded_img
    n_uploaded_img += 1
    """ Train for each 10 new uploaded images !"""
    if n_uploaded_img % 10 == 0:
        load_data_and_train()
        
    print(n_uploaded_img)
        
    model = load_model("saved_models/sequential_model.h5")
    
    patches = extract_patches(larger_image, patch_size, stride)
    predictions = predict_patches(model, patches)
    result = detect_faces(predictions, threshold=0.999)
    
    if result:
        img_blured = blur_faces(larger_image_path)
        print("The larger image contains a face.")
        os.chdir(r'C:\Users\Alex\Desktop\Scoala\master\anul I\PPC\proiectPPC\uploads') 
        img_name = "img" + str(n_uploaded_img) + ".jpg"
        cv2.imwrite(img_name, img_blured)
        return img_name
    else:
        print("No face detected in the larger image.")
        
    


