import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import cv2
import numpy as np

font=ImageFont.truetype("THSarabunNew.ttf",30)
font_s=ImageFont.truetype("THSarabunNew.ttf",20)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(img,knn_clf, distance_threshold=0.48):
    
    t1=time.time()
  
    # find location
    X_face_locations = face_recognition.face_locations(img)
    print(time.time()-t1)

    #if no face
    if len(X_face_locations) == 0:
        return []
   
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=5)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    



    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(cvframe,predictions):
    
    
    pilframe = Image.fromarray(cvframe)   
    draw = ImageDraw.Draw(pilframe)
    

    name1=predictions[0][00]            
        
        
    draw.text((left + 6, bottom - 40),name1, font=font, fill=(255,255,255))
       

    cvframe = np.asarray(pilframe)   
    cvframe=cv2.resize(cvframe, (768,768))    
    cv2.imshow('window', cvframe)
    

    

   

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)    


    t1=time.time()
    print("Training KNN classifier...")
    #classifier = train("train", model_save_path="trained_knn_model.clf", n_neighbors=5)
    print("Training complete!")
    print('training time : ',time.time()-t1)


    
    with open('trained_knn_model.clf', 'rb') as f:
        knn_clf = pickle.load(f)






 
    pred_this_frame=1
    
    while True:
        
        ret, cvframe = video_capture.read()
        if pred_this_frame == 1:        
            
            small_cvframe = cv2.resize(cvframe, (0, 0), fx=0.25, fy=0.25)
            rgb_small_cvframe = small_cvframe[:, :, ::-1]
            
            predictions = predict(rgb_small_cvframe, knn_clf)
            
            # Display results overlaid on an image            
            pred_this_frame = 0
           


        pred_this_frame +=1 
        show_prediction_labels_on_image(cvframe, predictions)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




