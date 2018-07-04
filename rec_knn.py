import math
from sklearn import neighbors
import os
import os.path
import argparse
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import cv2
import numpy as np
import csv
from tools.save_csv import save_csv




font=ImageFont.truetype("THSarabunNew.ttf",30)
font_s=ImageFont.truetype("THSarabunNew.ttf",20)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}







def train():
    train_dir="train/"
    verbose=True
    X = []
    y = []
    n_neighbors=5
    knn_algo='ball_tree'
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

    np.save('model/X.npy', X) 
    np.save('model/y.npy', y)
       


#___________________________________________________________________________________________________________________________ CLASS

class test():

    def __init__(self):
        self.oldname=[]
        self.t_start=time.time()

        X= np.load('model/X.npy')
        y= np.load('model/y.npy')

        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', weights='distance')
        self.knn_clf.fit(X, y)

        self.database_name=[]
        self.database_bday=[]
        self.database_food=[]
        with open('model/database.csv', 'r') as f:
            d = csv.reader(f)
            for row in d:
                self.database_name.append(row[0])
                self.database_bday.append(row[1])
                self.database_food.append(row[2]) 


    def predict(self,img,distance_threshold=0.48): 
        t1=time.time()       
        X_face_locations = face_recognition.face_locations(img) 
    
        if len(X_face_locations) == 0:
            return []
       
        faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=5)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        







        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

































#_____________________________________________________________BOX FUNCTION___________________________________________

    def show_box(self,cvframe, predictions):

        
        if len(self.oldname) > 0:
            if time.time() - self.oldname[0][1] >= 10:
                self.oldname.pop(0) if len(self.oldname) != 0 else None
                


        print(self.oldname)
        pilframe = Image.fromarray(cvframe)   
        draw = ImageDraw.Draw(pilframe)

        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))      
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - 40),name, font=font, fill=(255,255,255))


            if name != 'unknown':
                index=self.database_name.index(name)            
                draw.text((left + 50, top+5),self.database_bday[index], font=font_s, fill=(255,255,255))
                draw.text((left + 70, bottom-30),self.database_food[index], font=font_s, fill=(255,255,255))


                if name not in [item[0] for item in self.oldname]:
                    record=[time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), name]
                    csv_obj.save_this(record)
                    self.oldname.append((name,time.time()))
                else:
                    inx=[item[0] for item in self.oldname].index(name)
                    self.oldname.pop(inx)
                    self.oldname.append((name,time.time()))           

        cvframe = np.asarray(pilframe)        
        cv2.imshow('', cvframe)        
        







#_____________________________________________________________SNAP FUNCTION___________________________________________




    def show_snap(self,cvframe, predictions):

        count=len(predictions)

        if count != 0:
         
            cut_size=int(1024/(count*2))

            pilframe = Image.fromarray(cvframe)   
            

            #vis=np.array(np.zeros((cut_size,1,3)))

            
            for name, (top, right, bottom, left) in predictions:
                if name != 'unknown':

                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                                       
                    

                    impred = cvframe[top:bottom+20, left:left+(bottom-top)+20]       
                    impred = cv2.resize(impred,(cut_size,cut_size))

                    
                    imbase = Image.open('train/'+name+'/'+name+'.jpg')                   
                    imbase = imbase.resize((cut_size,cut_size))

                    # ________________________________________________________DRAW SNAP info
                    draw = ImageDraw.Draw(imbase)
                    index=self.database_name.index(name)  
                    draw.text((20, 350),name, font=font_s, fill=(255,255,255))
                    draw.text((20, 400),self.database_bday[index], font=font_s, fill=(255,255,255))
                    draw.text((20, 450),self.database_food[index], font=font_s, fill=(255,255,255))

                    imbase = np.asarray(imbase) 
                    imbase = cv2.cvtColor(imbase, cv2.COLOR_BGR2RGB)
                    fullim = np.concatenate((impred,imbase), axis=1) 
                
                   
                    
                else:
                    return []               
          
            
            cv2.imshow('', fullim)             

           






























if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='face_recognition using dlib and KNN')
    parser.add_argument('--mode',default='test',help='train or test (use test by default)')    
    args = parser.parse_args()
    print("running : ",args.mode)

    csv_obj=save_csv() 


    



    if args.mode == "train" :

        t1=time.time()
        print("Training...")
        train()     
        print('complete! : ',time.time()-t1)

    else:      


        test_obj=test()

        
        video_capture = cv2.VideoCapture(0)    
        while True:
            
            ret, cvframe = video_capture.read()
          
            
            small_cvframe = cv2.resize(cvframe, (0, 0), fx=0.25, fy=0.25)
            rgb_small_cvframe = small_cvframe[:, :, ::-1]
            
            predictions = test_obj.predict(rgb_small_cvframe)
        
            test_obj.show_box(cvframe, predictions) if args.mode =='test' else None
            test_obj.show_snap(cvframe, predictions) if args.mode == 'snap' else None


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




