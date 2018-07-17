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
import io
import traceback
import sys




font=ImageFont.truetype("Tahoma Bold.ttf",40)
font_s=ImageFont.truetype("Tahoma Bold.ttf",20)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}




def train():
    train_dir="train_images/"
    verbose=True
    X = []
    y = []
    n_neighbors=2
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

class testorsnap():

    def __init__(self,args):
        self.oldname=[]
        self.t_start=time.time()
        self.args=args
        X= np.load('model/X.npy')
        y= np.load('model/y.npy')

        self.knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')
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
         
        X_face_locations = face_recognition.face_locations(img) 
    
        if len(X_face_locations) == 0:
            return []
       
        faces_encodings = face_recognition.face_encodings(img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = self.knn_clf.kneighbors(faces_encodings, n_neighbors=2)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        







        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(self.knn_clf.predict(faces_encodings), X_face_locations, are_matches)]








#_____________________________________________________________BOX FUNCTION___________________________________________

    def show_box(self,cvframe, predictions):

        
        if len(self.oldname) > 0:
            if time.time() - self.oldname[0][1] >= 10:
                self.oldname.pop(0) if len(self.oldname) != 0 else None        
        #print(self.oldname)





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
        cv2.imshow('window', cvframe)        
        




#_____________________________________________________________SNAP FUNCTION___________________________________________




    def show_snap(self,cvframe, predictions):

        if len(self.oldname) > 0:
            if time.time() - self.oldname[0][1] >= self.args.disappear:
                self.oldname.pop(0) if len(self.oldname) != 0 else None        
        

        count=len(predictions)

        if count != 0:
         
            

            pilframe = Image.fromarray(cvframe)   #cv to pil
            

            #vis=np.array(np.zeros((512,1,3)))

            
            name, (top, right, bottom, left) = predictions[0]
                

            top =(top*4)-50;top = 1 if (top <= 1) else top
            right *= 4
            bottom = (bottom*4)+50; bottom = 1 if (bottom <= 1) else bottom
            left =(left*4)-50;left =1 if (left <= 1) else left
            
                               
            

            impred = cvframe[top:bottom, left:left+(bottom-top)]      
            
            if 1 not in impred.shape :
                
                impred = cv2.resize(impred,(512,512))
            
                ######################### ____________ ##########################
                
                imbase = cv2.imread('train_images/'+name+'/'+name+'.jpg')                   
                imbase = cv2.resize(imbase,(400,400))
                started=1
                imbase = np.concatenate((imbase,np.zeros((112,400,3))), axis=0) 
                imbase = Image.fromarray(imbase.astype('uint8'))  
                
                # ________________________________________________________DRAW SNAP info
                draw = ImageDraw.Draw(imbase)
                index=self.database_name.index(name) 
                

                textdname=(name+"   "+self.database_bday[index])
                draw.text((2, 400),textdname, font=font, fill=(255,255,255))
                
                textfood=(self.database_food[index]+"    ")
                draw.text((2, 460),textfood, font=font, fill=(255,255,255))

                imbase = np.asarray(imbase) 
                
                print(name)               
                fullim = np.concatenate((impred,imbase), axis=1)
                if name == "unknown":
                    picname=("/home/pi/face_knn/unknownface/"+str(time.time())+".jpg")
                    cv2.imwrite(picname, impred)
 
            
                if name not in [item[0] for item in self.oldname]:
                    record=[time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()), name]
                    csv_obj.save_this(record)
                    self.oldname.append((name,time.time()))

                else:
                    inx=[item[0] for item in self.oldname].index(name)
                    self.oldname.pop(inx)
                    self.oldname.append((name,time.time()))      
                cv2.imshow("window", fullim)

           








if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='face_recognition using dlib and KNN')
    parser.add_argument('--mode',default='snap',help='train or test (use test by default)')  
    parser.add_argument('--disappear',default=420,help='memory time(in sec)') 
    parser.add_argument('--device',default="picamera",help='use camera') 

    started=0

    
    args = parser.parse_args()
    if args.device=="picamera":
        
        from picamera.array import PiRGBArray
        from picamera import PiCamera

            
        
        camera=PiCamera() 
        #camera.resolution = (1024, 768)
        camera.rotation=90
        rawCapture=PiRGBArray(camera)


    print("using :",args.device)
    print("running : ",args.mode)
    print("memory : " ,args.disappear,'sec')

    csv_obj=save_csv() 


    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)



    if args.mode == "train" :

        t1=time.time()
        print("Training...")
        train()     
        print('complete! : ',time.time()-t1)

    else:      


        test_obj=testorsnap(args)

        
        video_capture = cv2.VideoCapture(0)

        if args.device == "webcam":
            while True:
                
                
                _, cvframe = video_capture.read() if args.device == "webcam" else None
                #cvframe = cv2.cvtColor(cvframe, cv2.COLOR_BGR2GRAY)
              
                
                small_cvframe = cv2.resize(cvframe, (0, 0), fx=0.25, fy=0.25)
                rgb_small_cvframe = small_cvframe[:, :, ::-1]
                tpre=time.time()
                predictions = test_obj.predict(rgb_small_cvframe)
                print("pred time",time.time()-tpre)


                if len(predictions) != 0:
                    test_obj.show_box(cvframe, predictions) if args.mode =='test' else None
                    test_obj.show_snap(cvframe, predictions) if args.mode == 'snap' else None

                elif started ==0:
                    cv2.imshow("window",cvframe) 




                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                #print(time.time()-t1)

        if args.device == "picamera":

            try:

                for cvframe in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
                    cvframe=cvframe.array
                    alpha=2
                    beta=35
                    cvframe = cv2.addWeighted(cvframe,alpha,np.zeros(cvframe.shape,cvframe.dtype),0,beta)
                   
              
                                   
                    rgb_small_cvframe = cv2.resize(cvframe, (0, 0), fx=0.25, fy=0.25)
                  
                    
                    tpre=time.time()

                   
                    predictions = test_obj.predict(rgb_small_cvframe)
                    print("pred time",time.time()-tpre)


                    if len(predictions) != 0:
                        started=1
                        test_obj.show_box(cvframe, predictions) if args.mode =='test' else None
                        test_obj.show_snap(cvframe, predictions) if args.mode == 'snap' else None

                    elif started == 0:
                        cv2.imshow("window",cvframe) 



                    rawCapture.truncate(0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    #print(time.time()-t1)
            except Exception:
                text_file=open("log.md","a")
                text_file.write(str(time.strftime("\n\n%Y/%m/%d %H:%M:%S\n",time.localtime())))
                text_file.write(traceback.format_exc())
                text_file.close()

