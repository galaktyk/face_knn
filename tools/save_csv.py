
import os
import subprocess as sp
import time
import csv
class save_csv():


    def __init__(self):
        
        self.filename=str(time.strftime("%Y-%m-%d", time.localtime())+'.csv')
     
        
        if not os.path.isfile('csv/'+self.filename) :  ## if file does not exist then make it!   
            
            with open('csv/'+self.filename,  mode='a') as f: 
                        
                        writer=csv.writer(f)
                        writer.writerow(['Datetime','name'])                        
                        print('Make file csv/'+self.filename)
                       
                        
 
        


    def save_this(self,record):

        
      
        with open('csv/'+self.filename,  mode='a') as f: #append mode
            
            writer=csv.writer(f)
            writer.writerow(record)
        #print('CSV Appended')
        self.df=None #clear




if __name__ == '__main__':
   
    obj=save_csv()
    in_,out_,cust=obj.startday()
    print(in_,out_,cust)








    
