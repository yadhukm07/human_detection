# human_detection

prerequistics::Nueral netwoks basics,convolutional nueral netwok basics,

HUMAN DETECTION USING DEEP LEARNING

 -->Detection of humans from the image captured by the drone images and low altitude drone image using deeplearning model with convolutional nueral network

-->The final model is the model trained using 50000 images which is trained on resnet101 which is a faster-rcnn model

The project can be devided into 5 phases

1.Data set collection(image collection)

2.Data set preprocessing+Data set annottation/labelling

3.installing and deveoloping tensorflow model,installing all the required dependencies and libraries

4.Training and Testing using the dataset on the model

5.Testing with the real examples and retraing by changing various parameters

PHASE 1:DATA SET COLLECTION 

--->For better accuracy and result the model have to be trained with a well defined dataset collecting the data set is the first and major major work

-->collected  50000 ariel and drone images
-->Source
    
    *UAV123------>  "https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx "
    
    *Kerala flood images-----> eg: "https://images.financialexpress.com/2018/08/pic-8.jpg"
    
    *flood ariel images
    
    *Screenshots  from Drone videos of humans from youtube and google-- eg: "https://youtu.be/2JuSrDF4bmo"
    
    *All collected data::--->"https://drive.google.com/open?id=1oAzlGOaeC575patIbAZEIPoE3T6NM89_"
    
PHASE  2: DATA SET PREPROCESSING AND ANNOTTATION

--->for training the model required tf-record which is build from the dataset

--->This tf-record cna be made from the images through the following steps

   1--making all image to a standard resolution(1920*720)
      
       --copy all iamges to a repository
       --make an additional repsitory 
       --copy the image repository address and destination folder address to labelimage.py python code
       







 
 
 



