# Human Detection from Drone Images using Deep Learning

![alt text](https://github.com/yadhukm07/face_mood/blob/master/index2.png)

Natural disasters across the world, many of them related to the climate, cause
massive losses of life and property. Dangers caused by disasters may cause in
human loss, property damage or a profound environmental effect. In situation of
floods, it is difficult to find affected people or trapped people. It is
difficult to detect humans in this situation as less signs would be available
for help. Moreover, humans are highly vulnerable to dynamical events such as
camera jitter, variations and illumination changes in object sizes.

We were able to generate a model(convolutional neural network) which could be
used to clearly detect and locate(label) humans from images(jpg).

Detection of humans from the image captured by the drone images and low altitude
drone image using deeplearning model with convolutional nueral network.The final
model is developed by trained using 57000 images which is trained on resnet101
which is a faster-rcnn model.

<!-- Prerequisites:::[Nueral netwoks](http://neuralnetworksanddeeplearning.com/chap1.html),[Convolutional Nueral NEtwork](https://skymind.ai/wiki/convolutional-network) -->


The project can be devided into 5 phases

1.Data set collection(image collection)

2.Data set preprocessing+Data set annottation/labelling

3.Tf-record Creation

4.installing and deveoloping tensorflow model,installing all the required dependencies and libraries

5.Training and Testing using the dataset on the model

6.Testing with the real examples and retraing by changing various parameters

## PHASE 1:DATA SET COLLECTION

For better accuracy and result the model have to be trained with a well defined dataset. collecting the data set is the first and major work

collected 57000 ariel and drone images

Source


   1. [UAV123]--[link](http://neuralnetworksanddeeplearning.com/chap1.html)

   2. [Kerala flood images]--[Example link](http://neuralnetworksanddeeplearning.com/chap1.html)

   3. [Flood ariel images]--[sample image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8eX5lCtfdKNb4OHancP85PjDtlqIHdsz5rUGNsU2SMWi7qJA2)

   4. [Screenshots  from Drone videos of humans from youtube]--[sample video](https://youtu.be/2JuSrDF4bmo)


   5. All collected data::--->[Drive link](https://drive.google.com/open?id=1oAzlGOaeC575patIbAZEIPoE3T6NM89)





## PHASE 2: DATA SET PREPROCESSING AND ANNOTTATION

For training the model required tf-record which is build from the dataset

```---
   ---image to xml conversion---(img+xml)-->csv conversion
   ---Csv to tf-record creation
```
This tf-record can be made from the csv file which can be made through the following steps

#### 1--making all image to a standard resolution(1920*720)

    --copy all iamges to a repository

    --make an additional repsitory

    --copy the image repository address and destination folder address to rezize.py python code

    --run the rezize.py and the rezized image will be in the destination folder

#### 2--Rename the all rezized images in serial number(eg:img (1), img (2))

    --This can be done in windows-->"https://youtu.be/5X8OdurpYyM"

#### 3-Annottate/labelling the image--->using labelimage tool


  3.1  ---clone the required git-hub repository-->"https://github.com/tzutalin/labelImg.git"

      ---install Python 3 + Qt5

       ---Steps----

        --> sudo apt-get install pyqt5-dev-tools
        --> sudo pip3 install -r requirements/requirements-linux-python3.txt
        --> make qt5py3
        --> python3 labelImg.py
        --> python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
   -
    --go to the labelimg directory


    ---Go to the data folder and edit the 'predifined_classess.txt' edit the names of the label that  are required to label
      (Add the label 'human')

    ---Run the labelimage.py file from the main directory(labelimage folder)

    ---In the LabelImg graphical image annotation tool select the directory where the images are kept

    ---Select the folder where the annotted file is to be saved

    ---Choose the pascalVOC format for the annotted output

    ---The output file is in 'xml' format


  3.2  --- converting the already annotted text file to the xml format


      Some dataset that available in the internet like UAV123 comes along with the annnotted file.Those files of
      multiple classess or objects.From this annotted files The one corresponds to the class human have to be
      sorted out

      --The annotted file is in txt file format, Convert this file to the xml file format

      --done using--"https://youtu.be/dqNwpIRBOrA"


##  PHASE 3:TF RECORD CREATION

For training the model tf-record file is needed.It is made from the xml files and iamges

  #### 1--xml to csv conversion

    ---Clone the directory "https://github.com/vijeshkpaei/legacy.git"

    ---create a folder images in the directory

    ---create two sub directory 'test' and 'train' in this 'images' directory

    ---copy the 75% of the total data set into the train folder(37500) both image and its annottation

    ---copy the 24% of the total data set into the test folder(12000) both image and its annottation

    ---create seperate csv file for both training and testing data

    ---csv file for training can be created using

        ---> python xml_to_csv.py -i /home/yadhu07/Downloads/legacy-master/images/train -o /home/yadhu07/Downloads/legacy-master/
             images/train.csv

     ---csv file for training can be created using        

         ---> python xml_to_csv.py -i /home/yadhu07/Downloads/legacy-master/images/train -o /home/yadhu07/Downloads/legacy-master
             images/train.csv

   #### 2--csv to tf-record creation

       ---copy the path of the train.csv and test.csv file paste in the respective position of below command

       ---move to the legacy-master directory

       ---tf-record for traning can be creaeted using

              --->python generate_tfrecord.py --label1=human --csv_input=/home/cvlab2/tensorflow/workspace/training_demo
                  /annotations/train.csv --output_path=/home/cvlab2/tensorflow/workspace/training_demo/annotations/train.record
                  --img_path=/home/cvlab2/tensorflow/workspace/training_demo/images/train


       ---tf-record for testing can be creaeted using
              --> --->python generate_tfrecord.py --label1=human --csv_input=/home/cvlab2/tensorflow/workspace/training_demo
                  /annotations/test.csv --output_path=/home/cvlab2/tensorflow/workspace/training_demo/annotations/test.record
                  --img_path=/home/cvlab2/tensorflow/workspace/training_demo/images/test          
## PHASE 4:  TENSORFLOW AND OTHER DEPENDENCIES INSTALLATION

 Required [Python](https://www.python.org/download/releases/3.0/),[PIP](https://pip.pypa.io/en/stable/installing/)

 Further reading and details [Link](https://www.python.org/download/releases/3.0/)

  #### 1--installing required packages and dependencies      

         --->pip install tf-nightly
         --->pip install pillow
         --->pip install lxml
         --->pip install jupyter
         --->pip install matplotlib
         --->clone the github-------"git clone https://github.com/tensorflow/models.git"

  #### 2--Environment setup in UBUNTU 18.0.4

          --->protoc object_detection/protos/*.proto --python_out=.

          --->export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

          --->sudo ./configure

          --->sudo make check

          --->sudo make install


   #### 3---downloading and setting pretrained model

 Download faster_rcnn_resnet101_coco [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

        --->extract it to the '/home/yadhu07/models/research/tensorflow'

        --->copy the corresponding faster_rcnn_resnet101_coco.config file from '/home/yadhu07/models/research/object_detection

            /samples/configs'




## PHASE 5:  TRAINING

  --->Training the model and create the frozen output model file (forzen_graph.pb).
  --->  [Graph file](https://drive.google.com/open?id=16-YOOQs3lfrvkHWpT1aPH7qoPYAMSeTb)

      -->move to the models directory

       -->copy the path of train.record and test.record and copy it to the path in the following command

       -->Editing the cofiguration file

          -->add the path of train.record and test.record

          -->add the path of label.pbtxt

       -->python object_detection/legacy/train.py --train_dir=/home/yadhu07/models/research/tensorflow/train1
          --pipeline_config_path=/home/yadhu07/models/research/tensorflow/faster_rcnn_resnet101_coco.config

        -->In the final_model.ipynb file edits the following line of paths

             ----PATH_TO_CKPT = '/home/yadhu07/models/research/tensorflow/output/frozen_inference_graph.pb'

             ----PATH_TO_LABELS = '/home/yadhu07/models/research/tensorflow/labels.pbtxt'

 ## PHASE 6:  LOADING THE MODEL AND VALIDATION

    --->open the jupyter notebook and load the final_model.ipynb file

    ---->put the 1% of the 50000 images in the '/home/yadhu07/models/research/tensorflow/test' directory'(50 images)

    ---->Give the path of the  file in the final_model.ipynb

    ---->Run the file and evaluate result

 ## MORE RESULTS


![alt text](https://github.com/yadhukm07/human_detection/blob/master/index7.png)

![alt text](https://github.com/yadhukm07/human_detection/blob/master/index.png)
