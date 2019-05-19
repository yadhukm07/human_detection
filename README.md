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

## How to use?
The developed model can be used for many purposes including detecting persons from ariel image and surviallance purposes.The model show beter result over images captured by the drone images captured from the low altitudes.Usage of the model is explained below
  1. To run the model some libraries and package are necessory.To install these Just run the bash file in Terminal using the following Command

          chmod +x install.sh

         ./install.sh

  2. Clone the github repository into the local system by using the below command

          git clone https://github.com/yadhukm07/human_detection.git

  3. Put the images for deetction in the test folder    

          human_detection/My_Model/Test

  4. open the terminal from the My_Model folder and run the jupyter notebook in the terminal

            jupter notebook

            run My_Model.ipynb




## Generating Model from Dataset

Generation of the model can be achieved through 5 phases. Overview regarding
which is described below, followed by necessary details.

####  1.Data Set Collection(Image Collection)
Dataset is the primary thing for a machine learning/Deep Learning model.we have to teach the machine how a human looks by giving images that contain the humans.Collection of  images that contain ariel view of human is done in this phase


#### 2 . Data Set Preprocessing and Data Set Annottation(Labelling)
The collected images can be used for training and testing.In order to feed the images to the model these images have to prepared so that the model can get information from the image.

####     3.  Tf-record Creation
Tf tecord is the Final file created from the dataset that is used to train the model

#### 4. Installing and deveoloping tensorflow model,installing all the required dependencies and libraries
To do all the works(training and testing) there is a need of bunch of libraries,dependencies and a framewok.The installation and setup of all these are done in this phase

#### 5. Training and Testing using the dataset on the model
The final training procedure and code are explained in this phase

#### 6. Testing with the real examples and retraing by changing various parameters
The developed model can be validated by testing with real examples and images that are not in the training and testing dataset

## PHASE 1:DATA SET COLLECTION

For better accuracy and result the model have to be trained with a well defined dataset. collecting the data set is the first and major work.More the number of images we can attain better accuracy for the model.we collected nearly 57000 images.90% of the images are from the UAV123 dataset which is a publically available dataset.Rest of the images are collected from  the google and youtube and other various online sources.

The UAV123 dataset is a dataset which is provided By the "king abdullah university for sciece and research ".The dataset contains a total of 123 video sequences and more than 110K frames making it the second largest object tracking dataset after ALOV300++.The dataset are updated constantly.At latest, the dataset has grown to 1,13,766 images which includes annotated images with 6
object classes out of which only human class is required that consists of 50,000 images used for
training and testing.

The source of all images that we used are provide below

Source


   1. [UAV123]--[link](http://neuralnetworksanddeeplearning.com/chap1.html)

   2. [Kerala flood Ariel images]--[Example link](http://neuralnetworksanddeeplearning.com/chap1.html)

   3. [Flood Ariel images]--[sample image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8eX5lCtfdKNb4OHancP85PjDtlqIHdsz5rUGNsU2SMWi7qJA2)

   4. [Screenshots  from Drone videos of humans from youtube]--[sample video](https://youtu.be/2JuSrDF4bmo)


The all collected image that we used are available in our  google drive the link is provide below

- [Final Data set](https://drive.google.com/open?id=1oAzlGOaeC575patIbAZEIPoE3T6NM89)



## PHASE 2: DATA SET PREPROCESSING AND ANNOTTATION

For training the model required tf-record which is build from the dataset.The image along with its xml file  is used for creting csv file from which tf-record is created.

we need to build xml file for each image collected.That is we require xml file for each 57000 images.An xml file for an image in our case is the file that contain the cordinates of postion where human are preseted in the image.

A bounding box is drawn for each humans presented in the image.The xml file contain the 4 co-cordinates of the drawn rectangle(bounding box)

The Final tf record for taining the model can be build through the series of steps explained there.

xml file created for each of the image.A CSV file created from The images and their corresponding xml file.seperate csv file is made for test and train dataset.From the 2 CSV file the tf-record can be made.

The steps for creation of xml file corresponding each image are explained below

#### 1--making all image to a standard resolution(1920*720)
The collected images across the various resources are in different resolution.The images from the UAV123 Dataset comes with a resolution 1920*720.So all the rest of the images are made into the
same resolution.

To make the all image in to the same resolution

  - copy all iamges to a folder rezize
        /human_detection/Rezize

  -  Run the rezie.py code in the terminal

           python rezize.py
Now all the image will be in the resolution 1920*720

#### 2--Rename the all rezized images in serial number
for conveniance it is better to name the image serially.from 1 to 57000.

example:: img (1), img (2).

Follow this youtube Video to rename the images
    -
    [Serial_Renaming](https://youtu.be/5X8OdurpYyM)
#### 3-Annottate/labelling the image--->using labelimage tool
Annottation or labelling is the process of labelling humans in an image and respective xml file is created.


Bounding boxes across the humanas are drawn and the cordinates postion are saved in the xml file.


The documentation are available in this [link](https://github.com/tzutalin/labelImg.git)

For our model and requirement we have done the following steps

  - clone the labelImg git-hub repository

        git clone https://github.com/tzutalin/labelImg.git

  - Run the following Bash file

        chmod +x install.sh

        ./label.sh

  - Naming the classess
   Since we are going to label only humans we need to modify the class name.For this modify the content of 'predifined_classess.txt'.

   remove all other label name and add label name 'human'
   - To open the label image tool run the following Command

          python labelimage.py



  In the LabelImg graphical image annotation tool select the directory where the images are kept which are to be labelled.
  Select the folder where the annotted file is to be saved.We need pascalVOC format for the xml file.So choose the pascalVOC in the labelImg Tool.Do the labelling for all the image and we will get the xml file for each image.




- converting the already annotted text file to the xml format


  Some dataset that available in the internet like UAV123 comes along with the annnotted file.That is the data set include images and their annotted file.

 There are 20 object classes and their annotted file in that dataset.we have to take out the images of humans and their corresponding anooted file.


  The annotted file is in txt file format,We have to  Convert this file to the xml file format suitable for our Training

  Follow this youtube video for the conversion [link](https://youtu.be/dqNwpIRBOrA)

##  PHASE 3:TF RECORD CREATION

Tf-record is the final file created from the dataset for the training.The TF-record file is created From 2 CSV files.The csv file can be prepare from the images and xml files

we need two csv files.From the total dataset(Images and Xml file) 75% are used for training and 25% for Testing

75% of the total data that is 37500 images and their xml file are used for creating Train.csv files

25% of the total data that is 15000 images and their xml file are used for creating Test.csv files


The csv file from the images and xml files can be created using below step

  #### 1. xml to csv conversion

  The 35000 images and their xml files are copied into the "TF record creation/images/Train" folder

  The 15000 images and their xml files are copied into the "TF record creation/images/Test" folder

  Then run the folllowing Command


  - generation of Train.csv ,Run the following command in Terminal

        python xml_to_csv.py -i /home/yadhu07/Downloads/legacy-master/images/train -o /home/yadhu07/Downloads/legacy-master/
        images/train.csv

  - Generation of Test.csv,Run the following command      

         python xml_to_csv.py -i /home/yadhu07/Downloads/legacy-master/images/train -o /home/yadhu07/Downloads/legacy-master
         images/train.csv


 #### 2.Csv to tf-record creation

The csv files for both the test and train data are now generated.Using this csv file we have to make the Tf-record

The tf-record from both train.csv and Test.csv_input

- To generate the test.record(tf-record for train data) Run the following command in the terminal

      python generate_tfrecord.py --label1=human --csv_input=/human_detection/TF record creation
      /CSV/train.csv --output_path=/human_detection/TF record creation/TF-record/train.record
      --img_path=/human_detection/TF record creation/images/train


- To generate the test.record(tf-record for train data) Run the following command in the terminal

      python generate_tfrecord.py --label1=human --csv_input=/human_detection/TF record creation
      /CSV/train.csv --output_path=/human_detection/TF record creation/TF-record/test.record
      --img_path=/human_detection/TF record creation/images/test

After running these two command the tf-records(train.recoerd ,test.record ) are generated.This record files can be used for final training.For This a bunch of libraries and dependencies are to be installed.The environment setup and installation are discussed below
## PHASE 4:  TENSORFLOW AND OTHER DEPENDENCIES INSTALLATION

Tensorflow framework is the most important to develop and run our model.Before installing tensorflow.Some other packages have to be installed.The required packages are python and pip.

The documentation for the installation are Given below



  [Python](https://www.python.org/download/releases/3.0/),[PIP](https://pip.pypa.io/en/stable/installing/)

 The installation of tensorflow and associated libraries are documented [Here](https://www.python.org/download/releases/3.0/)

 The same are explained here

  #### 1--installing required packages and dependencies .
  - Tensorflow Installation     
             pip install tf-nightly
  - Other dependencies/packages can be installed by running following Command in Terminal

            chmod +x install.sh

            ./required_modules.sh


  #### 2--Environment setup in UBUNTU 18.0.4

  The path setup and environment setup in the ubuntu run the following command in the Terminal

          protoc object_detection/protos/*.proto --python_out=.

          export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

          sudo ./configure

          sudo make check

          sudo make install


 #### 3-Downloading and setting pretrained model

 We are Traing our model in an pretrained model.The models which are trained on a very large dataset are used to train again with custom dataset for To get better prediction for problem specific Domain.

 In our case the problem is to detect isolated humans who are trapped in cases such as flood and other natural calamities

 so we choose a pretrained model faster_rcnn_resnet101_coco model to train over our dataset.

 This pretrained model which when again trained over our dataset gives better result over real time examples

There are many pretrained model available which are developed by researchers

The models can be downloaded from folllowing link

[ Download faster_rcnn_resnet101_coco ](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

The downloaded model are available in the "/human_detection/My_Model


## PHASE 5:  TRAINING

Once the necessory installation and tf-records are created the next step is TRAINING

The trainig yields the final model(frozen_graph.pb).Befor training there are two steps to be completed they are  
  #### 1- Setting up configuration files

   In the configuration file the path of the train record and path record are modified.Then the path of label.pbtxt are added.The label.pbtxt contains the label name which is to be displyed when the humans are detected

####  2-Trainng the models

Run the following command on the terminal

      cd human_detection/models/research/object_detection/legacy

      python train.py --train_dir=//media/yadhu07/New Volume/human_detection/My_Model
      --pipeline_config_path=/human_detection/My_Model/faster_rcnn_resnet101_coco.config


 ## PHASE 6:  LOADING THE MODEL AND VALIDATION
The final model is generated as 'frozen_graph.pb' now this model is to be loaded to detect humans from the images

The code for loading the model are written in the jupyter file 'My_Model.ipynb'

The path to the final model and the label file are edited in the My-Model.ipynb fi;e

      ----PATH_TO_CKPT = 'human_detection/My_Model/My_Model.ipynb'

      ----PATH_TO_LABELS = 'human_detection/My_Model/labels.pbtxt'

  Run the following command to open jupyter notebook

       jupyter notebook


put the images from which the humans to be detected in the '/human_detection/detection/detection'


Restart and run all the cells in jupyter notebook

Evaluate the RESULTS

 ## MORE RESULTS


![alt text](https://github.com/yadhukm07/human_detection/blob/master/index7.png)

![alt text](https://github.com/yadhukm07/human_detection/blob/master/index.png)
