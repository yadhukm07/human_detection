# Human Detection from Drone Images using Deep Learning

![alt text](https://github.com/yadhukm07/face_mood/blob/master/index2.png)

Natural disasters across the world, many of them related to the climate, can cause massive loss of life and property. 
Disasters may cause in human loss, property damage or a profound environmental effect. 
In situation of floods, it is difficult to find affected people or trapped people. Moreover, humans are highly vulnerable to dynamical events such as camera jitter, variations and illumination changes in object sizes.
It is difficult to detect humans in these situations as less signs would be available for help. 

We were able to generate a model(convolutional neural network) which could be
used to clearly detect and locate(label) humans from images(jpg).

The detection of humans from the image captured by low altitude drones is done by using a deep learning model with convolutional nueral network. The final model is developed by using 57000 images which is trained on ResNet101, which is a faster-rcnn model.

<!-- Prerequisites:::[Nueral netwoks](http://neuralnetworksanddeeplearning.com/chap1.html),[Convolutional Nueral NEtwork](https://skymind.ai/wiki/convolutional-network) -->

## How to use?
The developed model can be used for many purposes including detecting Humans from aeriel images and surviallance purposes.
The model shows better results over images captured by the drone from low altitudes. The Usage of the model is explained below
 
1. To run the model, some libraries and package are necessary. To install these Just run the bash file in Terminal using the following Command-

          chmod +x install.sh

         ./install.sh

  2. Clone the github repository into the local system by using the below command-

          git clone https://github.com/yadhukm07/human_detection.git

  3. Put the images for deetction in the test folder-  

          human_detection/My_Model/Test

  4. Open the terminal from the My_Model folder and run the jupyter notebook in the terminal-

            jupter notebook

            run My_Model.ipynb




## Generating Model from Dataset

Generation of the model can be achieved through 5 phases. Overview regarding
which is described below, followed by necessary details:

####  1.Data Set Collection(Image Collection)
Dataset plays a vital role for a Machine learning/Deep Learning model. We have to teach the machine how a human looks by giving images that contain the humans. Collection of images that contain ariel view of human is done in this phase.


#### 2 . Data Set Preprocessing and Data Set Annottation(Labelling)
The collected images can be used for training and testing. In order to feed the images to the model, these images need to be prepared so that the model can get information from the images.

####     3.  Tf-record Creation
Tf tecord is the Final file created from the dataset that is used to train the model.

#### 4. Installing and deveoloping tensorflow model,installing all the required dependencies and libraries
To do all the works(training and testing) there is a need of bunch of libraries, dependencies and a framewok. The installation and setup of all these are done in this phase.

#### 5. Training and Testing using the dataset on the model
The final training procedure and code are explained in this phase.

#### 6. Testing with the real examples and retraing by changing various parameters
The developed model can be validated by testing with real examples and images that are not in the training and testing dataset.

## PHASE 1:DATA SET COLLECTION

For better accuracy and result the model have to be trained with a well defined dataset. Collecting the data set is the first and major work. More the number of images, we can attain better accuracy for the model. We collected nearly 57000 images. 90% of the images constitutes the UAV123 dataset which is a publicly available dataset. Rest of the images were collected from google and youtube and other various online sources.

The UAV123 dataset is a dataset which is provided By the "King Abdullah University for Science and Research ". The dataset contains a total of 123 video sequences and more than 110K frames making it the second largest object tracking dataset after ALOV300++. The dataset is updated constantly. At latest, the dataset has grown to 1,13,766 images which includes annotated images with 6
object classes out of which only human class is required, that consists of 50,000 images used for
training and testing.

The source of all images that we used are provide below

Source


   1. [UAV123]--[link](http://neuralnetworksanddeeplearning.com/chap1.html)

   2. [Kerala flood Ariel images]--[Example link](http://neuralnetworksanddeeplearning.com/chap1.html)

   3. [Flood Ariel images]--[sample image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8eX5lCtfdKNb4OHancP85PjDtlqIHdsz5rUGNsU2SMWi7qJA2)

   4. [Screenshots  from Drone videos of humans from youtube]--[sample video](https://youtu.be/2JuSrDF4bmo)


All the collected image that we used are available in the google drive of which the link is provided below-

- [Final Data set](https://drive.google.com/open?id=1oAzlGOaeC575patIbAZEIPoE3T6NM89)



## PHASE 2: DATA SET PREPROCESSING AND ANNOTTATION

For training the model tf-record is required which is build from the dataset. The image along with its xml file is used for creting csv file from which tf-record is created.

We need to build xml file for each image collected. That is we require xml file for each 57000 images. An xml file for an image in our case is the file that contain the cordinates of postion where human are preseted in the image.

A bounding box is drawn for each humans presented in the image. The xml file contain the 4 co-cordinates of the drawn rectangle(Bounding Box)

The Final tf record for taining the model can be build through the series of steps explained there.

The xml file is created for each of the image. A csv file is created from the original image and their corresponding xml file. Seperate csv file is made for test and train dataset. From the 2 csv file the tf-record is made.

The steps for creation of xml file corresponding each image are explained below-

#### 1--Making all image to a Standard Resolution(1920*720)
The collected images across the various resources are in different resolution.The images from the UAV123 Dataset comes with a resolution 1920*720. So all the rest of the images are made into the same resolution.

To make the all images to the same resolution:

  - copy all iamges to a folder rezize
        /human_detection/Rezize

  -  Run the rezie.py code in the terminal

           python rezize.py
Now all the images will be in the resolution 1920*720.

#### 2--Rename the all rezized images in serial number
For conveniance it is better to name the image serially from 1 to 57000.

example:: img (1), img (2).

Follow this youtube Video to rename the images
    -
    [Serial_Renaming](https://youtu.be/5X8OdurpYyM)
    
#### 3-Annotating/labelling the image--->Using labelimage tool
Annotation or labelling is the process of labelling humans in an image and respective xml file is created.


Bounding boxes across the humanas are drawn and the cordinates postion are saved in the xml file.


The documentation is available in this [link](https://github.com/tzutalin/labelImg.git)

For our model and requirement we have done the following steps

  - Clone the labelImg git-hub repository

        git clone https://github.com/tzutalin/labelImg.git

  - Run the following Bash file

        chmod +x install.sh

        ./label.sh

  - Naming the classes
   Since we are going to label only humans we need to modify the class name. For this modify the content of 'predifined_classes.txt'.

   Remove all other label name and add label name 'human'
   - To open the label image tool run the following Command

          python labelimage.py



  In the LabelImg graphical image annotation tool select the directory where the images are kept which are to be labelled.
  Select the folder where the annotated file is to be saved. We need pascalVOC format for the xml file. So choose the pascalVOC in the     labelImg Tool. Do the labelling for all the images and we will get the xml file for each image.




- Converting the already annotted text file to the xml format


  Some dataset that are available on internet like UAV123 comes along with the annnotted file. That is the data set include images         and their annotted file.

  There are 20 object classes and their annotated file in that dataset. We have to take out the images of humans and their corresponding   annotated file.


  The annotated file is in txt file format. We have to Convert this file to the xml file format suitable for our Training

  Follow this youtube video for the conversion [link](https://youtu.be/dqNwpIRBOrA)

##  PHASE 3:TF RECORD CREATION

Tf-record is the final file created from the dataset for the training. The Tf-record file is created From 2 CSV files. The csv file can be prepared from the original images and their respective xml files.

We need two csv files. From the total dataset(Images and Xml file) 75% are used for training and 25% for Testing

75% of the total data that is 37500 images and their xml file are used for creating Train.csv files

25% of the total data that is 15000 images and their xml file are used for creating Test.csv files


The creation of csv file from the images and xml files is explained further-

  #### 1. xml to csv conversion

  The 35000 images and their xml files are copied into the "TF record creation/images/Train" folder

  The 15000 images and their xml files are copied into the "TF record creation/images/Test" folder

  Then run the folllowing Command


  - generation of Train.csv, run the following command in Terminal

        python xml_to_csv.py -i /home/yadhu07/Downloads/legacy-master/images/train -o /home/yadhu07/Downloads/legacy-master/
        images/train.csv

  - Generation of Test.csv, run the following command      

         python xml_to_csv.py -i /home/yadhu07/Downloads/legacy-master/images/train -o /home/yadhu07/Downloads/legacy-master
         images/train.csv


 #### 2.Csv to tf-record creation

The csv files for both the test and train data are now generated. Using this csv file we have to make the Tf-record.

The tf-record from both train.csv and Test.csv_input is created.

- To generate the test.record(tf-record for train data) Run the following command in the terminal:

      python generate_tfrecord.py --label1=human --csv_input=/human_detection/TF record creation
      /CSV/train.csv --output_path=/human_detection/TF record creation/TF-record/train.record
      --img_path=/human_detection/TF record creation/images/train


- To generate the test.record(tf-record for train data) Run the following command in the terminal:

      python generate_tfrecord.py --label1=human --csv_input=/human_detection/TF record creation
      /CSV/train.csv --output_path=/human_detection/TF record creation/TF-record/test.record
      --img_path=/human_detection/TF record creation/images/test

After running these two commands the tf-records(train.recoerd, test.record ) is generated. This record files can be used for final training. For this a bunch of libraries and dependencies are to be installed. The environment setup and installation are discussed below:

## PHASE 4: TENSORFLOW AND OTHER DEPENDENCIES INSTALLATION

Tensorflow framework is most important to develop and run our model. Before installing tensorflow some other packages have to be installed. The required packages are python and pip.

The documentation for the installation is given below:



  [Python](https://www.python.org/download/releases/3.0/),[PIP](https://pip.pypa.io/en/stable/installing/)

 The installation of tensorflow and associated libraries are documented [Here](https://www.python.org/download/releases/3.0/)

 The same is explained here:

  #### 1--installing required packages and dependencies .
  - Tensorflow Installation    
  
             pip install tf-nightly
             
  - Other dependencies/packages can be installed by running the following command in terminal-

            chmod +x install.sh

            ./required_modules.sh


  #### 2--Environment setup in UBUNTU 18.0.4

  For the path setup and environment setup in Ubuntu, run the following command in the Terminal:

          protoc object_detection/protos/*.proto --python_out=.

          export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

          sudo ./configure

          sudo make check

          sudo make install


 #### 3-Downloading and setting pretrained model

 We are training our model in a pretrained network. The models which are trained on a very large dataset are used to train again with 
 custom dataset to get better prediction for problem specific Domain.

 In our case the problem is to detect isolated humans who are trapped in cases such as flood and other natural calamities.

 So we choose a pretrained model faster_rcnn_resnet101_coco model to train over our dataset.

 This pretrained model which when again trained over our dataset gives better result over real time examples.

There are many pretrained model available which are developed by researchers.

The models can be downloaded from following link:

[ Download faster_rcnn_resnet101_coco ](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

The downloaded model is available in the "/human_detection/My_Model.


## PHASE 5:  TRAINING

Once the necessory installation and tf-records are created the next step is TRAINING.

The trainig yields the final model(frozen_graph.pb). Before training, their are two steps to be completed that are-

#### 1- Setting up configuration files

   In the configuration file the path of the trained record and path record are modified. Then the path of label.pbtxt is added. The
   label.pbtxt contains the label name which is to be displyed when the humans are detected.

####  2-Trainng the models

Run the following command on the terminal

      cd human_detection/models/research/object_detection/legacy

      python train.py --train_dir=//media/yadhu07/New Volume/human_detection/My_Model
      --pipeline_config_path=/human_detection/My_Model/faster_rcnn_resnet101_coco.config


 ## PHASE 6:  LOADING THE MODEL AND VALIDATION
 
The final model is generated as 'frozen_graph.pb' now this model is to be loaded to detect humans from the images.

The code for loading the model are written in the jupyter file 'My_Model.ipynb'.

The path to the final model and the label file are edited in the My-Model.ipynb file.

      ----PATH_TO_CKPT = 'human_detection/My_Model/My_Model.ipynb'

      ----PATH_TO_LABELS = 'human_detection/My_Model/labels.pbtxt'

  Run the following command to open jupyter notebook:

       jupyter notebook


Put the images from which the humans are to be detected in the '/human_detection/detection/detection'


Restart and run all the cells in jupyter notebook.

Evaluate the RESULTS.
GOOD LUCK!!!!

 ## MORE RESULTS


![alt text](https://github.com/yadhukm07/face_mood/blob/master/index7.png)

![alt text](https://github.com/yadhukm07/face_mood/blob/master/index.png)
