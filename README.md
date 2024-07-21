
Predicting Pedestrian Crossing Intention with Feature Fusion and Spatio-Temporal Attention

This Repo contain a reimplementation of the homonimus paper of Pedestrian Crossing Intention Prediction (https://arxiv.org/pdf/2104.05485) implemented on TensorFlow.

Our implementation use a slightly simplified version of the original model now based on Pythorch.

ABSTRACT

Predicting pedestrian behavior is a crucial prerequisite for deploying Automated Driving Systems
(ADS) in the real-world. 
The aim of the subject is to recognize the pedestrian crossing intention in real-time, with short time reaction especially in urban contex such as citis. 

Our work use deep neural network model with Rnn for extracting potentially usefull information of the sourranding contex, resolvin the problem of previous implementation by introducing also the temporal domain.

Regarding the imput of our model we made 2 principal branch: the non visual brach and the visual brach. 
The first one extract for each pedestrian in each frame of the video the pose and the bounding box. 
The other brach instead extract the semantic mask of the whole frame, named, global context and the cropped image, around the pedestrian's bounding box, of the pedestrian named Local context.

We fuse different feature such as sequences of RGB images, semantic segmentation masks, pose keypoints and bounding box in an optimum way using attention mechanisms and a stack of recurrent neural networks.

Extensive comparative experiments on the JAAD pedestrian action prediction benchmark demonstrate the effectiveness of the proposed implementation, where state-of-the-art performance was nearly achived.

DATASET

for the dataset we use the well-known JAAD dataset containing 346 video of different duration.
The dataset is available here https://data.nvision2.eecs.yorku.ca/JAAD_dataset/.

We have copied the jaad_data.py corresponding repositories into this project's root directory for managin the interface of the dataset.

In order to use the dataset in the correct way we use the function split_clips_to_frames.sh to convert video clips  into images (frames).

Above operation will create a folder called images and save the extracted images grouped by corresponding video ids in the ./JAAD/images folder.

After this basic operation we did a precise work of dataset balancig, because JAAD was severly unbalanced, leading to bias for our  model.

We use the already implemented function generate_database() to generate a custom dataset with less videos, around
60 to resolve the probelm of computation.


TRAINING

For the feature:

Global context: we have extracted the semantic mask with a pretrained deeplab model on cityscapes in order to have the best possible semantic mask we take the best model available suitable for our problem.
before the semantic mask computation we have resized the image to 512x512 for faster computation.
After the semantic mask computation we have resized the image to 224x224 for the input of the Vgg16 model.
To be more precise the model is DeepLAbV3Plus_mobilenet which can be found here : https://github.com/VainF/DeepLabV3Plus-Pytorch?tab=readme-ov-file

Local context: we just cropped the image on the BoundingBox coordinate and then resized on 224x224 dimension 
needed for the Vgg16 model used.

Pose keypoint: for the pose key point we used the pretrained model named openpose to get 18 keypoint in the for of 36 point.

Bounding box: the Bounding Box was taken directly from the Jaad dataset.



AUTHORS
Jacopo Tamarri, Filippo Croce

