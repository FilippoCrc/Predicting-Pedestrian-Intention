
Predicting Pedestrian Crossing Intention with Feature Fusion and Spatio-Temporal Attention

This repository contains a reimplementation of the paper "Predicting Pedestrian Crossing Intention with Feature Fusion and Spatio-Temporal Attention" https://arxiv.org/pdf/2104.05485, implemented using TensorFlow.

Our implementation use a slightly simplified version of the original model now based on Pythorch.

ABSTRACT

Predicting pedestrian behavior is crucial for deploying Automated Driving Systems (ADS) in real-world scenarios. This project aims to recognize pedestrian crossing intention in real-time, focusing on urban contexts where rapid reaction times are necessary. Our implementation uses a deep neural network model with RNNs to extract useful information from the surrounding context, incorporating the temporal domain to resolve issues present in previous implementations.

Regarding the imput of our model we made 2 principal branch: the non visual brach and the visual brach. 
The first one extract for each pedestrian in each frame of the video the pose and the bounding box. 
The other brach instead extract the semantic mask of the whole frame, named, global context and the cropped image, around the pedestrian's bounding box, of the pedestrian named Local context.

We fuse different feature such as sequences of RGB images, semantic segmentation masks, pose keypoints and bounding box in an optimum way using attention mechanisms and a stack of recurrent neural networks.

DATASET

We use the JAAD dataset, which contains 346 videos of varying lengths. The dataset is available here https://data.nvision2.eecs.yorku.ca/JAAD_dataset/.

We have copied the jaad_data.py corresponding repositories into this project's root directory for managin the interface of the dataset.

In order to use the dataset in the correct way we use the function split_clips_to_frames.sh to convert video clips  into images (frames).

Above operation will create a folder called images and save the extracted images grouped by corresponding video ids in the ./JAAD/images folder.

After this basic operation we did a precise work of dataset balancig, because JAAD was severly unbalanced, leading to bias for our model.

We use the already implemented function generate_database() to generate a custom balanced dataset with less videos, around 60 to resolve also the problem of computation.

Link to our sub-set : https://drive.google.com/file/d/1-89ibP96qLaDEpR6okq4TjJ6dTAUYYeN/view?usp=drive_link

TRAINING

Global Context:
1. Resize images to 512x512 for faster computation
2. Extract semantic masks using a pretrained DeepLabV3Plus model on Cityscapes.
3. Resize to 224x224 for input to the VGG16 model.
4. Model: DeepLabV3Plus_resnet101. which can be found here : https://github.com/VainF/DeepLabV3Plus-Pytorch?tab=readme-ov-file

Local Context:
1. Crop images based on bounding box coordinates and resize them to 224x224 for the VGG16 model.

Pose Keypoints:
1. Use the pretrained OpenPose model to extract 18 keypoints, resulting in a 36-point vector.
    The openpose model is available at https://github.com/Hzzone/pytorch-openpose

Bounding Box:
1. Extract bounding box data directly from the JAAD dataset.


AUTHORS
Jacopo Tamarri
Filippo Croce


