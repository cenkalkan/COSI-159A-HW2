# SphereFace Implementation for Face Recognition

## Purpose

The goal of this assignment is to implement and understand the SphereFace algorithm, a deep learning model for face recognition based on deep hypersphere embedding. The model utilizes A-Softmax Loss to enhance the discriminative power of the deeply learned features.

This implementation is tested on the Labeled Faces in the Wild (LFW) dataset, which is widely used for studying unconstrained face recognition.


## Setting Up the Environment

To run the code, I set up a Conda environment 
and installed the necessary libraries to execute
the code. Follow these steps:

1. **Create a Virtual Environment**:
   - Open a terminal or command prompt.
   - Navigate to your project directory.
   - Run the command to create a virtual environment named `cosi159a`:

     ```
     conda create -yn cosi159a python=3.11
     ```

2. **Activate the Virtual Environment**:
 
     ```
     conda activate cosi159a
     ```


3. **Install Required Packages**:
   - Ensure you have Python already installed.
   - Install PyTorch and torchvision 
   by running the following within the desired
   environment. Note that the environment has to
   be activated.

     ```
     conda install pytorch torchvision -c pytorch
     ```
   - Ensure you have `pip` already installed.
   - Install Flask by running the following within 
   the desired environment. Note that the environment
   has to be activated.
      ```
     pip install Flask
     ```
   - Install scikit- learn by running the following within 
   the desired environment. Note that the environment
   has to be activated.
   ```
     pip install scikit-learn
     ```
   
## Data Preparation

Before running the code, ensure the LFW dataset and the pairs files are correctly retrieved. This can be done at the website:  http://vis-www.cs.umass.edu/lfw/index.html

1. Download the LFW dataset and extract it. The folder should be named 'lfw' under the download link 'All images as gzipped tar file' and contain subfolders representing each individual's name containing their face images.

2. Obtain `pairsDevTrain.txt` and `pairsDevTest.txt` files that contain pairs of images for training and testing respectively.

3. Place the 'lfw' folder and the two pairs files in a directory, for example, `/Users/yourname/Downloads/COSI-159A-HW2/`. Ensure that the paths in the program match these locations.

## Updating Paths in the Program

To incorporate the correct paths into the program, modify the `parse_args` function in your Python code:

- For `--train_file`, set the default argument as the path to your `pairsDevTrain.txt` file.
- For `--eval_file`, set the default argument as the path to your `pairsDevTest.txt` file.
- For `--img_folder`, set the default argument as the path to your 'lfw' folder.

## Running the Program

1. Navigate to the root directory of the project.
2. Run the program using the following command:

     
     python main.py 

## Notes
- Default settings use GPU 0. Ensure your CUDA environment is configured correctly.
- Modify hyperparameters in parse_args as necessary to experiment with different settings.


