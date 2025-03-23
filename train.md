# Creating your own model

This guide will walk you trough on how to prepare and create a dataset as well as how to train your own model

## Data preparation

If you have already collected images of a dartboard and the darts and you want to train a model on them or evaluate an existing one, the images have to be annotated.

This project uses the open source labeling software LabelStudio. Locally running the LabelStudio instance is preferred an can easily be achieved by running the docker-compose in this project with `docker compose up`.
This will spin up 2 Containers. One for running the Label studio software locally (default to `localhost:8080`) and one for an LabelStudio ml backend to run prediction for pre labeling new data.

After running the docker compose you should be able to access the LS in your browser. First you need to sign in to an account. Either create an account with any email and password. No need to provide a functional email or use the default user with email and password : amin@admin.com.  
After authenticating you can create a project manually or run `python label-studio-setup/create-project.py` to create a project with the necessary labeling config.

>For running the scripts in the `label-studio-setup` folder you need to define the env variables for the LS URL and an API Key which can retrieved by opening LS, navigating to your account settings, Access Token. 

### LS ml backend

To speed up the labeling process LS provided support for an ml backend which runs a model on the images and pre-annotates the images.
The annotator now only needs to correct the images with bad predictions.
To uses this run `python label-studio-setup/connect-ml-backend.py` which will automatically connect the ml backend running on another container to your project.
  
By default the ml backend expects a model in the `models/` which it will use to make predictions. Which model it will use can be configured in the Projects Labeling interface. Change the `model_path` value to any model relative to the `models/` folder.


## Creating a Dataset

After labeling the images we need to create a dataset on which the model can be trained on. Since this project uses YOLO, we need to follow the yolo dataset structure. Sadly LS only supports a limited number of export formats we have to export the data in the LS JSON format and use the conversion script included in this project to transform the JSON annotations into YOLO format.
This can be done by running the `format_data` script like this in the root directory of this project:

` python -m src.preparation.format_data --keypoint-order top right bottom left --keypoint-order dart dart-flight --classes dartboard dart --split .7 .2 .1 --clear-existing <path_to_exported_annotations.json>`

> Some of the parameters have to be changed if the LS view is defined with different keypoint or class names as well if you want different splits or outputs. Refer to `python -m src.preparation.format_data --help` for more information about the parameters.

## Training a model

Training the model on the dataset requires computational power.
If you don't have a dedicated GPU it is recommended to use Google Colab to train your model, which provide a free tier for accessing GPU's.
A dedicated [notebook](src/train/colab_train.ipynb) for the training on Google Colab can be used. Walk through the notebook to get more insights about the training process.