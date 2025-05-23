# Scored

## Single Camera automated darts scoring using keypoint detection

The project aims to detect dart scores by using a yolo v11 keypose model on a single camera image to predict the score of the thrown darts.
4 Keypoints on the dartboard, the darts tip and flight are detected.

![image](images/detected_keypoints.png)

After detecting the keypoints, the 4 dartboard keypoints are used to apply a perspective transform to warp the perspective in a top down view. With the warped points of the dart tips the score of the dart can easily be calculated by their distance to the center and their angle.

![image](images/warped_dartboard.png)

## How to use

1. Preferably create a virtual environment with conda, poetry or whatever you use.
2. Install the necessary requirements with `pip install -r requirements.txt`.
3. Install the module locally `pip install -e .`

## Training

To see how to train your own model and how to prepare and create a dataset, refer to [Train Instructions](train.md)