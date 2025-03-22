# Label Studio Setup Scripts

This directory contains setup scripts for local Label Studio development.  
Running these scripts will set up the project for local development, although everything can also be done manually.

> To run these scripts, an API_KEY and LABEL_STUDIO_URL must be provided in the root `.env` file.

### Scripts:

- `create-project`:  
  Run this script to create a project with the appropriate view for the dart keypoint labeling.

- `connect-ml-backend`:  
  Run this script to connect the YOLO ML backend to the given project.