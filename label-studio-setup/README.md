# Label Studio Setup Scripts

This directory contains setup scripts for local Label Studio development.  
Running these scripts will set up the project for local development, although everything can also be done manually.

> To run these scripts, an API_KEY and LABEL_STUDIO_URL must be provided in the root `.env` file.

### Scripts:

- `project.py`:  
  Run this script to create a project thats correctly set up.
  The Script will add the labeling interface, connect the ml backend and the local storage.  
  Running the project with the `--sync` flag will sync all storage buckets with the latest updates

- `connect-ml-backend.py`:  
  Run this script to connect the YOLO ML backend to the given project.