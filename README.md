# Scored
**Single Camera automated darts scoring using keypoint detection**

> Currently under development, this project is a work in progress. Please check back later for updates.

## Scored GUI
A simple python tkinter based GUI for collecting data, playing dart legs and testing inferences of the model. The GUI is located in the `gui` folder. For more information on how to use the GUI, please refer to the [GUI README](gui/README.md).

## Data Collection

For data collection, you can use your own setup and label the data or use the **Scored GUI** to collect and label the data in one step while following the correct labeling structure that is need for the learning process. For more information on how to use the GUI, please refer to the [GUI README data collection](gui/README.md#data_collection).

## Logging

Logging is set up in `scored_lib.logging_setup`. Call `setup_logging()` once at
startup and then use `import logging` anywhere:

```python
from scored_lib.logging_setup import setup_logging

setup_logging()                      # ./logs/<start time>/scored.log + stderr
setup_logging("logs", "my.log", "DEBUG")
```

Records go to a rotating file (`2 MB` x `5`) and to the console. The level can
be changed at runtime with `set_log_level("DEBUG")`.

## Training

WIP
