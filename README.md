# Object-Tracking-PyTorch-YOLOv5-DeepSort

This project represents the implementation of YOLOv5 and DeepSort in PyTorch, forming a powerful object tracking system that enhances the capabilities of object detection and tracking in various applications. YOLOv5, short for "You Only Look Once version 5," is a state-of-the-art object detection model known for its speed and accuracy in identifying objects within images and video frames. It excels at real-time object detection, making it a popular choice in computer vision tasks.

However, while YOLOv5 is excellent at detecting objects, it doesn't inherently provide tracking capabilities. This is where DeepSort comes into play. DeepSort, or "Deep Simple Online and Realtime Tracking," is a tracking algorithm that works seamlessly with object detection models like YOLOv5. It employs deep learning techniques to assign unique IDs to detected objects and tracks them across consecutive frames of a video or image stream.

The reason for using DeepSort alongside YOLOv5 is to bridge the gap between object detection and tracking. YOLOv5 excels in detecting objects accurately and quickly, but it doesn't inherently maintain the identity of objects as they move across frames. DeepSort takes the detected objects and associates them with unique IDs, allowing for continuous tracking. This tracking information is crucial in applications where knowing not only what objects are in the scene but also where they are and how they move over time is essential.

__In essence, the combination of YOLOv5 and DeepSort in this project enables robust and real-time object detection and tracking. This is incredibly valuable in various domains, including surveillance, autonomous vehicles, retail analytics, and more, where understanding the trajectory and interactions of objects or individuals is crucial for decision-making and analysis. By harnessing the power of these two advanced deep learning techniques, this project offers a comprehensive solution for object monitoring and tracking in dynamic environments.__


<img src="img.gif" alt="workflow" width="100%" >


# Watch Demo:



### Setting up Project Environment :

This project is a separate project using Conda of Anaconda. It provides a clear structure for organizing your code and dependencies using a virtual environment. The project includes the following folders and files:

- YOLOv5_DeepSort_Tracking.ipynb: This folder is used to store Jupyter notebooks for data exploration or experimentation.
- runs/: This folder contains the video generate by running the project and get detection in video.
- environment/: This folder is used to store the virtual environment created by Conda.
- .gitignore: This file specifies which files and folders should be ignored by Git.


#### Getting Started

To set up the project, please follow the instructions below:

##### Prerequisites

- Anaconda or Miniconda should be installed on your system.
- Installation
- Clone the project repository:
```
git clone <repository_url>
cd <project_directory>
```

- Create a new Conda environment for the project:
```
conda create --name <env_name> python=3.9
```
- Activate the newly created environment:
```
conda activate <env_name>
```
- Install the project dependencies:

```
pip install -r requirements.txt
```
- Running the Application

Make sure you are in the project directory and the Conda environment is activated.

#### Additional Notes

- Having a separate project environment, such as the one created with Conda, offers several advantages:

- Dependency Isolation: By creating a virtual environment, you can install project-specific dependencies without interfering with the system-wide Python installation or other projects on your machine.

- Reproducibility: The project environment ensures that all project contributors are using the same set of dependencies, making it easier to reproduce and share the project's results.

- Version Control: By including the virtual environment in your project repository, you can easily recreate the same environment on different machines or after a fresh clone of the repository.

- Package Management: Using setup.py allows you to define the project's dependencies and make it easier to package and distribute your project as a Python package.

- By following these instructions and organizing your project with a separate environment, you can maintain a clean and reproducible development environment while keeping your project dependencies isolated and well-managed.

<br>


### Important Things to Notice:

In __detect.py__ , all the things remain same as in yolov5/detect.py , the main thing change in this code is applying __Deepsort Algorithm__ that help to detect each frame using __Kalman filter__ 
'''
  cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

'''

This code snippet configures and initializes the DeepSort tracking algorithm. Here's a breakdown of what it does:

- cfg = get_config(): This line obtains a configuration object (cfg) that likely contains various settings and parameters required for running the DeepSort algorithm. The get_config() function is likely defined elsewhere in the code to load or create these configurations.

- cfg.merge_from_file(opt.config_deepsort): This command merges additional configuration settings from a file specified by opt.config_deepsort. It allows for the dynamic adjustment of DeepSort's behavior by loading parameters from an external configuration file.

- deepsort = DeepSort(...): This line initializes the DeepSort tracker with various parameters:

- cfg.DEEPSORT.REID_CKPT: This is the path to a pre-trained model checkpoint used for re-identification of objects. Re-identification is crucial for ensuring that the - tracker can associate objects correctly across frames.
- max_dist: It sets the maximum distance threshold for associating detections with existing tracks.
- min_confidence: This parameter specifies the minimum confidence score required for a detection to be considered reliable and eligible for tracking.
- nms_max_overlap: It defines the maximum overlap allowed during non-maximum suppression (NMS). NMS is used to remove redundant or highly overlapping bounding boxes.
- max_iou_distance: This parameter sets the maximum intersection-over-union (IoU) distance allowed between objects for them to be considered the same in different frames.
- max_age: It determines the maximum number of frames that a track can be "lost" (i.e., not associated with any detection) before it is terminated.
- n_init: This specifies the number of frames an object must be detected in consecutively before it is considered a valid track.
- nn_budget: It sets the maximum number of nearest neighbors to consider during data association.
- use_cuda=True: This flag indicates that DeepSort should utilize GPU acceleration if available, which can significantly speed up the tracking process.

__In summary, this code is responsible for configuring and initializing the DeepSort tracker, setting various tracking parameters, and potentially loading a pre-trained model checkpoint for re-identification. DeepSort is a critical component in object tracking systems and helps maintain the continuity of object tracking across video frames.__

### AUTHOR
<hr>
<strong>Shehryar Gondal</strong>


You can get in touch with me on my LinkedIn Profile:<br>
 <a href = "https://linkedin.com/in/shehryar-gondal-data-analyst"><img src="https://img.icons8.com/fluent/48/000000/linkedin.png"/></a>

You can also follow my GitHub Profile to stay updated about my latest projects:<br>
<a href = "https://github.com/ShehryarGondal1"><img src="https://img.icons8.com/fluent/48/000000/github.png"/></a>


If you liked the repo then kindly support it by giving it a star ⭐.
