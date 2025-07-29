# Adult-Child-Tracker
### Project Overview
This project aims to develop an advanced person detection and tracking system focusing on two specific categories: children and adults. The system is designed to assign unique IDs to detected individuals, track their movement across video frames, and handle scenarios such as re-entries, occlusions, and new appearances. The main goal is to apply this system for monitoring and understanding the behavior and interactions of children with Autism Spectrum Disorder (ASD) and therapists, assisting in the development of personalized treatment plans.

### Key Features
+ **Person Detection**: Detect and classify individuals in a video as either children or adults using YOLOv9.
+ **Unique ID Assignment**: Assign unique IDs to persons and track them across the entire video.
+ **Re-entry Tracking**: Track individuals when they leave and re-enter the video frame, retaining their original ID.
+ **New ID Assignment**: Assign a new ID when a previously unseen individual enters the frame.
+ **Post-Occlusion Tracking**: Continue tracking and correctly assigning IDs even after occlusions or when individuals become partially visible.

### Technologies Used
+ **[YOLOv9](https://github.com/WongKinYiu/yolov9)**: Used for person detection (specifically child and adult).
+ **[StrongSORT](https://github.com/TheNobody-12/MOT_WITH_YOLOV9_STRONG_SORT/tree/main)** Tracker: Utilized for object tracking. 
+ **Re-Identification (ReID)**: Incorporated using StrongSORT for person re-identification, ensuring accurate tracking through re-entries and occlusions.

### Project Structure
+ **Model Training**: YOLOv9 is used to detect children and adults in the video. The model was trained on a custom dataset ([link](https://app.roboflow.com/projects-20qd2/adult-child/3)), fine-tuned to differentiate between these two categories. Model performance was optimized using various configurations and hyperparameters.
+ **Tracking System**: StrongSORT, a state-of-the-art object tracker, is implemented to maintain person identities throughout the video. This tracker ensures that unique IDs are assigned and retained even during complex scenarios like re-entries and occlusions.
+ **ReID Module**: StrongSORT is enhanced by a ReID module that improves the system’s ability to recognize individuals who re-enter the frame or are temporarily occluded.


### Usage Instructions
+ ##### Prerequisites
    Before running this project, make sure you have following dependencies -
    * Python 3.x
    * pytubefix
    * gdown
    * ffmpeg

+ ##### Running the Project
    1. ```git clone https://github.com/shag527/Adult-Child-Tracker.git```
    2. ```cd Adult-Child-Tracker```
    3. ```python3 -m venv name_of_env```
    4. ```source name_of_env/bin/activate```
    5. Add the list of youtube videos to test_list.txt
         + https://www.youtube.com/example1
         + https://www.youtube.com/example2
    6. ```python3 track.py```
    7. Results will be stored in runs/track/exp(i) folder of MOT_WITH_YOLOV9_STRONG_SORT

  
  + ###### Command-Line Arguments 
  1. ```--links_file```: Path to a text file containing YouTube video URLs (default: Test_list.txt in the script's directory).
  2. ```--output_dir```: Directory where the downloaded videos and outputs will be saved (default: runs/track/exp in the MOT_WITH_YOLOV9_STRONG_SORT directory).
  3. ```--yolo_weights_path```: Path to the YOLO weights file (default: best.pt in the weights folder of the MOT_WITH_YOLOV9_STRONG_SORT).
  4. ```--strong_sort_weights_path```: Path to the StrongSORT weights file (default: osnet_ain_x1_0_imagenet.pt in the Models folder).
  5. ```--device```: Optional argument to specify which device to run the tracker on (e.g., 0 for GPU or leave empty for CPU).

### Logic Behind Model Prediction Analysis
+ Detection Phase:
    * Each frame of the video is passed through the YOLOv9 model to detect and classify objects as either children or adults.
    * The confidence threshold is set to ensure accurate detections, reducing false positives.
    * Why Yolov9?
>>>> I chose YOLOv9 for person detection due to its superior performance in balancing accuracy and efficiency. Traditional deep learning methods often face challenges with information loss as data undergoes multiple layers of transformation. YOLOv9 addresses this by introducing the concept of Programmable Gradient Information (PGI), which helps retain critical input information throughout the network. YOLOv9's architecture, Generalized Efficient Layer Aggregation Network (GELAN), enhances parameter utilization and reduces computational cost, which is crucial for real-time applications like person detection. Despite using fewer parameters and less computational power than previous versions like YOLOv8, YOLOv9 achieves better accuracy (up to 1.7% higher AP on MS COCO).

+ Tracking and ID Assignment:
    * StrongSORT is used to assign a unique ID to each detected person.
    * The tracker generates trajectories based on the movement of individuals across frames.
    * During the tracking process, new IDs are assigned only when an individual enters the frame for the first time.
    * Why StrongSORT?
>>>> For tracking, I integrated StrongSORT, which builds on the proven tracking-by-detection paradigm of DeepSORT but with significant upgrades in both appearance and motion tracking. StrongSORT uses a more powerful appearance feature extractor, BoT, combined with ResNeSt50, pre-trained on DukeMTMC-reID for more discriminative embeddings. This upgrade enhances re-identification accuracy, allowing the model to track persons effectively even after re-entries or occlusions. To further improve tracking, StrongSORT incorporates camera motion compensation using ECC and adapts to low-quality detections with a modified Kalman filter that adjusts noise covariance based on detection confidence. StrongSORT also solves the assignment problem using both appearance and motion data, ensuring more robust matching of tracks to unique persons in complex scenarios.

+ Re-entry Handling:
    * StrongSORT’s ReID feature ensures that if a person exits and later re-enters the frame, they are reassigned the same ID based on their appearance and movement trajectory.
    * The ReID module compares extracted features to ensure consistency in ID assignment even after breaks in visibility, using OsNet.
    * Why osnet_ain_x1_0_imagenet?
>>>> I chose osnet_ain_x1_0_imagenet for my project because it excels in person re-identification by capturing omni-scale features, which are crucial for distinguishing individuals in varied environments. Its innovative architecture and lightweight design make it well-suited for complex tracking scenarios, such as occlusions and re-entries. AIN (Attribute Instance Normalization) was particularly beneficial for handling domain shifts and diverse conditions, providing more accurate re-identifications compared to IBN. Integrating this model with StrongSORT enhances tracking reliability by combining both motion and appearance information efficiently.

+ Post-Occlusion Tracking:
    * If a person is temporarily occluded or becomes partially visible, the system will attempt to reassign the correct ID upon reappearance.
    * This is done by comparing spatial proximity and visual features using the ReID module, which prevents the tracker from mislabeling individuals after occlusion events.

### Training and Fine-tuning Details
The model was trained using a custom dataset with labeled images of children and adults in various environments. The following steps were taken:

+ **Dataset Preparation**: The dataset was divided into training, validation, and test sets. It was preprocessed and augmented to improve model generalization.
+ **Model Training**: YOLOv9 was fine-tuned with specific configurations to enhance detection accuracy, especially for child and adult categories. A batch size of 8 was used to balance memory limitations and training efficiency.
+ **Performance Metrics**: The model was evaluated using metrics such as precision, recall, and mAP (mean Average Precision) to ensure robust performance across different test cases.

### Model Performance
The trained YOLOv9 model in combination with StrongSORT achieves an accuracy of 85% for person detection and tracking in the test environment. The system demonstrates reliable performance in handling occlusions and re-entries, ensuring accurate tracking across various scenarios.


Test videos can be seen [Here](https://drive.google.com/drive/folders/1HNa2CPYz7EiGkazPX4aSZUqZLT-1BiAk)

<p align="center">
  <br>
<img align="center" src="https://github.com//Adult-Child-Tracker/blob/main/Images/results.png" width="600" height="450">   
 </p>
<p align="center">
  <br>
<img align="center" src="https://github.com/rounak351/Adult-Child-Tracker/blob/main/Images/confusion_matrix.png" width="500" height="450">   
 </p>
 <p align="center">
  <br>
<img align="center" src="https://github.com/rounak351/Adult-Child-Tracker/blob/main/Images/pred.jpg" width="500" height="450">   
 </p>
 

### Future Enhancements
1. Currently, identity switches are happening due to various angles encompasses in single video. Recent approaches like TransTrack could help.
2. Improve the system's speed and scalability for larger datasets.
3. Expand the model to detect more categories or additional attributes like emotional states.
4. Integrate behavior analysis algorithms for better interpretation of ASD children's interactions.

### Conclusion
This project provides a robust system for detecting and tracking children and adults in video streams. By leveraging YOLOv9 for detection and StrongSORT with ReID for tracking, the system accurately assigns unique IDs, handles re-entries and occlusions, and serves as a tool for analyzing the behavior of children with ASD. This information can be used by therapists and researchers to develop tailored treatment plans.
