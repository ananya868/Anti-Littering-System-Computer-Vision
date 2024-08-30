# Anti-Littering System 🚯
This system uses YOLO for object detection (specifically, garbage), MoveNet for hand landmark detection, and DeepFace for facial recognition. It analyzes the relationship between detected humans and garbage to identify potential littering incidents in real-time

## Introduction 😶‍🌫️ 
*The system comprises of integration of object detection, pose detection and face detection using cv2 library to effectively identify littering action. We have trained YoloV8 on custom garbage dataset (which include classes like cups, tissues etc.). Pose detection is used to capture hand landmarks which is furthur needed to get relation between human and object. Lastly, Single shot face detection has been used to identify known and unknown litterers and a fine system has been put that stores and updates fines for individuals based on littering action in a csv file. Please find the details on how to setup and use this repo below!*  

## Features 🗒
- **Real-Time Detection**: Identifies littering incidents as they happen.
- **Multi-Model Integration**: Combines object detection, hand landmark detection, and facial recognition.
- **Scalable**: Deployable in various public spaces like parks, streets, and campuses.
- **Alerts and Reporting**: Generates alerts and fines for potential litterers.

## Prerequisites 
- Git
- Python 3.x
- Pip
- Virtual environment tools (e.g., virtualenv, conda)
- Webcam, or any camera device for real-time testing

  *Note: The model will only work with objects (garbage) trained in our YoloV8, these classes include 'Nescafe cup', 'Foam plate', 'Plastic bottle', 'Tissue Paper' and 'Juice cup'*

## Installation and Usage 📲
Follow these steps to set up the Anti-Littering System on your local machine:

1. **Clone the repository**:
```python
git clone https://github.com/ananya868/Anti-Littering-System-Computer-Vision.git
cd Anti-Littering-System-Computer-Vision
```

2. **Install dependencies**
```python
pip install -r requirements.txt
```

3. **Setup Webcam**
- The default frame being used is webcam (0). Please setup your webcam before usage.

4. **Run**
- To run the integrated model, use:
```python
python IntegratedArchitecture.py
```

## Research Work
Paper link for the research work carried out: https://ieeexplore.ieee.org/document/10616763

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact 📇
For any inquiries or feedback, please contact us at:
- Email: ananya8154@gmail.com

## Acknowledgements
We would like to thank the following resources and communities for their support:
- [YOLO](https://github.com/AlexeyAB/darknet) for the object detection model.
- [MoveNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) for hand landmark detection.
- [DeepFace](https://github.com/serengil/deepface) for facial recognition.

Thank you for checking out the Anti-Littering System! We hope it makes a positive impact on keeping our environment clean and litter-free. 🌍✨
