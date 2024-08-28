from ultralytics import YOLO
import cv2
import time
# import mediapipe as mp
import math
# from face_recognition_base import FaceRecognitionBase
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import copy 
import numpy as np 
from deepface import DeepFace
import tensorflow as tf 
import matplotlib.pyplot as plt



"""Face Rec Module"""
# List of available backends, models, and distance metrics
backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]



# Initialize and load the trained YOLO model
model = YOLO('best.pt')
classNames = ['bottle', 'juice-cup', 'nescafe', 'plate', 'tissue']



"""Fine System"""
# Set up directory and path for fines database
directory = 'FineDatabase'
if not os.path.exists(directory):
    os.makedirs(directory)
excel_path = os.path.join(directory, 'fines.xlsx')

# Load or create fines DataFrame
try:
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    df = pd.DataFrame(columns=["Reg_No", "Date", "Fine"])
    print("No existing file found. Created a new DataFrame.")


last_updated = {}

def save_dataframe(df, path):
    """Save the DataFrame to an Excel file."""
    try:
        df.to_excel(path, index=False)
        print(f"DataFrame saved/updated successfully to {path}")
    except Exception as e:
        print("Failed to save DataFrame:", e)


def update_fines(name, current_time):
    """Update fines for the detected person."""
    global df  # Add this line to ensure df is referenced from the global scope
    if name in df['Reg_No'].values:
        if name not in last_updated or current_time - last_updated[name] > timedelta(minutes=2):
            df.loc[df['Reg_No'] == name, 'Fine'] += 200
            df.loc[df['Reg_No'] == name, 'Date'] = current_time
            last_updated[name] = current_time
            save_dataframe(df, excel_path)
    else:
        new_record = pd.DataFrame([{"Reg_No": name, "Date": current_time, "Fine": 200}])
        df = pd.concat([df, new_record], ignore_index=True)
        last_updated[name] = current_time
        save_dataframe(df, excel_path)



"""Models"""
def detect_objects(frame):
    """Detect objects in the frame using YOLO model."""
    predictions = model(frame, stream=True)
    return predictions

        

"""MoveNet"""
def run_inference(interpreter, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv2.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.uint8)  # uint8へキャスト

    # 推論
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    """Index for only hands"""

    index01, index02 = 9, 10

    point01 = keypoints[index01] # x1, y1
    point02 = keypoints[index02]    
    cv2.circle(debug_image, point01, 6, (255, 255, 255), -1)
    cv2.circle(debug_image, point02, 6, (255, 255, 255), -1)

    cv2.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv2.LINE_AA)

    cv2.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv2.LINE_AA)

    return debug_image, point01, point02


def calculate_lengths(frame, p1, p2, x1, y1):
    """Calculate the lengths of left and right arms."""
    len_left = len_right = 0 
    # points 
    a1, b1 = p1[0], p1[1] # Left hand
    a2, b2 = p2[0], p2[1] # Right hand

    len_left = max(len_left, math.hypot(a1 - x1, b1 - y1))
    len_right = max(len_right, math.hypot(a2 - x1, b2 - y1))

    return len_left, len_right


def detect_fines(frame, length, threshold, people):
    # Detects face and assign fine if length exceeds the threshold  

    print("[INFO] Running Face Detection")
    for person in people:
        # Retrieve the coordinates of the face bounding box
        x = person['source_x'][0] # x-co
        y = person['source_y'][0] # y-co
        w = person['source_w'][0] # Width 
        h = person['source_h'][0] # height

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the person's name and display it on the image
        name = person['identity'][0].split('/')[1]
        
        # cv2.putText(frame, name, (x, y), cv2.FONT_ITALIC, 1, (0, 0, 255), 2) # We dont want to display the name

        hh = h/2
        wh = w/2

        print(f"----------------------------")
        print(f"LENGTH = {length}")
        print(f"NAME = {name}")
        print(f"----------------------------")
        if length > threshold:
            cv2.putText(frame, "Potential Litterer Detected", (int(x-wh), int(y-hh)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update fines 
            # if name!="Unknown": 
            #     update_fines(name, datetime.now())


"""Set the Video Capture""" 
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)


def main():
    """Main function to run the video stream and process frames."""
    print("[INFO] Starting Stream...")

    model_path = 'MoveNet/tflite/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'

    # Define Interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()


    while True:
        start_time = time.time()

        # --- Frame Read Starts ---
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        predictions = detect_objects(frame)


        # DeepFace Model
        people = DeepFace.find(img_path=frame, db_path="face_database/", model_name=models[2], distance_metric=metrics[2], enforce_detection=False)

        for r in predictions:
            boxes = r.boxes
            for box in boxes: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                text = f"{classNames[cls]}: {conf}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                """Pose Detection"""
                keypoints, scores = run_inference(
                    interpreter=interpreter,
                    input_size=192,
                    image=frame,
                )

                elapsed_time = time.time() - start_time

                frame, p1, p2 = draw_debug(
                    frame,
                    elapsed_time,
                    keypoint_score_th=0.4,
                    keypoints=keypoints,
                    scores=scores,
                )

                # Show the interaction bw hands and object
                l1, l2 = calculate_lengths(frame, p1, p2, x1, y1)
                
                # Face detection
                if l2 > l1:
                    # For left hand
                    # We have: x1, y1, p1
                    a, b = p1[0], p1[1]
                    cv2.line(frame, (a, b), (x1, y1), (0, 0, 0), 1)
                    # Detect potential litterer
                    detect_fines(frame, l1, 100, people)
                else:
                    # For right hand 
                    c, d = p2[0], p2[1]
                    cv2.line(frame, (c, d), (x1, y1), (0, 0, 0), 1)
                    detect_fines(frame, l2, 100, people)


        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
