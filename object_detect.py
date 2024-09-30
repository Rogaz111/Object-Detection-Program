import cv2
import numpy as np
import tensorflow as tf
import time
from coco_labels import COCO_LABELS
from text_writer import write_to_file

# Load the model
model = tf.saved_model.load(r"C:\Users\RaynDaries\tensorflowObjectDetection\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the maximum number of objects to detect
MAX_OBJECTS = 3

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for object detection
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform detection
        detections = model(input_tensor)

        # Process the detections
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Get the indices of the top MAX_OBJECTS detections
        top_indices = np.argsort(detections['detection_scores'])[-MAX_OBJECTS:][::-1]

        # Visualize the results for the top detections
        for i in top_indices:
            score = detections['detection_scores'][i]
            if score > 0.5:  # You can adjust this threshold
                ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
                im_height, im_width, _ = frame.shape
                left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)

                # Draw bounding box
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

                # Add label and score
                class_id = detections['detection_classes'][i]
                label = f"{COCO_LABELS.get(class_id, 'Unknown')}: {score:.2f}"
                if label:
                    write_to_file(label)
                cv2.putText(frame, label, (int(left), int(top) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add a delay to slow down the loop (adjust as needed)
        time.sleep(0.1)  # 100 milliseconds delay

except KeyboardInterrupt:
    print("Interrupted by user. Closing gracefully.")

finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()