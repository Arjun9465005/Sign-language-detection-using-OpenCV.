import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model from pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize lists to store hand landmark coordinates
x_ = []
y_ = []

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Dictionary to map prediction indices to characters
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB format (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Reset the lists for new frame data
    x_.clear()
    y_.clear()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect x and y coordinates of hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

        if x_ and y_:
            # Calculate normalized coordinates relative to the minimum x and y
            # Ensure data_aux is cleared for each frame to avoid accumulating data from previous frames
            data_aux = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                if x_ and y_:
                    # Calculate normalized coordinates relative to the minimum x and y
                    for i in range(len(x_)):
                        x = x_[i]
                        y = y_[i]
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Ensure data_aux has exactly 42 features (adjust as per your model's requirements)
                    data_aux = data_aux[:42]  # Adjust this based on your model's expected input size

                    # Predict gesture character using the model
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

            # Calculate bounding box coordinates around the detected hand landmarks
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw rectangle around the detected hand landmarks
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            # Display predicted character label on the frame
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame with annotations
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
