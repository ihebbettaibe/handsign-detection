import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 9
dataset_size = 100

cap = cv2.VideoCapture(0)  # Test with different indices if necessary

# Ensure camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create class directories and collect images for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Ready state before collecting images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Skipping...")
            continue

        # Display ready message
        cv2.putText(frame, 'Press "Q" to start capturing images', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Ready', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('Ready')

    # Collecting images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Skipping...")
            continue

        cv2.imshow('Capture', frame)
        cv2.waitKey(25)

        # Save frame as an image
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

    cv2.destroyWindow('Capture')

cap.release()
cv2.destroyAllWindows()
