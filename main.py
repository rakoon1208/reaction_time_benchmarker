import cv2
import time
import random
from hand_detection import HandDetector
from ui_renderer import UIRenderer

# Function to generate a random position for the circle within screen bounds
def generate_random_position(width, height, radius):
    x = random.randint(radius, width - radius)
    y = random.randint(radius, height - radius)
    return (x, y)

# Function to check if the circle overlaps with the hand's bounding box
def is_touching_circle_bounding_box(hand_bounding_box, circle_position, circle_radius):
    """
    Check if the circle overlaps with the hand's bounding box.
    :param hand_bounding_box: (x, y, width, height) of the bounding box
    :param circle_position: (x, y) position of the circle
    :param circle_radius: Radius of the circle
    :return: True if the circle overlaps the bounding box, otherwise false
    """
    x, y, width, height = hand_bounding_box
    cx, cy = circle_position
    # Check if any part of the circle overlaps with the bounding box
    return (x - circle_radius <= cx <= x + width + circle_radius and
            y - circle_radius <= cy <= y + height + circle_radius)

def main():
    # Set the camera resolution and circle radius
    width, height = 1980, 1080  # Set to native or preferred resolution
    radius = 25
    
    hand_detector = HandDetector()
    ui_renderer = UIRenderer(circle_radius=radius)
    
    circle_position = generate_random_position(width, height, radius)
    last_reaction_time = 0
    total_reaction_time = 0
    attempts = 0

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Set the camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hands and draw landmarks
        frame = hand_detector.find_hands(frame)
        
        # Get bounding boxes for all detected hands
        hand_bounding_boxes = hand_detector.get_hand_bounding_boxes(frame)

        touched = any(is_touching_circle_bounding_box(bbox, circle_position, radius) for bbox in hand_bounding_boxes)
        
        if touched:
            # Calculate and display reaction time
            end_time = time.time()
            last_reaction_time = end_time - start_time
            total_reaction_time += last_reaction_time
            attempts += 1

            circle_position = generate_random_position(width, height, radius)
            start_time = time.time()

        ui_renderer.draw_circle(frame, circle_position)

        average_reaction_time = total_reaction_time / attempts if attempts > 0 else 0

        ui_renderer.display_scores(frame, last_reaction_time, average_reaction_time)

        cv2.imshow("Reaction Time Test", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break


    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()  

if __name__ == "__main__":
    main()
