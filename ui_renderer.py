import cv2

class UIRenderer:
    def __init__(self, circle_radius=30):
        # Initialize default circle radius
        self.circle_radius = circle_radius

    def draw_circle(self, frame, position, color=(0, 0, 255)):
        """
        Draw a circle at a specified position on the frame.
        :param frame: The input frame from the video feed.
        :param position: (x, y) coordinates for the center of the circle.
        :param color: Color of the circle in BGR format (default is red).
        """
        cv2.circle(frame, position, self.circle_radius, color, -1)

    def draw_text(self, frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
        """
        Draw text on the frame at a specified position.
        :param frame: The input frame from the video feed.
        :param text: The text string to display.
        :param position: (x, y) coordinates for the bottom-left corner of the text.
        :param font_scale: Scale of the font (default is 0.7).
        :param color: Color of the text in BGR format (default is white).
        :param thickness: Thickness of the text strokes (default is 2).
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def display_scores(self, frame, last_reaction_time, average_reaction_time):
        """
        Display last and average reaction times on the frame.
        :param frame: The input frame from the video feed.
        :param last_reaction_time: The last recorded reaction time.
        :param average_reaction_time: The average reaction time across attempts.
        """
        # Draw last reaction time
        self.draw_text(frame, f"Last: {last_reaction_time:.3f} s", (10, 30))
        # Draw average reaction time
        self.draw_text(frame, f"Average: {average_reaction_time:.3f} s", (10, 60))
