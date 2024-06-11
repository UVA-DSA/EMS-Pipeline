from multiprocessing import Process
import mediapipe as mp
from PIL import Image
from classes import MPQueueImage

# Global variables for MediaPipe utilities
global mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


class MediaPipeProcess(Process):
    # Constructor for the MediaPipeProcess class
    def __init__(self, input_queue, output_queue):
        """ Instantiates a separate process for MediaPipe image annotation.

        Args:
            input_queue (multiprocessing.Queue): input queue for process_image() of MPQueueImage objects
            output_queue (multiprocessing.Queue):output queue for annotated images of MPQueueImage objects

        MPQueueImage is defined in /classes.py
        """
        super(MediaPipeProcess, self).__init__()
        self.input_queue = input_queue  # Queue for input images
        self.output_queue = output_queue  # Queue for output images
        self.is_running = True

    def process_image(self, QueueImage:MPQueueImage):
        """Performs hand landmark detection on an inputted image and returns the result.

        Args:
            QueueImage (MPQueueImage): QueueImage object of type MPQueueImage

        Returns:
            MPQueueImage: processed image.
        """
        hand_detection_results = None

        # Initialize the MediaPipe model
        with mp_hands.Hands(
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            # Process the image through the model
            hand_detection_results = hands.process(QueueImage.image)

        # Annotate the image with hand landmarks if any hands are detected
        if hand_detection_results and hand_detection_results.multi_hand_landmarks:
            for hand_landmarks in hand_detection_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(QueueImage.image,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

        return QueueImage  # Return the annotated image

    # Overridden run method from Process class
    def run(self):
        while self.is_running:  # Loop to process images
            if not self.input_queue.empty():
                queued_image = self.input_queue.get()
                processed_image = self.process_image(QueueImage=queued_image)
                self.output_queue.put(processed_image)