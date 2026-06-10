"""
===============================================
    ADVANCED HUMAN FACE DETECTION SYSTEM
===============================================

Description:
    This script detects human faces in a group photo using OpenCV's Haar Cascade classifier.
    It draws bounding boxes around detected faces and saves the annotated image as 'face_detection.png'.
    The script also prints the number of faces detected to the console.
    
"""


import cv2
import sys
import os
from pathlib import Path

#============================================
# CONFIGURATION
#============================================

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
OUTPUT_FILE = "face_detection.png"

MIN_IMAGE_DIMENSION = 100   # Minimum width/height in pixels to consider for face detection
MIN_IMAFE_HEIGHT = 100

BOX_COLOR = (0, 255, 0)     # Green color for bounding boxes (BGR format)
BOX_THICKNESS = 3           # Thickness of bounding box lines

TEXT_COLOR = (0, 255, 0)    # Green color for labels
COUNT_COLOR = (0, 0, 255)   # Red color for face count text

SHOW_IMAGE_WINDOW = True

#============================================
# UTILITY FUNCTIONS
#============================================

def abort(message: str):
    """Print an error message and exit the program."""
    print(f"[ERROR] {message}")
    sys.exit(1)
    
def print_header():
    """Display program header."""
    print("=" * 70)
    print("ADVANCED HUMAN FACE DETECTION SYSTEM")
    print("=" * 70)


#============================================
# VALIDATION FUNCTIONS
#============================================

def validate_command_line():
    """Validate command-line arguments."""
    if len(sys.argv) < 2:
        print("Usage : python face_detection.py <path_to_group_photo>")
        print("Example: python face_detection.py group_photo.jpg")
        sys.exit(1)
    if len(sys.argv) > 2:
        abort("Too many arguments. Please provide a single image file path.")
    return sys.argv[1]

def validate_file(image_path):
    """Validate image file path and extension."""
    
    path = Path(image_path)
    
    # File existence
    if not path.exists():
        abort(f"File not found: '{image_path}'")
        
    # Ensure it is a file
    if not path.is_file():
        abort(f"'{image_path}' is a directory, not a file.")
        
    # Validate file extension
    if path.suffix.lower() not in VALID_EXTENSIONS:
        abort(
            f"Unsupported file type '{path.suffix}'.\n"
            f"  Accepted formats: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
        
def load_and_validate_image(image_path):
    """Load the image and validate its dimensions."""
    image = cv2.imread(image_path)
    
    if image is None:
        abort(
            "OpenCV failed to load the image. This may indicate a corrupted file or unsupported format."
        )
    height, width = image.shape[:2]
    
    print(f"\n[INFO] Image loaded successfully")
    print(f"[INFO] Image dimensions: {width}×{height} px")
    
    # Resolution check
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        abort(
            f"Image is too small ({width}×{height} px). "
            f"Please provide an image at least {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px."
        )
        
    # Detect almost blank images
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = cv2.meanStdDev(gray)[1][0][0]
    if std_dev < 5:
        abort(
            "The image appears to be almost entirely a single color (std dev < 5). "
            "Please provide a valid group photo with distinguishable features."
        )
    return image

def load_cascade_classifier():
    """Load Haar Cascade model."""
    
    cascade_path = (
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )
    if not os.path.exists(cascade_path):
        abort(
            "Haar Cascade XML file not found.\n"
            "Try reinstalling OpenCV:\n"
            " pip install --upgrade opencv-python"
        )
        
    classifier = cv2.CascadeClassifier(cascade_path)
    
    if classifier.empty():
        abort(
            "Failed to load Haar Cascade classifier. The XML file may be corrupted."
        )
    print(f"[INFO] Haar Cascade loaded successfully from:\n  {cascade_path}")
    return classifier

#============================================
# FACE DETECTION
#============================================


def detect_faces(image, classifier):
    """
    Detect faces in image using Haar Cascade.
    Returns:
    faces -> detected coordinates
    annotated -> image with rectangles
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Improve contrast for better detection
    gray = cv2.equalizeHist(gray)
    scale_up = 2.0
    resized = cv2.resize(gray, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC)
    
    # Detect Faces
    faces = classifier.detectMultiScale(
        resized,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(70, 70),
    )
    annotated = image.copy()
    scaled_faces = []
    
    # Draw rectangles and labels
    for index, (x, y, w, h) in enumerate(faces, start=1):
        x = int(x / scale_up)
        y = int(y / scale_up)
        w = int(w / scale_up)
        h = int(h / scale_up)
        scaled_faces.append((x, y, w, h))
        # Face rectangle
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            BOX_COLOR,
            BOX_THICKNESS
        )
        
        # FACE LABEL
        cv2.putText(
            annotated,
            f"Face {index}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COUNT_COLOR,
            2
        )
    return annotated, scaled_faces
    
#=============================================
# OUTPUT FUNCTIONS
#=============================================

def save_output(image):
    """Save annotated output image."""
    
    success = cv2.imwrite(OUTPUT_FILE, image)
    if not success:
        abort(
            f"Failed to save output image to '{OUTPUT_FILE}'. "
            "Check file permissions and available disk space."
        )
    print(f"[INFO] Annotated image saved successfully as '{OUTPUT_FILE}'")
    
def display_image(image):
    """Display result window."""
    
    if not SHOW_IMAGE_WINDOW:
        return
    try:
        cv2.imshow("Face Detection Result", image)
        print("[INFO] Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print(
            "[WARNING] Unable to display image window. "
            "This may be due to running in a headless environment (e.g., server without GUI)."
        )
        
#=============================================
# MAIN FUNCTION
#=============================================

def main():
    print_header()
    
    #Step 1: Ask user or image path
    image_path = r"C:\Desktop\Sejong\Sejong 3\Open Source Software\FaceDetection\group_photo.jpg"
    #Step 2: Validate file path and extension
    validate_file(image_path)
    #Step 3: Load and validate image content
    image = load_and_validate_image(image_path)
    #Step 4: Load Haar Cascade classifier
    classifier = load_cascade_classifier()
    
    print("\n[INFO] Detecting faces...")
    #Step 5: Detect faces and annotate image
    annotated, faces = detect_faces(image, classifier)
    face_count = len(faces)
    
    print("\n" + "-" * 70)
    
    if face_count == 0:
        print("[WARNING] No faces detected in the image.")
    else:
        print(f"[SUCCESS] Detected {face_count} face(s) in the image.")
        
    print("-" * 70)
    #Step 6: Save annotated image
    save_output(annotated)
    #Step 7: Display result
    display_image(annotated)
    
    print("\nProgram completed successfully.")
    
if __name__ == "__main__":
    main()