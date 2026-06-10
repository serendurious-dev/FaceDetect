# FaceDetect — Technical Documentation

> A deeper write-up of the implementation, the design decisions, and what
> working with Haar Cascade actually taught me.


---

## 1. Project overview

FaceDetect is a command-line Python script that detects human faces in image
files using OpenCV's Haar Cascade classifier. It takes a single image as input,
runs a detection pipeline, annotates the image with numbered bounding boxes, and
saves the result.

The goal was to go beyond a minimal demo. The project includes input validation,
image quality checks, preprocessing steps that actually improve accuracy, tuned
detection parameters, and structured terminal output. It behaves like a proper
tool, not a classroom exercise.

- **Language:** Python 3.13
- **Libraries:** `cv2` (opencv-python), `sys`, `os`, `pathlib`
- **Model:** `haarcascade_frontalface_default.xml` (bundled with OpenCV)
- **Input:** Any image file via command-line argument
- **Output:** `face_detection.png` with annotated faces

---

## 2. Program flow

The `main()` function runs the program in seven steps:

Each step is isolated in its own function. If anything fails at any step, the
program calls `abort()`, prints a clear error message, and exits with a non-zero
code. Nothing crashes silently.

---

## 3. Validation layer

Before any detection happens, the program validates everything that could go
wrong.

### 3.1 File validation (`validate_file`)

```python
def validate_file(image_path):
```

This function checks three things in order:

- Does the file exist at the given path?
- Is it actually a file, not a directory?
- Is the file extension in the allowed set?

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

Errors at this stage tell the user exactly what went wrong instead of letting
OpenCV produce a vague failure later.

### 3.2 Image validation (`load_and_validate_image`)

```python
def load_and_validate_image(image_path):
```

After loading the image with `cv2.imread()`, two more checks run:

**Dimension check** — If either dimension is below `MIN_IMAGE_DIMENSION` (100px),
the image is too small to contain detectable faces and the program exits with a
clear message.

**Blank image check** — The standard deviation of the grayscale pixel values is
calculated. If `std_dev < 5`, the image is almost entirely one color, meaning
there is nothing useful to detect. This filters out test images, blank canvases,
or solid-colored files that would otherwise produce zero faces with no explanation.

### 3.3 Classifier validation (`load_cascade_classifier`)

The Haar Cascade XML file is loaded from OpenCV's own data directory:

```python
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
```

Two checks run:

- Does the file path exist on disk?
- Does `CascadeClassifier.empty()` return False?

If either fails, the error message suggests reinstalling OpenCV, since that is
almost always the cause.

---

## 4. Detection pipeline (`detect_faces`)

This is the core of the project. The function takes the original BGR image and
the loaded classifier, runs the full detection pipeline, and returns the annotated
image and a list of bounding box coordinates.

### Step 1: Grayscale conversion

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

Haar Cascade works on single-channel grayscale images. Color data is discarded
here because it is not used by the classifier.

### Step 2: Histogram equalization

```python
gray = cv2.equalizeHist(gray)
```

This redistributes the pixel intensity histogram so that the full 0–255 range is
used. In practice, it makes faces that are darker or lighter than the background
more visible to the classifier.

Without this step, images with poor lighting or high-contrast backgrounds
produce noticeably worse detection results.

### Step 3: 2× upscaling

```python
scale_up = 2.0
resized = cv2.resize(gray, None, fx=scale_up, fy=scale_up,
                     interpolation=cv2.INTER_CUBIC)
```

The grayscale image is doubled in size before detection.

This is the most impactful preprocessing decision in the project. The reason is
that `detectMultiScale` has a `minSize` parameter that sets the smallest face
region it will look for. If the actual face in the photo is only slightly above
that minimum size in the original image, detection becomes unreliable. Upscaling
makes every face effectively larger relative to the minimum size, which improves
recall significantly on group photos where some people are further from the
camera.

The trade-off is that the image being passed to the detector is four times as
large in pixel count, so detection is slower. For a single image, this is
completely acceptable.

### Step 4: `detectMultiScale`

```python
faces = classifier.detectMultiScale(
    resized,
    scaleFactor=1.1,
    minNeighbors=7,
    minSize=(70, 70),
)
```

**`scaleFactor=1.1`** — At each pyramid level, the image is shrunk by 10%. A
smaller value like 1.05 checks more scales and finds more faces, but takes longer
and produces more false positives. A larger value like 1.3 is faster but misses
faces that fall between scale levels.

**`minNeighbors=7`** — A candidate rectangle has to be confirmed by at least 7
neighboring detections before it counts as a face. Lower values (3–5) produce
more detections but also more false positives. Higher values (8+) are more precise
but can miss real faces. 7 worked well in testing for the group photo use case.

**`minSize=(70, 70)`** — Applied to the upscaled image. Since the image was
doubled, this corresponds to a 35×35 pixel face region in the original image.
This filters out tiny regions that are noise rather than faces.

### Step 5: Coordinate scaling and annotation

```python
x = int(x / scale_up)
y = int(y / scale_up)
w = int(w / scale_up)
h = int(h / scale_up)
```

The bounding box coordinates returned by the classifier are in upscaled image
space. Dividing by `scale_up` brings them back to original image coordinates.
Without this step, the rectangles would be drawn in the wrong positions and would
appear too large.

Each detected face then gets:

- A green rectangle (`BOX_COLOR = (0, 255, 0)`, `BOX_THICKNESS = 3`)
- A red label `"Face N"` placed above the rectangle (`COUNT_COLOR = (0, 0, 255)`)

The label is positioned at `y - 8` or at minimum `y = 25` to prevent labels
at the very top of the image from being clipped.

---

## 5. Output

### Saving (`save_output`)

```python
cv2.imwrite(OUTPUT_FILE, image)
```

The annotated image is saved as `face_detection.png`. `imwrite` returns a boolean
indicating success, which is checked. If writing fails (permissions issue, full
disk), the error is caught and reported cleanly.

### Display (`display_image`)

If `SHOW_IMAGE_WINDOW = True`, the result is shown in an OpenCV window. The
program waits for a key press before closing. If the environment is headless
(server, SSH session without X forwarding), the display call is wrapped in a
try/except and prints a warning instead of crashing.

---

## 6. Configuration constants

All tunable values are at the top of the file for easy adjustment:

```python
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
OUTPUT_FILE = "face_detection.png"
MIN_IMAGE_DIMENSION = 100
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3
TEXT_COLOR = (0, 255, 0)
COUNT_COLOR = (0, 0, 255)
SHOW_IMAGE_WINDOW = True
```

This makes it easy to adjust output format, visual style, and behavior without
digging through the logic.

---

## 7. What Haar Cascade can and cannot do

Understanding where the method fails is as important as understanding where it
works.

**It works well when:**
- Faces are clearly forward-facing.
- The image has reasonable lighting and contrast.
- Faces are large enough relative to the frame.
- There are no extreme occlusions.

**It struggles when:**
- Faces are in profile or looking down.
- Faces are partially blocked.
- Lighting creates heavy shadows directly on the face.
- Small faces are far from the camera (this is partly what the 2× upscaling
  helps with).
- Backgrounds contain patterns that loosely resemble face-like structures.

Haar Cascade is a sliding-window classifier trained on hand-designed features.
It is fast and needs no GPU, but it has been largely replaced by deep-learning
approaches for production use. For this project, the goal was to work with it
directly and understand its behavior, not to achieve perfect accuracy.

---

## 8. Testing notes

I tested the program across several different input images during development.

The main things I observed:

- **Image quality is the biggest factor.** High-resolution, well-lit photos with
  clear frontal faces produce near-perfect results. Blurry, dark, or low-resolution
  images are much harder.
- **The 2× upscaling made a measurable difference.** Before adding it, some faces
  in the sample group photo were being missed. After adding it, the detection
  became consistent.
- **`minNeighbors=7` was the right balance for group photos.** Going lower
  started introducing false positives on background patterns. Going higher started
  missing faces that were slightly off-angle.
- **The blank image check was necessary.** During testing I accidentally passed
  the wrong file and got zero detections with no feedback. The standard deviation
  check now catches that case with a clear error.

---

## 9. Limitations

- The program only handles static images, not video streams.
- Fill mode and undo are not concepts that apply here, but the lack of batch
  processing is a real limitation. Right now it handles one image per run.
- Haar Cascade is not state-of-the-art. If the goal is high accuracy on difficult
  images, a deep-learning model like MTCNN or RetinaFace would perform better.
- The output format is always PNG. The format is not configurable from the command
  line.

---

## 10. How to run

```bash
pip install opencv-python
python face_detection.py group_photo.jpg
```

If the window does not appear, set `SHOW_IMAGE_WINDOW = False` and check the
saved PNG directly.

---

## 11. Final note

The thing that surprised me most while building this was how much the
preprocessing decisions mattered. The classifier itself is just one function call.
The real work was everything around it: the upscaling, the equalization, the
coordinate scaling, the validation checks. That is the part where the project
goes from a demo to something that actually behaves reliably.

That lesson applies well beyond face detection.
