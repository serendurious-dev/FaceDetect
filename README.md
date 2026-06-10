# FaceDetect

> A command-line face detection tool that finds human faces in photos and 
> annotates them with numbered bounding boxes.

This is my assignment for Introduction to Open Source Software. It detects human
faces in any image you give it, draws green rectangles around each face it finds,
labels them Face 1, Face 2, and so on, and saves the annotated result as a PNG
file.

Underneath, it runs OpenCV's Haar Cascade classifier. The interesting part was
not just calling the function that finds faces. It was the layer around it: the
image upscaling before detection, the histogram equalization to improve contrast,
the coordinate scaling back to original size, and all the validation logic that
keeps the program from crashing silently when the input is bad.

Built by **Prodipta Acharjee**.

---

## Why I built it

I wanted to understand how classical computer vision actually works before jumping
to deep learning. Haar Cascade is one of the foundational face detection methods,
and working with it directly taught me a lot about how image classifiers scan at
multiple scales, why preprocessing matters, and what the actual trade-offs are
between sensitivity and false positives.

I also wanted to build something that is more than a five-line script. A lot of
OpenCV demos online load the image and call `detectMultiScale` and nothing else.
I cared about building something with proper error handling, structured output,
and tunable parameters, so the project actually behaves like a real tool.

---

## What it does

You give it an image file. It:

- Validates the file path, extension, and image content before doing anything.
- Converts the image to grayscale and applies histogram equalization.
- Upscales the grayscale image by 2× before detection to improve accuracy on
  smaller faces.
- Runs the Haar Cascade classifier with tuned parameters.
- Scales the detected coordinates back to match the original image size.
- Draws a green bounding box and a red numbered label around each face.
- Saves the annotated image as `face_detection.png`.
- Prints the face count and step-by-step status to the terminal.

---

## Preview

| Input | Output |
|---|---|
| Plain group photo | Same photo with green boxes and Face 1–N labels |

The output on the sample image detected all 5 visible frontal faces correctly
with no false positives.

---

## Requirements

- Python 3.8 or newer
- OpenCV

```bash
pip install opencv-python
```

No other third-party dependencies. `sys`, `os`, and `pathlib` are all standard
library.

---

## How to run

```bash
python face_detection.py <image_file>
```

Example:

```bash
python face_detection.py group_photo.jpg
```

The annotated output will be saved as `face_detection.png` in the same directory.
If `SHOW_IMAGE_WINDOW` is set to `True` in the configuration, the result also
opens in a live window. Press any key to close it.

---

## Supported formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

---

## Configuration

These constants at the top of the file control the detection behavior:

| Constant | Default | Description |
|---|---|---|
| `OUTPUT_FILE` | `face_detection.png` | Output file name |
| `MIN_IMAGE_DIMENSION` | `100` | Minimum image width/height in pixels |
| `BOX_COLOR` | `(0, 255, 0)` | Bounding box color (BGR green) |
| `BOX_THICKNESS` | `3` | Bounding box line thickness |
| `COUNT_COLOR` | `(0, 0, 255)` | Label text color (BGR red) |
| `SHOW_IMAGE_WINDOW` | `True` | Whether to open the result in a window |

The detection parameters inside `detect_faces()`:

| Parameter | Value | Meaning |
|---|---|---|
| `scaleFactor` | `1.1` | How much the image shrinks at each scale step |
| `minNeighbors` | `7` | How many neighbors a detection needs before it counts |
| `minSize` | `(70, 70)` | Smallest face region to consider (before upscaling) |

---

## How detection actually works

The program does not just pass the original image directly to the classifier.
It goes through a few extra steps first, and those steps matter.

1. **Grayscale conversion** — Haar Cascade only works on single-channel images.
   Color information is not needed for the detection.

2. **Histogram equalization** — This redistributes pixel intensity across the
   full range. It makes dark faces in bright scenes, or bright faces in dark
   scenes, more distinguishable.

3. **2× upscaling** — The grayscale image is resized to double its dimensions
   before detection. This helps significantly with faces that appear small in
   a photo, because the classifier's minimum size threshold is applied to the
   scaled image. Larger faces are easier to detect reliably.

4. **Scale-back** — After detection, every bounding box coordinate is divided
   by the scale factor to get back to the original image dimensions. The boxes
   are then drawn on the original unscaled image.

5. **Annotation** — Each face gets a rectangle and a "Face N" label drawn over
   the original image. Labels appear above the box when possible.

---

## Known limitations

Haar Cascade is a classical method and it has real limits:

- It works best on clear, front-facing faces.
- Profile faces, downward-turned faces, or partially hidden faces often get
  missed.
- Busy backgrounds can cause false positives. A window frame, a shadow, or
  certain patterns on clothing can sometimes look like a face to the classifier.
- Results vary a lot based on image quality, lighting, and face size.

I tested it on multiple images during development and found that the performance
really depends on how clear and forward-facing the faces are. The sample group
photo worked well. Some other test images did not. That is an honest limitation
of the method, not a bug in the code.

---

## Project structure

```text
face_detection.py     main script
README.md             this file
DOCUMENTATION.md      deeper technical write-up
LICENSE               MIT license
group_photo.jpg       sample input image
face_detection.png    sample output (generated at runtime)
```

---

## License

MIT. See [LICENSE](LICENSE).

---

## Author

Prodipta Acharjee. Built as an assignment for Introduction to Open Source Software,
Sejong University, 2026.
