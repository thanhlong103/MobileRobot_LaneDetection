# Lane Detection with OpenCV

This project aims to detect lane lines on the road using computer vision techniques and OpenCV library in Python. The detection process involves several steps, such as color selection, region of interest selection, edge detection, and line detection using the Hough transform.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- MoviePy

You can install the required packages using the following command:

```
pip install opencv-python numpy matplotlib moviepy
```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/<your_username>/lane-detection.git
   ```
2. Install the required packages (see Prerequisites)

### Usage

1. Place your video file in the same directory as `main.py`.
2. Modify the input and output video file paths in `main.py` accordingly.
3. Run the following command to start the lane detection process:
   ```sh
   python main.py
   ```

## Algorithm Overview

1. Convert the input color image to grayscale.
2. Apply Gaussian blur to the grayscale image to smooth it and remove noise.
3. Apply Canny edge detection to the smoothed image to detect edges.
4. Define the region of interest (ROI) as a polygon and mask the edge image to keep only the edges within the ROI.
5. Apply the Hough transform to the masked edge image to detect lines.
6. Separate the detected lines into left and right lanes based on their slopes.
7. Extrapolate the left and right lanes to cover the full extent of the ROI.
8. Draw the extrapolated left and right lanes on the input color image.
9. Combine the input color image with the lane image to obtain the final output.

## Results

Sample input:

![Sample input](https://i.imgur.com/Piybm1m.png)

Sample output:

![Sample output](https://i.imgur.com/d7KuMkM.png)

## Acknowledgments

This project is inspired by the [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) program by Udacity.
