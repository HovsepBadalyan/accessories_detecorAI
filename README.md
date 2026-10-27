# Electronic Equipment Recognition

A real-time object recognition project built with Python, OpenCV, and TensorFlow. The application uses a webcam to capture video frames and the MobileNetV2 deep learning model to identify objects, displaying the predicted label and confidence percentage directly on the video feed.

## Features

- **Real-time recognition:** Identifies objects using a webcam.
- **Deep learning:** Uses the pretrained MobileNetV2 model.
- **Confidence score:** Displays the prediction confidence as a percentage.
- **Live video processing:** Processes frames continuously.
- **Simple interface:** Shows the predicted class on the webcam window.
- **Easy exit:** Press `q` to close the application.

## Technologies Used

- Python
- OpenCV (`cv2`)
- NumPy
- TensorFlow / Keras
- MobileNetV2

## Project Structure

```text
ElectronicRecognition/
├── electronic_recognition.py
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ElectronicRecognition.git
cd ElectronicRecognition
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the repository name if necessary.

### 2. Install the dependencies

Make sure Python is installed, then run:

```bash
pip install opencv-python numpy tensorflow
```

## How to Run

Execute the Python script:

```bash
python electronic_recognition.py
```

The application will:

1. Load the pretrained MobileNetV2 model with ImageNet weights.
2. Open the default webcam.
3. Capture video frames continuously.
4. Resize each frame to 224 × 224 pixels.
5. Preprocess the image for MobileNetV2.
6. Predict the object's class.
7. Display the predicted label and confidence percentage on the video.

Press **`q`** to exit.

## Example Output

The webcam window displays a prediction in the following format:

```text
laptop (94.5%)
```

*Example only — the actual label and confidence depend on the object and the model's prediction.*

## How the Model Works

The project uses **MobileNetV2**, a convolutional neural network pretrained on the ImageNet dataset.

The model predicts an object category from each captured frame. The `decode_predictions()` function converts the model's output into a human-readable label and confidence score.

The confidence percentage is calculated as:

```python
confidence * 100
```

The current implementation displays the top prediction only.

## Important Notes

- The model recognizes classes from its pretrained ImageNet label set. It is not specifically trained to recognize electronic equipment.
- The confidence score is the model's prediction score, not a guarantee that the classification is correct.
- A working webcam is required.
- The first run may take longer because the pretrained model weights need to be downloaded.
- Recognition speed depends on your computer's hardware.

## Future Improvements

- Train or fine-tune a model specifically for electronic equipment.
- Recognize multiple objects in a single frame.
- Display the top three predictions.
- Add support for image uploads.
- Save recognition results and timestamps.
- Build a graphical user interface.
- Integrate the model into a Telegram bot.

## License

This project is available for educational and personal use. Add a license file to specify the permissions and restrictions for redistribution and modification.
