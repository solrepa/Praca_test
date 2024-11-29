import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as fd

from PIL import Image, ImageTk
from threading import Thread
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the pre-trained emotion recognition model
model = load_model('my_model_v2.h5')

# Define emotions based on the model's training dataset
emotions = ["Zlosc", "Zniesmaczenie", "Strach", "Szczescie", "Smutek", "Zaskoczenie", "Neutralny"]

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Global variables
camera_on = False
cap = None
emotion_counts = {emotion: 0 for emotion in emotions}  # Count durations for each emotion
start_time = None  # Time when the recording starts

# Function to preprocess the frame before prediction
def preprocess_frame(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
    return reshaped_face

# Function to predict emotion
def predict_emotion(face):
    processed_face = preprocess_frame(face)
    prediction = model.predict(processed_face)
    emotion_label = emotions[np.argmax(prediction)]
    return emotion_label

# Function to start the camera
def start_camera():
    global camera_on, cap, start_time, emotion_counts
    if not camera_on:
        # Reset data for the new recording session
        emotion_counts = {emotion: 0 for emotion in emotions}
        start_time = time.time()

        camera_on = True
        cap = cv2.VideoCapture(0)
        update_frame()

# Function to stop the camera
def stop_camera():
    global camera_on, cap
    camera_on = False
    if cap:
        cap.release()
        cap = None

def display_report():
    global emotion_counts, start_time
    if start_time is None:
        return  # No data to display the report

    total_time = time.time() - start_time  # Total duration of the session
    if total_time == 0:
        return  # Avoid division by zero

    # Calculate percentages for each emotion
    percentages = {emotion: (duration / total_time) * 100 for emotion, duration in emotion_counts.items()}

    # Ensure percentages add up to 100% in case of rounding errors
    total_percentage = sum(percentages.values())
    if total_percentage > 0:  # Avoid division by zero
        percentages = {emotion: (value / total_percentage) * 100 for emotion, value in percentages.items()}

    # Create a new window for the report
    report_window = tk.Toplevel(root)
    report_window.title("Raport z emocji")
    report_window.geometry("500x400")

    # Add a label to display session duration
    duration_label = tk.Label(report_window, text=f"Czas trwania sesji: {total_time:.2f} sekund", font=("Arial", 14))
    duration_label.pack(pady=10)

    # Create a Treeview widget to display the report as a table
    columns = ("Emocja", "Czas [%]")
    tree = ttk.Treeview(report_window, columns=columns, show="headings", height=8)

    # Define column headings
    tree.heading("Emocja", text="Emocja")
    tree.heading("Czas [%]", text="Czas [%]")

    # Define column widths
    tree.column("Emocja", width=200, anchor=tk.CENTER)
    tree.column("Czas [%]", width=100, anchor=tk.CENTER)

    # Insert data into the table
    for emotion, percentage in percentages.items():
        tree.insert("", tk.END, values=(emotion, f"{percentage:.2f}%"))

    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Save report function
    def save_report():
        file_path = fd.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Pliki tekstowe", "*.txt")],
                                         title="Zapisz raport")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Raport z rozpoznawania emocji:\nCzas trwania sesji: {total_time:.2f} sekund\n\n")
                f.write("Procentowy udział emocji:\n")
                for emotion, percentage in percentages.items():
                    f.write(f"{emotion}: {percentage:.2f}%\n")

    # Add a "Save" button
    save_button = tk.Button(report_window, text="Zapisz", command=save_report)
    save_button.pack(pady=10)



def display_chart():
    global emotion_counts, start_time
    if start_time is None:
        return  # No data to display the chart

    total_time = time.time() - start_time  # Total duration of the session
    if total_time == 0:
        return  # Avoid division by zero

    # Calculate percentages for each emotion
    percentages = {emotion: (duration / total_time) * 100 for emotion, duration in emotion_counts.items()}

    # Ensure percentages add up to 100% in case of rounding errors
    total_percentage = sum(percentages.values())
    if total_percentage > 0:  # Avoid division by zero
        percentages = {emotion: (value / total_percentage) * 100 for emotion, value in percentages.items()}

    # Create the chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(percentages.keys(), percentages.values(), color=['red', 'brown', 'purple', 'yellow', 'blue', 'orange', 'green'])
    ax.set_xlabel('Emocje')
    ax.set_ylabel('Procentowy udział')
    ax.set_title('Rozkład emocji')
    ax.set_xticks(range(len(percentages.keys())))
    ax.set_xticklabels(percentages.keys(), rotation=45)
    plt.tight_layout()

    # Create a new window for the chart
    chart_window = tk.Toplevel(root)
    chart_window.title("Wykres emocji")
    chart_window.geometry("700x500")

    # Embed the chart in the new window
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    # Save chart function
    def save_chart():
        file_path = fd.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("Obrazy PNG", "*.png")],
                                         title="Zapisz wykres")
        if file_path:
            fig.savefig(file_path)

    # Add a "Save" button
    save_button = tk.Button(chart_window, text="Zapisz", command=save_chart)
    save_button.pack(pady=10)


# Function to update the video feed
def update_frame():
    global camera_on, cap, emotion_counts
    if camera_on and cap:
        ret, frame = cap.read()
        if ret:
            # Convert the frame to RGB for tkinter display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape

                        # Calculate bounding box coordinates
                        x = max(0, int(bboxC.xmin * iw))
                        y = max(0, int(bboxC.ymin * ih))
                        w = min(iw - x, int(bboxC.width * iw))
                        h = min(ih - y, int(bboxC.height * ih))

                        # Ensure the bounding box is valid
                        if w > 0 and h > 0:
                            # Crop the face for emotion prediction
                            face = frame[y:y+h, x:x+w]

                            if face.size != 0:  # Ensure the face crop is valid
                                emotion = predict_emotion(face)

                                # Update emotion count (approximation: 1 frame = ~33ms for 30fps)
                                emotion_counts[emotion] += 1 / 30.0

                                # Draw bounding box
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                                # Draw emotion label
                                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert the frame to ImageTk format
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            video_label.config(image=img)
            video_label.image = img

        # Schedule the next frame update
        video_label.after(10, update_frame)

# Create the GUI using tkinter
root = tk.Tk()
root.title("Rozpoznawanie emocji")
root.geometry("1000x600")

# Create frames for layout
left_frame = tk.Frame(root, width=800, height=600)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root, width=200, height=600, bg="lightgray")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

# Video feed label
video_label = tk.Label(left_frame)
video_label.pack(padx=10, pady=10)

# Buttons in the right frame
start_button = tk.Button(right_frame, text="Włącz kamerę", command=start_camera, width=20, height=2)
start_button.pack(pady=20)

stop_button = tk.Button(right_frame, text="Wyłącz kamerę", command=stop_camera, width=20, height=2)
stop_button.pack(pady=20)

report_button = tk.Button(right_frame, text="Wyświetl raport", command=display_report, width=20, height=2)
report_button.pack(pady=20)

chart_button = tk.Button(right_frame, text="Wyświetl wykres", command=display_chart, width=20, height=2)
chart_button.pack(pady=20)

exit_button = tk.Button(right_frame, text="Zamknij", command=root.quit, width=20, height=2)
exit_button.pack(pady=20)

root.mainloop()
