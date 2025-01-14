# Importy pozostają bez zmian
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
emotions = ["Zlosc", "Zniesmaczenie", "Strach", "Radosc", "Smutek", "Zaskoczenie", "Neutralny"]

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Global variables
camera_on = False
cap = None
emotion_counts = {emotion: 0 for emotion in emotions}  # Count durations for each emotion
start_time = None  # Time when the recording starts
current_emotion = None

report_window = None
chart_window = None
time_report_window = None

emotion_time_intervals = []  # List of tuples to store (start_time, end_time, emotion)

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
    global camera_on, cap, start_time, emotion_counts, emotion_time_intervals, current_emotion
    if not camera_on:
        # Reset data for the new recording session
        emotion_counts = {emotion: 0 for emotion in emotions}
        start_time = time.time()
        emotion_time_intervals = []
        current_emotion = None

        camera_on = True
        cap = cv2.VideoCapture(0)
        update_frame()

def stop_camera():
    global camera_on, cap, emotion_time_intervals, current_emotion, start_time, session_end_time
    camera_on = False
    current_emotion = None

    # Finalizowanie przedziałów czasowych emocji
    if emotion_time_intervals and emotion_time_intervals[-1][1] is None:
        emotion_time_intervals[-1] = (emotion_time_intervals[-1][0], time.time(), emotion_time_intervals[-1][2])

    # Zwolnienie kamery
    if cap and cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()

    # Czarny placeholder dla obrazu po zatrzymaniu kamery
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    black_image = ImageTk.PhotoImage(Image.fromarray(black_frame))
    video_label.config(image=black_image)
    video_label.image = black_image

    # Zatrzymanie aktualizacji obrazu
    video_label.after_cancel(update_frame)

    # Ustawienie końcowego czasu sesji
    session_end_time = time.time()

def display_report():
    global emotion_counts, start_time, session_end_time, report_window
    if report_window is not None and report_window.winfo_exists():
        report_window.lift()  # Jeśli okno istnieje, przywróć je na wierzch
        return

    if start_time is None:
        return

    # Użyj końcowego czasu sesji, jeśli kamera jest wyłączona
    end_time = session_end_time if not camera_on else time.time()
    total_time = end_time - start_time
    if total_time == 0:
        return

    percentages = {emotion: (duration / total_time) * 100 for emotion, duration in emotion_counts.items()}
    total_percentage = sum(percentages.values())
    if total_percentage > 0:
        percentages = {emotion: (value / total_percentage) * 100 for emotion, value in percentages.items()}

    report_window = tk.Toplevel(root)
    report_window.title("Raport z emocji")
    report_window.geometry("500x400")
    report_window.wm_attributes("-topmost", True)
    report_window.protocol("WM_DELETE_WINDOW", lambda: reset_window_reference("report"))

    duration_label = tk.Label(report_window, text=f"Czas trwania sesji: {total_time:.2f} sekund", font=("Arial", 14))
    duration_label.pack(pady=10)

    columns = ("Emocja", "Czas [%]")
    tree = ttk.Treeview(report_window, columns=columns, show="headings", height=8)

    tree.heading("Emocja", text="Emocja")
    tree.heading("Czas [%]", text="Czas [%]")

    tree.column("Emocja", width=200, anchor=tk.CENTER)
    tree.column("Czas [%]", width=100, anchor=tk.CENTER)

    for emotion, percentage in percentages.items():
        tree.insert("", tk.END, values=(emotion, f"{percentage:.2f}%"))

    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

    button_frame = tk.Frame(report_window)
    button_frame.pack(pady=10)

    save_button = tk.Button(button_frame, text="Zapisz", command=save_report, width=15)
    save_button.pack(side=tk.LEFT, padx=5)

    close_button = tk.Button(button_frame, text="Zamknij", command=report_window.destroy, width=15)
    close_button.pack(side=tk.LEFT, padx=5)

def display_chart():
    global emotion_counts, start_time, chart_window
    if chart_window is not None and chart_window.winfo_exists():
        chart_window.lift()  # Jeśli okno istnieje, przywróć je na wierzch
        return

    if start_time is None:
        return

    total_time = time.time() - start_time
    if total_time == 0:
        return

    percentages = {emotion: (duration / total_time) * 100 for emotion, duration in emotion_counts.items()}
    total_percentage = sum(percentages.values())
    if total_percentage > 0:
        percentages = {emotion: (value / total_percentage) * 100 for emotion, value in percentages.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(percentages.keys(), percentages.values(), color=['red', 'brown', 'purple', 'yellow', 'blue', 'orange', 'green'])
    ax.set_xlabel('Emocje')
    ax.set_ylabel('Procentowy udział')
    ax.set_title('Rozkład emocji')
    ax.set_xticks(range(len(percentages.keys())))
    ax.set_xticklabels(percentages.keys(), rotation=45)
    plt.tight_layout()

    chart_window = tk.Toplevel(root)
    chart_window.title("Wykres emocji")
    chart_window.geometry("700x500")
    chart_window.wm_attributes("-topmost", True)
    chart_window.protocol("WM_DELETE_WINDOW", lambda: reset_window_reference("chart"))

    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    def save_chart():
        file_path = fd.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("Obrazy PNG", "*.png")],
                                         title="Zapisz wykres")
        if file_path:
            fig.savefig(file_path)

    button_frame = tk.Frame(chart_window)
    button_frame.pack(pady=10)

    save_button = tk.Button(button_frame, text="Zapisz", command=save_chart, width=15)
    save_button.pack(side=tk.LEFT, padx=5)

    close_button = tk.Button(button_frame, text="Zamknij", command=chart_window.destroy, width=15)
    close_button.pack(side=tk.LEFT, padx=5)

def display_time_based_report():
    global emotion_time_intervals, start_time, time_report_window
    if time_report_window is not None and time_report_window.winfo_exists():
        time_report_window.lift()  # Jeśli okno istnieje, przywróć je na wierzch
        return

    if not emotion_time_intervals:
        return

    # Sort intervals by start time and merge them sequentially
    emotion_time_intervals.sort(key=lambda x: x[0])
    merged_intervals = []

    for start, end, emotion in emotion_time_intervals:
        if merged_intervals and merged_intervals[-1][1] and merged_intervals[-1][1] > start:
            merged_intervals[-1] = (merged_intervals[-1][0], start, merged_intervals[-1][2])

        if not merged_intervals or merged_intervals[-1][2] != emotion:
            adjusted_start = merged_intervals[-1][1] + 0.001 if merged_intervals and merged_intervals[-1][1] else start
            merged_intervals.append((adjusted_start, end, emotion))
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], end, emotion)

    time_report_window = tk.Toplevel(root)
    time_report_window.title("Raport czas/emocje")
    time_report_window.geometry("520x450")
    time_report_window.wm_attributes("-topmost", True)
    time_report_window.protocol("WM_DELETE_WINDOW", lambda: reset_window_reference("time_report"))

    report_label = tk.Label(time_report_window, text="Raport czasowy emocji", font=("Arial", 14))
    report_label.pack(pady=10)

    # Frame to hold the Treeview and Scrollbar
    tree_frame = tk.Frame(time_report_window)
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Treeview for displaying data
    columns = ("Czas rozpoczęcia", "Czas zakończenia", "Emocja")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10)

    tree.heading("Czas rozpoczęcia", text="Czas rozpoczęcia")
    tree.heading("Czas zakończenia", text="Czas zakończenia")
    tree.heading("Emocja", text="Emocja")

    tree.column("Czas rozpoczęcia", width=150, anchor=tk.CENTER)
    tree.column("Czas zakończenia", width=150, anchor=tk.CENTER)
    tree.column("Emocja", width=100, anchor=tk.CENTER)

    # Vertical scrollbar
    v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=v_scrollbar.set)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    for start, end, emotion in merged_intervals:
        start_time_str = time.strftime("%H:%M:%S", time.gmtime(start - start_time)) + f".{int((start - int(start)) * 1000):03d}"
        end_time_str = time.strftime("%H:%M:%S", time.gmtime(end - start_time) if end else time.time()) + f".{int((end - int(end)) * 1000):03d}"
        tree.insert("", tk.END, values=(start_time_str, end_time_str, emotion))

    tree.pack(fill=tk.BOTH, expand=True)

    # Buttons to save or close the report
    button_frame = tk.Frame(time_report_window)
    button_frame.pack(pady=10)

    def save_report():
        file_path = fd.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Pliki tekstowe", "*.txt")],
                                         title="Zapisz raport czasowy")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Raport czasowy emocji:\n")
                for start, end, emotion in merged_intervals:
                    start_time_str = time.strftime("%H:%M:%S", time.gmtime(start - start_time)) + f".{int((start - int(start)) * 1000):03d}"
                    end_time_str = time.strftime("%H:%M:%S", time.gmtime(end - start_time) if end else time.time()) + f".{int((end - int(end)) * 1000):03d}"
                    f.write(f"{start_time_str} - {end_time_str}: {emotion}\n")

    save_button = tk.Button(button_frame, text="Zapisz", command=save_report, width=15)
    save_button.pack(side=tk.LEFT, padx=5)

    close_button = tk.Button(button_frame, text="Zamknij", command=time_report_window.destroy, width=15)
    close_button.pack(side=tk.LEFT, padx=5)


def reset_window_reference(window_type):
    global report_window, chart_window, time_report_window
    if window_type == "report":
        report_window = None
    elif window_type == "chart":
        chart_window = None
    elif window_type == "time_report":
        time_report_window = None

# Function to update the video feed
def update_frame():
    global camera_on, cap, emotion_counts, current_emotion, emotion_time_intervals
    if camera_on and cap:
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape

                        x = max(0, int(bboxC.xmin * iw))
                        y = max(0, int(bboxC.ymin * ih))
                        w = min(iw - x, int(bboxC.width * iw))
                        h = min(ih - y, int(bboxC.height * ih))

                        if w > 0 and h > 0:
                            face = frame[y:y + h, x:x + w]
                            if face.size != 0:
                                emotion = predict_emotion(face)

                                if current_emotion != emotion:
                                    if current_emotion is not None:
                                        emotion_time_intervals[-1] = (emotion_time_intervals[-1][0], time.time(), current_emotion)
                                    emotion_time_intervals.append((time.time(), None, emotion))

                                current_emotion = emotion

                                emotion_counts[emotion] += 1 / 30.0
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            video_label.config(image=img)
            video_label.image = img

        video_label.after(10, update_frame)

# Create the GUI using tkinter

def on_enter(event):
    event.widget.config(bg="#d9d9d9")  # Jasnoszary kolor przy hover

def on_leave(event):
    event.widget.config(bg="SystemButtonFace")  # Przywrócenie domyślnego koloru


root = tk.Tk()
root.title("Rozpoznawanie emocji")
root.geometry("1000x600")

style = ttk.Style()
style.configure("Custom.TLabelframe.Label", font=("Arial", 14, "bold"), foreground="navy")

left_frame = tk.Frame(root, width=800, height=600)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_frame = tk.Frame(root, width=200, height=600, bg="lightgray")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

video_label = tk.Label(left_frame)
video_label.grid(row=0, column=0, sticky="nsew")  # Wyśrodkowanie we wszystkich kierunkach

left_frame.rowconfigure(0, weight=1)
left_frame.columnconfigure(0, weight=1)

camera_frame = ttk.LabelFrame(right_frame, text="Kontrola kamery", padding=(10, 10), style="Custom.TLabelframe")
camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

camera_inner_frame = tk.Frame(camera_frame)
camera_inner_frame.pack(expand=True)

start_button = tk.Button(camera_inner_frame, text="Włącz kamerę", command=start_camera, width=20, height=2)
start_button.pack(pady=10)
start_button.bind("<Enter>", on_enter)
start_button.bind("<Leave>", on_leave)

stop_button = tk.Button(camera_inner_frame, text="Wyłącz kamerę", command=stop_camera, width=20, height=2)
stop_button.pack(pady=10)
stop_button.bind("<Enter>", on_enter)
stop_button.bind("<Leave>", on_leave)

report_frame = ttk.LabelFrame(right_frame, text="Raporty i wykres", padding=(10, 10), style="Custom.TLabelframe")
report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

report_inner_frame = tk.Frame(report_frame)
report_inner_frame.pack(expand=True)

report_button = tk.Button(report_inner_frame, text="Raport procentowy", command=display_report, width=20, height=2)
report_button.pack(pady=10)
report_button.bind("<Enter>", on_enter)
report_button.bind("<Leave>", on_leave)

chart_button = tk.Button(report_inner_frame, text="Wykres", command=display_chart, width=20, height=2)
chart_button.pack(pady=10)
chart_button.bind("<Enter>", on_enter)
chart_button.bind("<Leave>", on_leave)

report_time_button = tk.Button(report_inner_frame, text="Raport czas/emocje", command=display_time_based_report, width=20, height=2)
report_time_button.pack(pady=10)
report_time_button.bind("<Enter>", on_enter)
report_time_button.bind("<Leave>", on_leave)

exit_frame = ttk.LabelFrame(right_frame, text="Zamknięcie aplikacji", padding=(10, 10), style="Custom.TLabelframe")
exit_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

exit_inner_frame = tk.Frame(exit_frame)
exit_inner_frame.pack(expand=True)

exit_button = tk.Button(exit_inner_frame, text="Zamknij", command=root.quit, width=20, height=2)
exit_button.pack(pady=10)
exit_button.bind("<Enter>", on_enter)
exit_button.bind("<Leave>", on_leave)

root.mainloop()
