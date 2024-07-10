#!/home/soporte/.ESS_Vinyl_Inspector/Sources/.venv/bin/python3
# coding=utf-8

# Import necessary libraries
from tkinter import *  # GUI library for creating windows, buttons, labels, etc.
import cv2  # OpenCV library for image processing
from PIL import Image, ImageTk  # Libraries for handling images
import os  # Library for interacting with the operating system
import subprocess  # Library for running external processes
import time  # Library for time-related functions
import serial.tools.list_ports  # Library for handling serial ports
from pypylon import pylon  # Pylon library for Basler cameras
import smtplib  # Library for sending emails
from email.mime.text import MIMEText  # Library for creating email text parts
from email.mime.multipart import MIMEMultipart  # Library for creating multipart email messages
from ultralytics import YOLO  # Library for YOLO object detection
import torch  # PyTorch library, which YOLO uses

# Global variables
reference_images = []  # List to store reference images captured
ser = None  # Serial connection object

# Setup the serial connection
def setup_serial():
    global ser
    arduino_port = find_arduino_port()  # Find the Arduino port
    if arduino_port:
        ser = serial.Serial(arduino_port, 9600, timeout=1)  # Establish the serial connection at 9600 baud rate
        time.sleep(2)  # Wait for the connection to stabilize
        print("Serial connection established.")

# Find the Arduino port
def find_arduino_port():
    available_ports = serial.tools.list_ports.comports()  # List all available serial ports

    for port in available_ports:
        print(port.device, "-", port.description)  # Print each available port and its description

    arduino_ports = [
        port.device
        for port in available_ports
        if 'ttyACM0' in port.description  # Check if 'ttyACM0' is in the port description to identify the Arduino
    ]

    if arduino_ports:
        arduino_port = arduino_ports[0]
        print("Arduino found on port:", arduino_port)
        return arduino_port  # Return the first Arduino port found
    else:
        print("No Arduino found")
        return None  # Return None if no Arduino is found

# Setup camera settings
def setup_camera(camera):
    camera.ExposureTime.SetValue(1500)  # Set the exposure time of the camera
    camera.TriggerSelector.SetValue("FrameStart")  # Set the trigger selector to start frame capture
    camera.TriggerMode.SetValue("On")  # Enable trigger mode
    camera.TriggerSource.SetValue("Line1")  # Set the trigger source to Line1

# Start capturing images
def start_capture():
    global cameras, root, Captura, reference_images, ser
    reference_images = Referencia()  # Capture and store reference images
    tl_factory = pylon.TlFactory.GetInstance()  # Get the transport layer factory for camera enumeration
    devices = tl_factory.EnumerateDevices()  # Enumerate all connected devices (cameras)
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")  # Raise an error if no camera is found

    cameras = []
    for i, device in enumerate(devices):
        camera = pylon.InstantCamera(tl_factory.CreateDevice(device))  # Create a camera object for each device
        camera.Open()  # Open the camera
        setup_camera(camera)  # Setup camera with predefined settings
        cameras.append(camera)  # Add the camera to the cameras list
        print(f"Camera {i+1}: Exposure time set to 5000 and configured for hardware trigger on Line 1")

    if ser:
        ser.write(b'F')  # Send 'F' only once to initiate the capture cycle
        print("Command 'F' sent. Cycle started.")
    
    start_button.config(state=DISABLED)  # Disable start button
    start_button.destroy()  # Remove start button from the UI
    toggle_captura()  # Start capturing images automatically
    captura_button.pack(side=LEFT)  # Show the Pause button

# Stop capturing images
def stop_capture():
    global cameras, Captura
    Captura = False  # Stop capturing
    cameras = []  # Reset the cameras list
    for label in image_labels:
        label.configure(image="")  # Clear the image labels
    if ser:
        ser.write(b'P')  # Send 'P' command to stop the cycle
        print("Command 'P' sent through serial.")
    root.destroy()  # Close the UI

# Update images from cameras
def update_images():
    global reference_images, capture_counts, update_frequency, prev_defect_locations
    if not Captura:
        for camera in cameras:
            if camera.IsGrabbing():
                camera.StopGrabbing()  # Stop grabbing if capture is not active
        return

    for i, camera in enumerate(cameras):
        if not Captura:
            if camera.IsGrabbing():
                camera.StopGrabbing()  # Ensure grabbing stops if capture is not active
            continue

        try:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)  # Retrieve result with timeout
            if grabResult.GrabSucceeded():
                current_image = grabResult.GetArray()  # Get the image array
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BAYER_BG2BGR)  # Convert to BGR color space

                processed_image, defects_found, defect_locations = process_yolo_detection(current_image, yolo_model)

                processed_display_image = cv2.resize(processed_image, (320, 320))  # Resize image for display
                img_display = Image.fromarray(cv2.cvtColor(processed_display_image, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
                img_display = ImageTk.PhotoImage(img_display)  # Convert to ImageTk for display
                image_labels[i].configure(image=img_display)  # Update label with new image
                image_labels[i].image = img_display

                if defects_found:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp
                    filename = f"/home/soporte/.ESS_Vinyl_Inspector/Sources/Defects_Found/camera_{i+1}_defect_{timestamp}.jpg"
                    
                    # Compress and save the image
                    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # Set compression parameters
                    cv2.imwrite(filename, processed_image, compression_params)  # Save the processed image

                    processed_defect_image = cv2.resize(processed_image, (320, 320))  # Resize for latest defect display
                    img_defect = Image.fromarray(cv2.cvtColor(processed_defect_image, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
                    img_defect = ImageTk.PhotoImage(img_defect)  # Convert to ImageTk for display
                    latest_image_labels[i].configure(image=img_defect)  # Update label with new defect image
                    latest_image_labels[i].image = img_defect

                    if ser:
                        ser.write(b'G')  # Send 'G' command through serial if defects are found

                capture_counts[i] += 1  # Increment the capture count
            else:
                print(f"Error: Camera {i+1} failed to grab.")  # Log error if image grab failed
            grabResult.Release()  # Release the grab result
        except Exception as e:
            print(f"Exception during image grab: {e}")  # Log exception during image grab

    if Captura:
        root.after(1, update_images)  # Schedule the next update if capture is still active

# Toggle image capturing on and off
def toggle_captura():
    global Captura, captura_button
    Captura = not Captura  # Toggle capture state
    if Captura:
        for camera in cameras:
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)  # Start grabbing images one by one
        update_images()  # Start updating images
        captura_button.config(text="Pausar")  # Update button text to "Pausar" when capturing
    else:
        for camera in cameras:
            camera.StopGrabbing()  # Stop grabbing images
        captura_button.config(text="Reanudar")  # Update button text to "Reanudar" when paused

# Open the folder containing defect images
def open_defects_folder():
    folder_path = "/home/soporte/.ESS_Vinyl_Inspector/Sources/Defects_Found"  # Path to the defects folder
    try:
        subprocess.run(['xdg-open', folder_path], check=True)  # Open the folder using the default file manager
    except subprocess.CalledProcessError as e:
        print(f"Failed to open the folder: {e}")  # Log error if folder opening fails

# Capture and store reference images from all cameras
def Referencia():
    reference_images = []  # List to store captured reference images
    for i, camera in enumerate(cameras):
        camera.StartGrabbing(pylon.GrabStrategy_OneByOne)  # Start grabbing images one by one
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)  # Retrieve result with timeout
        if grabResult.GrabSucceeded():
            reference_image = grabResult.GetArray()  # Get the image array
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BAYER_BG2BGR)  # Convert to BGR color space
            reference_images.append(reference_image)  # Store the reference image
        else:
            print(f"Error: Camera {i+1} failed to grab reference image.")  # Log error if image grab failed
        grabResult.Release()  # Release the grab result

    for i, ref_img in enumerate(reference_images):
        ref_img = cv2.resize(ref_img, (640, 640))  # Resize reference images for saving
        filename = f"/home/soporte/.ESS_Vinyl_Inspector/Sources/Referencias/camera_{i+1}_reference.jpg"
        cv2.imwrite(filename, ref_img)  # Save reference image
        print(f"Saved: {filename}")

    return reference_images  # Return the list of captured images

# Send an email with the specified subject and body
def send_email(subject, message_body):
    sender_email = "reportesproquinaless@gmail.com"
    receiver_email = "cristofher.solis@scientificmodeling.com"
    password = "olyi ulxz vane fmdr"

    email = MIMEMultipart()
    email["From"] = sender_email
    email["To"] = receiver_email
    email["Subject"] = subject

    email.attach(MIMEText(message_body, "plain"))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Setup SMTP server
        server.starttls()  # Start TLS for security
        server.login(sender_email, password)  # Login to the email account
        text = email.as_string()
        server.sendmail(sender_email, receiver_email, text)  # Send the email
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")  # Log error if email sending fails
    finally:
        server.quit()  # Close the SMTP server connection

# Process YOLO detection on an image
def process_yolo_detection(img, model, conf_threshold=0.3, y_threshold=200):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
    results = model(img_rgb)  # Run YOLO detection on the image

    defects_found = False  # Flag to indicate if defects are found
    defect_locations = []  # List to store defect locations

    img_height, img_width = img.shape[:2]  # Get the image dimensions

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()  # Get the confidence score
            if conf > conf_threshold:  # Check if the confidence exceeds the threshold
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get the bounding box coordinates
                if y1 >= y_threshold and y2 <= (img_height - y_threshold):  # Check if the box is within vertical limits
                    defects_found = True  # Set defects found flag
                    defect_locations.append((x1, y1, x2, y2))  # Add the bounding box coordinates to the list

    for (x1, y1, x2, y2) in defect_locations:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw bounding boxes on the image

    return img, defects_found, defect_locations  # Return the processed image and defect information

# Compose an email through a GUI
def compose_email():
    new_window = Toplevel()
    new_window.title("Reportar Error")
    new_window.geometry("400x300")

    Label(new_window, text="Asunto:").pack()  # Label for the subject entry
    subject_entry = Entry(new_window, width=50)  # Entry for the subject
    subject_entry.pack()

    Label(new_window, text="Describa el Error:").pack()  # Label for the message text
    message_text = Text(new_window, height=10, width=50)  # Text box for the message
    message_text.pack()

    def on_send():
        subject = subject_entry.get()
        message_body = message_text.get("1.0", "end-1c")
        send_email(subject, message_body)  # Send email with entered subject and message
        new_window.destroy()  # Close the new window

    send_button = Button(new_window, text="Enviar", command=on_send)  # Button to send the email
    send_button.pack()

# Setup serial connection at the beginning
yolo_model = YOLO('/home/soporte/.ESS_Vinyl_Inspector/Sources/best.pt')  # Load YOLO model
if torch.cuda.is_available():
    print("CUDA is available. GPU acceleration will be used.")
else:
    print("CUDA is not available. GPU acceleration is not possible.")
setup_serial()

# Initialize the Tkinter window
root = Tk()
root.title("Camera Capture")

# Create a top title label
top_title_label = Label(root, text="Último Defecto Detectado", font=("Helvetica", 20))
top_title_label.pack(side="top", pady=10)  # pady adds some padding above and below the label

# Frame for latest images
latest_images_frame = Frame(root)
latest_images_frame.pack(side=TOP, pady=(5, 0))
update_frequency = 10
reference_images = []  # To store reference images captured
capture_counts = [0] * 4
Captura = False
latest_defect_images = [None] * 4  # Assuming 4 cameras
cameras = []
prev_defect_locations = [None] * 4  # To store previous defect locations for each camera
latest_image_labels = []  # For displaying latest saved images
image_labels = []

captura_button = Button(root, text="Pausar", command=toggle_captura)

# Label for real-time camera view
ultimo_defecto_label = Label(root, text="Cámaras en Tiempo Real", font=("Helvetica", 16))
ultimo_defecto_label.pack(side=TOP, pady=(5, 0))

# Initialize labels for each camera feed
for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))
    img = ImageTk.PhotoImage(img)
    label = Label(root, image=img)
    label.pack(side=LEFT)
    image_labels.append(label)

# Prepare labels for latest saved images
for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))  # Smaller size for latest images
    img = ImageTk.PhotoImage(img)
    label = Label(latest_images_frame, image=img)
    label.pack(side=LEFT, padx=(5, 0))
    latest_image_labels.append(label)

# Add buttons for different functionalities
compose_email_button = Button(root, text="Reportar Error", command=compose_email)
compose_email_button.pack(pady=20)

start_button = Button(root, text="Iniciar", command=start_capture)
start_button.pack(side=TOP)

stop_button = Button(root, text="Cerrar", command=stop_capture)
stop_button.pack(side=TOP)

open_folder_button = Button(root, text="Abrir Carpeta con Defectos", command=open_defects_folder)
open_folder_button.pack(side=TOP)

# Run the Tkinter main loop
root.mainloop()

compose_email_button = Button(root, text="Reportar Error", command=compose_email)
compose_email_button.pack(pady=20)



start_button = Button(root, text="Iniciar", command=start_capture)
start_button.pack(side=TOP)

stop_button = Button(root, text="Cerrar", command=stop_capture)
stop_button.pack(side=TOP)

open_folder_button = Button(root, text="Abrir Carpeta con Defectos", command=open_defects_folder)
open_folder_button.pack(side=TOP)

root.mainloop()
