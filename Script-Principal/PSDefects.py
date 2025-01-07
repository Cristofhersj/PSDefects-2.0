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
import csv
from datetime import datetime, timedelta  # Libraries for date and time handling

# Global variables
reference_images = []  # List to store reference images captured
ser = None  # Serial connection object
start_time = None  # Variable to store the start time

csv_file = '/home/soporte/.ESS_Vinyl_Inspector/Sources/usage_log.csv'
update_interval_ms = 60000  # 5 minutes in milliseconds
end_time = None
timer_thread = None
logging_active = False
continue_updating = False
balance_auto_state = "Continuous"


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
    camera.ExposureTime.SetValue(5000)  # Set the exposure time of the camera
    camera.TriggerSelector.SetValue("FrameStart")  # Set the trigger selector to start frame capture
    camera.TriggerMode.SetValue("On")  # Enable trigger mode
    camera.TriggerSource.SetValue("Line1")  # Set the trigger source to Line1
    camera.BslBrightness.SetValue(0) # Set brightness
    camera.BslLightSourcePreset.SetValue("FactoryLED6000K")
    camera.BslLightSourcePresetFeatureSelector.SetValue("ColorAdjustment")
    camera.BalanceWhiteAuto.SetValue("Continuous")
   

# Function to toggle the BalanceWhiteAuto setting
def toggle_white_balance():
    global balance_auto_state, cameras
    for camera in cameras:
        if balance_auto_state == "Continuous":
            camera.BalanceWhiteAuto.SetValue("Off")  # Turn off BalanceWhiteAuto
            camera.BalanceRatioSelector.SetValue("Red")
            camera.BalanceRatio.SetValue(1.0)
            camera.BalanceRatioSelector.SetValue("Green")
            camera.BalanceRatio.SetValue(1.1)
            camera.BalanceRatioSelector.SetValue("Blue")
            camera.BalanceRatio.SetValue(1.5)
            print(f"Camera {camera.GetDeviceInfo().GetFriendlyName()}: white balance set to off and configured manually")
    
        else:
            camera.BalanceWhiteAuto.SetValue("Continuous")  # Set BalanceWhiteAuto to Continuous
            
            print(f"Camera {camera.GetDeviceInfo().GetFriendlyName()}: white balance set to continuous and configured automatically")
    if balance_auto_state == "Continuous":
        balance_auto_state = "Off"
    else:
        balance_auto_state = "Continuous"
    
def save_start_time():
    """Saves the start time to a new line in the CSV."""
    global continue_updating
    continue_updating = True  # Start periodic updates

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if the file exists and if it needs a header
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if file is new
            writer.writerow(['Start Time', 'End Time', 'Total Usage (Minutes)'])
        # Write a new row with start time, leaving end time and total usage empty
        writer.writerow([start_time, '', ''])

    # Start periodic updates using root.after
    root.after(update_interval_ms, update_end_time)

def update_end_time():
    """Updates the last line with the current end time and total usage."""
    if not continue_updating:
        return  # Stop updating if the flag is false

    end_time = datetime.now()
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # Read all rows from the CSV
    with open(csv_file, 'r', newline='') as file:
        reader = list(csv.reader(file))

    if len(reader) < 2:
        print("No start time found to update.")
        return

    # Update the last row
    last_row = reader[-1]
    start_time_str = last_row[0]
    if start_time_str:
        # Calculate total usage
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        total_usage = (end_time - start_time).total_seconds() / 60  # Total usage in minutes
        
        # Update end time and total usage in the last row
        last_row[1] = end_time_str  # Update end time
        last_row[2] = f"{total_usage:.2f}"  # Update total usage

        # Write all rows back to the CSV
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(reader)
    else:
        print("End time already recorded or start time missing.")

    # Schedule the next update
    root.after(update_interval_ms, update_end_time)

def save_final_end_time():
    """Saves the final end time when the application is closed."""
    global continue_updating
    continue_updating = False  # Stop periodic updates

    end_time = datetime.now()
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # Read all rows from the CSV
    with open(csv_file, 'r', newline='') as file:
        reader = list(csv.reader(file))

    if len(reader) < 2:
        print("No start time found to finalize.")
        return

    # Update the last row
    last_row = reader[-1]
    start_time_str = last_row[0]
    if start_time_str:
        # Calculate total usage
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        total_usage = (end_time - start_time).total_seconds() / 60  # Total usage in minutes
        last_row[1] = end_time_str
        last_row[2] = f"{total_usage:.2f}"

        # Write all rows back to the CSV
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(reader)
    else:
        print("End time already recorded or start time missing.")

# Start capturing images
def start_capture():
    global cameras, root, Captura, reference_images, ser, start_time
    save_start_time()
    update_end_time()
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
    captura_button.pack()  # Show the Pause button
    captura_button.pack(pady=(5, 5), before=toggle_white_balance_button)

# Stop capturing images
def stop_capture():
    global cameras, Captura, start_time, time_logs
    Captura = False  # Stop capturing
    cameras = []  # Reset the cameras list
    
    for label in image_labels:
        label.configure(image="")  # Clear the image labels
    if ser:
        ser.write(b'P')  # Send 'P' command to stop the cycle
        print("Command 'P' sent through serial.")
    save_final_end_time()
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

                # Display the first image in the defects found section
                if capture_counts[i] == 0:  # Check if this is the first image
                    first_image_filename = f"/home/soporte/.ESS_Vinyl_Inspector/Sources/Defects_Found/camera_{i+1}_first_image.jpg"
                    cv2.imwrite(first_image_filename, current_image)  # Save the first image
                    first_image_display = cv2.resize(current_image, (320, 320))  # Resize for display
                    img_first = Image.fromarray(cv2.cvtColor(first_image_display, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
                    img_first = ImageTk.PhotoImage(img_first)  # Convert to ImageTk for display
                    latest_image_labels[i].configure(image=img_first)  # Update label with the first image
                    latest_image_labels[i].image = img_first  # Keep a reference to avoid garbage collection
                
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
            print(f"Exception during image grab: {e}")  # Log exception during image grabbing

    root.after(update_frequency, update_images)  # Schedule next update

# Toggle image capturing state
def toggle_captura():
    global Captura, cameras
    Captura = not Captura  # Toggle the capturing state
    if Captura:
        for camera in cameras:
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # Start grabbing latest image only
        captura_button.config(text="Pausar")  # Change button text to "Pausar"
    else:
        for camera in cameras:
            camera.StopGrabbing()  # Stop grabbing images
        captura_button.config(text="Reanudar")  # Change button text to "Reanudar"
    update_images()  # Update images

# Open the folder containing defect images
def open_defects_folder():
    folder_path = "/home/soporte/.ESS_Vinyl_Inspector/Sources/Defects_Found"  # Path to the defects folder
    try:
        subprocess.run(['xdg-open', folder_path], check=True)  # Open the folder using xdg-open
    except Exception as e:
        print(f"Failed to open folder: {e}")  # Log error if folder opening fails


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
def process_yolo_detection(img, model, conf_threshold=0.2, y_threshold=200, size_threshold=10, location_threshold=20):
    # Original image dimensions
    img_height, img_width = img.shape[:2]

    # Resize the image to the model's input size (640x640)
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert the image to RGB

    results = model(img_rgb)  # Run YOLO detection on the resized image

    defects_found = False  # Flag to indicate if defects are found
    defect_locations = []  # List to store defect locations
    previous_defects = []  # Initialize list to store previous defects

    # Iterate through detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()  # Get the confidence score
            if conf > conf_threshold:  # Check if the confidence exceeds the threshold
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get the bounding box coordinates

                # Adjust the bounding box coordinates to match the original image size
                x1 = int(x1 / 640 * img_width)
                y1 = int(y1 / 640 * img_height)
                x2 = int(x2 / 640 * img_width)
                y2 = int(y2 / 640 * img_height)

                if y1 >= y_threshold and y2 <= (img_height - y_threshold):  # Check if the box is within vertical limits
                    new_defect = (x1, y1, x2, y2)

                    # Compare with previous defects
                    similar = False
                    for prev_defect in previous_defects:
                        x1_prev, y1_prev, x2_prev, y2_prev = prev_defect
                        new_size = (x2 - x1) * (y2 - y1)
                        prev_size = (x2_prev - x1_prev) * (y2_prev - y1_prev)
                        new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        prev_center = ((x1_prev + x2_prev) / 2, (y1_prev + y2_prev) / 2)

                        # Check if the sizes and positions are similar
                        if abs(new_size - prev_size) <= size_threshold and np.linalg.norm(np.array(new_center) - np.array(prev_center)) <= location_threshold:
                            similar = True
                            break

                    # Only add new defects if they are not similar to previous ones
                    if not similar:
                        defects_found = True  # Set defects found flag
                        defect_locations.append(new_defect)  # Add the bounding box coordinates to the list
                        previous_defects.append(new_defect)  # Update previous defects list

    # Draw bounding boxes on the original image
    for (x1, y1, x2, y2) in defect_locations:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
yolo_model = YOLO('/home/soporte/.ESS_Vinyl_Inspector/Sources/best.pt')  # or the path to your custom-trained model
if torch.cuda.is_available():
    print("CUDA is available. GPU acceleration will be used.")
else:
    print("CUDA is not available. GPU acceleration is not possible.")
setup_serial()

# Initialize the Tkinter window
root = Tk()
root.title("Camera Capture")
root.configure(bg="#2e2e2e")  # Dark grey background


update_frequency = 10
reference_images = []  # To store reference images captured
capture_counts = [0] * 4
Captura = False
latest_defect_images = [None] * 4  # Assuming 4 cameras
cameras = []
prev_defect_locations = [None] * 4  # To store previous defect locations for each camera
latest_image_labels = []  # For displaying latest saved images
image_labels = []


# Create a top title label
top_title_label = Label(root, text="Último Defecto Detectado", font=("Helvetica", 20), bg="#2e2e2e", fg="#ffffff")
top_title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="n")  # Align at the top

# Frame for latest images
latest_images_frame = Frame(root, bg="#2e2e2e")
latest_images_frame.grid(row=1, column=0, pady=(5, 0), sticky="ew")

# Label for real-time camera view
ultimo_defecto_label = Label(root, text="Cámaras en Tiempo Real", font=("Helvetica", 16), bg="#2e2e2e", fg="#ffffff")
ultimo_defecto_label.grid(row=2, column=0, pady=(5, 0), sticky="n")  # Place label below the latest images frame

# Frame for real-time camera view
camera_images_frame = Frame(root, bg="#2e2e2e")
camera_images_frame.grid(row=3, column=0, pady=(5, 0), sticky="ew")  # Place below the label

# Initialize labels for each camera feed
image_labels = []
for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))
    img = ImageTk.PhotoImage(img)
    label = Label(camera_images_frame, image=img, bg="#2e2e2e", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1)
    label.pack(side=LEFT)
    image_labels.append(label)

# Prepare labels for latest saved images
latest_image_labels = []
for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))  # Smaller size for latest images
    img = ImageTk.PhotoImage(img)
    label = Label(latest_images_frame, image=img, bg="#2e2e2e", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1)
    label.pack(side=LEFT, padx=(5, 0))
    latest_image_labels.append(label)

# Create a frame for buttons to position them on the right
buttons_frame = Frame(root, bg="#2e2e2e")
buttons_frame.grid(row=2, column=2, rowspan=3, padx=(10, 10), sticky="ns")  # Place the frame on the far right

# Adjust button placements
start_button = Button(buttons_frame, text="Iniciar", command=start_capture, bg="#2e2e2e", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1, fg="#ffffff")
start_button.pack(pady=(5, 5))

# Create the pause button but keep it hidden initially
captura_button = Button(buttons_frame, text="Pausar", command=toggle_captura, bg="#2e2e2e", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1, fg="#ffffff")
# Initially hidden (not packed yet)

# Add other buttons
toggle_white_balance_button = Button(buttons_frame, text="Ajustar Colores", command=toggle_white_balance, bg="#2e2e2e", fg="#ffffff", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1)
toggle_white_balance_button.pack(pady=(5, 5))

open_folder_button = Button(buttons_frame, text="Abrir Carpeta con Defectos", command=open_defects_folder, bg="#2e2e2e", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1, fg="#ffffff")
open_folder_button.pack(pady=(5, 5))

compose_email_button = Button(buttons_frame, text="Reportar Error", command=compose_email, bg="#2e2e2e",highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1,fg="#ffffff")
compose_email_button.pack(pady=(5, 5))

stop_button = Button(buttons_frame, text="Cerrar", command=stop_capture, bg="#2e2e2e", highlightbackground="#2e2e2e", highlightcolor="#2e2e2e", highlightthickness=1, fg="#ffffff")
stop_button.pack(pady=(5, 5))


# Run the Tkinter main loop
root.mainloop()
