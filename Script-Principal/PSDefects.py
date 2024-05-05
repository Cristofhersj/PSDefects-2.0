#!/home/vioro/tr/.venv/bin/python3
# coding=utf-8

# The rest of your Python code follows
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import subprocess
import time
import serial.tools.list_ports
from pypylon import pylon
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Global variables
reference_images = []  # To store reference images captured
ser = None

def setup_serial():
    global ser
    arduino_port = find_arduino_port()
    if arduino_port:
        ser = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)
        print("Serial connection established.")

def find_arduino_port():
    available_ports = serial.tools.list_ports.comports()

    for port in available_ports:
        print(port.device, "-", port.description)

    arduino_ports = [
        port.device
        for port in available_ports
        if 'ttyACM0' in port.description
    ]

    if arduino_ports:
        arduino_port = arduino_ports[0]
        print("Arduino found on port:", arduino_port)
        return arduino_port
    else:
        print("No Arduino found")
        return None

def setup_camera(camera):
    camera.ExposureTime.SetValue(1500)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Line1")

# Example mapping of friendly names to camera numbers

def start_capture():
    global cameras, root, Captura, reference_images, ser
    reference_images = Referencia()  # Capture and store reference images
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    cameras = []
    for i, device in enumerate(devices):
        camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
        camera.Open()
        setup_camera(camera)
        cameras.append(camera)
        print(f"Camera {i+1}: Exposure time set to 5000 and configured for hardware trigger on Line 1")

    if ser:
        ser.write(b'F')  # Send 'F' only once when setting up the serial connection
        print("Command 'F' sent. Cycle started.")
    
    start_button.config(state=DISABLED)
    start_button.destroy()
    toggle_captura()  # Start capturing images automatically
    captura_button.pack(side=LEFT)  # Show the Pause button

def stop_capture():
    global cameras, Captura
    Captura = False
    cameras = []  # Reset the cameras list
    for label in image_labels:
        label.configure(image="")
    if ser:
                        ser.write(b'P')
                        print("Command 'P' sent through serial.")
    root.destroy()

def update_images():
    global reference_images, capture_counts, update_frequency
    # Check if capturing is paused right at the start to avoid unnecessary processing
    if not Captura:
        for camera in cameras:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        return  # Exit the function if capturing is paused

    for i, camera in enumerate(cameras):
        # Check again if capturing has been paused before trying to grab an image
        if not Captura:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            continue  # Skip to the next iteration
        
        try:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                current_image = grabResult.GetArray()
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BAYER_BG2BGR)
                current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

                if capture_counts[i] % 10 == 0:
                    reference_images[i] = current_image.copy()
                    print("Se actualizó la referencia")

                processed_image, defects_found = process_contours(current_image, reference_images[i])

                # Resize for display in the main image labels
                processed_display_image = cv2.resize(processed_image, (320, 320))
                img_display = Image.fromarray(processed_display_image).convert("RGB")
                img_display = ImageTk.PhotoImage(img_display)
                image_labels[i].configure(image=img_display)
                image_labels[i].image = img_display

                if defects_found:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"/home/vioro/tr/defects_found/camera_{i+1}_defect_{timestamp}.png"
                    cv2.imwrite(filename, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

                    # Resize for display in the latest image labels (second row)
                    # Adjust the size here to whatever is needed for the defect images
                    processed_defect_image = cv2.resize(processed_image, (320, 320))
                    img_defect = Image.fromarray(processed_defect_image).convert("RGB")
                    img_defect = ImageTk.PhotoImage(img_defect)
                    latest_image_labels[i].configure(image=img_defect)  # Assuming you want to use the same logic for the latest images
                    latest_image_labels[i].image = img_defect

                    print(f"Defect detected, saving image as: {filename}")
                    if ser:
                        ser.write(b'G')
                        print("Command 'G' sent through serial.")
                capture_counts[i] += 1
                #print("Cuenta de capturas: ", capture_counts)
            else:
                print(f"Error: Camera {i+1} failed to grab.")
            grabResult.Release()
        except Exception as e:
            print(f"Exception during image grab: {e}")
    
    # Schedule the next call of update_images only if capturing is not paused
    if Captura:
        root.after(1, update_images)


def toggle_captura():
    global Captura, captura_button
    Captura = not Captura
    if Captura:
        for camera in cameras:
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        update_images()
        captura_button.config(text="Pausar")  # Update text to "Pausar" when capturing
    else:
        for camera in cameras:
            camera.StopGrabbing()
        captura_button.config(text="Reanudar")  # Update text to "Reanudar" when paused

def open_defects_folder():
    folder_path = "/home/vioro/tr/defects_found"
    try:
        subprocess.run(['xdg-open', folder_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to open the folder: {e}")

def calculate_difference(img1, img2):
    diff = cv2.absdiff(img1, img2)
    _, diff = cv2.threshold(diff, 11, 255, cv2.THRESH_BINARY)
    return diff

def denoise_binary_image(binary_image, kernel_size=3):
    # Apply erosion to remove small white regions and protrusions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Apply dilation to add small white regions and broaden protrusions
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    return dilated_image

def process_contours(img1, reference, ksize=3, min_defect_size=50, max_defect_size=500):
    defects_found = False  # Flag to indicate if any defects were found
    # Convert input images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between the images
    diff = calculate_difference(img1_gray, reference_gray)

    # Perform morphological operations on the binary difference image
    closed = denoise_binary_image(diff)
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))

    # Find contours on the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours
    for c in contours:
        # Calculate the area of the contour
        contour_area = cv2.contourArea(c)
        
        # Check if the contour area is within the defined range
        if min_defect_size <= contour_area <= max_defect_size:
            x, y, w, h = cv2.boundingRect(c)
            # Draw rectangle for significant defects
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 3)
            defects_found = True  # Update flag since a defect was found
            
            
    return img1, defects_found
def Referencia(save_path='./captured_images', exposure_time=1500, num_images_per_camera=1):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    num_cameras = len(devices)
    cameras = pylon.InstantCameraArray(num_cameras)
    images_captured = []  # To store reference images captured

    for i, cam in enumerate(cameras):
        cam.Attach(tl_factory.CreateDevice(devices[i]))
        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        setup_camera(cam)  # Set up hardware trigger
        print(f"Camera {i+1}: Exposure time set to {exposure_time} and configured for hardware trigger on Line 1")

    if ser:
                        ser.write(b'F')
                        print("Command 'F' sent through serial.")
    cameras.StartGrabbing(pylon.GrabStrategy_OneByOne)

    try:
        while cameras.IsGrabbing() and len(images_captured) < num_cameras:
            grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            camera_index = grabResult.GetCameraContext()  # Identifies which camera provided the image
            if grabResult.GrabSucceeded():
                # Convert grabbed image to RGB
                img = grabResult.GetArray()
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                images_captured.append(img)  # Add the captured image to the list
            else:
                print(f"Error: Camera {camera_index+1} failed to grab.")
            grabResult.Release()

    finally:
        # Stop grabbing and close all cameras
        cameras.StopGrabbing()
        for cam in cameras:
            cam.Close()
        ser.write(b'P')
        print("Command 'P' sent. Cycle Ended.")

    # Save images as PNG
    for i, img in enumerate(images_captured):
        if img is not None:
            filename = os.path.join(save_path, f"camera_{i+1}_reference.png")
            cv2.imwrite(filename, img)
            print(f"Saved: {filename}")

    return images_captured  # Return the list of captured images


def send_email(subject, message_body):
    sender_email = "-----@gmail.com"
    receiver_email = "-----"
    password =  "----" 

    # Create the email head (sender, receiver, and subject)
    email = MIMEMultipart()
    email["From"] = sender_email
    email["To"] = receiver_email
    email["Subject"] = subject

    # Attach the email body
    email.attach(MIMEText(message_body, "plain"))

    # Connect to the SMTP server and send the email
    try:
        # Note: You might need to update the SMTP server and port for Gmail
        server = smtplib.SMTP('smtp.gmail.com', 587)  
        server.starttls()
        server.login(sender_email, password)
        text = email.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()

def compose_email():
    new_window = Toplevel()
    new_window.title("Reportar Error")
    new_window.geometry("400x300")

    Label(new_window, text="Asunto:").pack()
    subject_entry = Entry(new_window, width=50)
    subject_entry.pack()

    Label(new_window, text="Describa el Error:").pack()
    message_text = Text(new_window, height=10, width=50)
    message_text.pack()

    def on_send():
        subject = subject_entry.get()
        message_body = message_text.get("1.0", "end-1c")
        send_email(subject, message_body)
        new_window.destroy()

    send_button = Button(new_window, text="Enviar", command=on_send)
    send_button.pack()


# Setup serial connection at the beginning
setup_serial()
root = Tk()
root.title("Camera Capture")

# Create a top title label
top_title_label = Label(root, text="Último Defecto Detectado", font=("Helvetica", 20))
top_title_label.pack(side="top", pady=10)  # pady adds some padding above and below the label


latest_images_frame = Frame(root)
latest_images_frame.pack(side=TOP, pady=(5, 0))
update_frequency = 10
reference_images = []  # To store reference images captured
capture_counts = [0] * 4
Captura = False
latest_defect_images = [None] * 4  # Assuming 4 cameras
cameras = []
latest_image_labels = []  # For displaying latest saved images
image_labels = []

captura_button = Button(root, text="Pausar", command=toggle_captura)

ultimo_defecto_label = Label(root, text="Cámaras en Tiempo Real", font=("Helvetica", 16))
ultimo_defecto_label.pack(side=TOP, pady=(5, 0))

for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))
    img = ImageTk.PhotoImage(img)
    label = Label(root, image=img)
    label.pack(side=LEFT)
    image_labels.append(label)

# Add label for "Último defecto detectado"


# Prepare labels for latest saved images
for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))  # Smaller size for latest images
    img = ImageTk.PhotoImage(img)
    label = Label(latest_images_frame, image=img)
    label.pack(side=LEFT, padx=(5, 0))
    latest_image_labels.append(label)

compose_email_button = Button(root, text="Reportar Error", command=compose_email)
compose_email_button.pack(pady=20)



start_button = Button(root, text="Iniciar", command=start_capture)
start_button.pack(side=TOP)

stop_button = Button(root, text="Cerrar", command=stop_capture)
stop_button.pack(side=TOP)

open_folder_button = Button(root, text="Abrir Carpeta con Defectos", command=open_defects_folder)
open_folder_button.pack(side=TOP)

root.mainloop()
