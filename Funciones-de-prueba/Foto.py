#!/home/vioro/tr/.venv/bin/python3
# coding=utf-8

# The rest of your Python code follows

import os
from pypylon import pylon
import cv2
import serial.tools.list_ports
import time
import numpy as np
from tkinter import *
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import tkinter as tk

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

def update_images():
                for i, camera in enumerate(cameras):
                    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grabResult.GrabSucceeded():
                        img = grabResult.GetArray()
                        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                        img = cv2.resize(img, (320, 320))  # Resize the image to 480x480
                        img = Image.fromarray(img).convert("RGB")  # Convert to RGB
                        img = ImageTk.PhotoImage(img)
                        image_labels[i].configure(image=img)
                        image_labels[i].image = img
                    else:
                        print(f"Error: Camera {i+1} failed to grab.")
                    grabResult.Release()
                root.after(1, update_images)
                
                
def Referencia(save_path='./captured_images', exposure_time=5000, num_images_per_camera=1):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    num_cameras = len(devices)
    cameras = pylon.InstantCameraArray(num_cameras)

    for i, cam in enumerate(cameras):
        cam.Attach(tl_factory.CreateDevice(devices[i]))
        cam.Open()
        cam.ExposureTime.SetValue(exposure_time)
        setup_camera(cam)  # Set up hardware trigger
        print(f"Camera {i+1}: Exposure time set to {exposure_time} and configured for hardware trigger on Line 1")
    arduino_port = find_arduino_port()
    if arduino_port:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
    ser.write(b'F')
    print("Command 'F' sent. Cycle started.")
    cameras.StartGrabbing(pylon.GrabStrategy_OneByOne)

    images_captured = [None] * num_cameras  # Initialize a list to store images per camera

    try:
        while cameras.IsGrabbing() and not all(x is not None for x in images_captured):
            grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            camera_index = grabResult.GetCameraContext()  # Identifies which camera provided the image
            if grabResult.GrabSucceeded():
                # Convert grabbed image to RGB
                img = grabResult.GetArray()
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
                images_captured[camera_index] = img
            else:
                print(f"Error: Camera {camera_index+1} failed to grab.")
            grabResult.Release()

    finally:
        # Stop grabbing and close all cameras
        cameras.StopGrabbing()
        for cam in cameras:
            cam.Close()
        ser.write(b'P')
        print("Command 'F' sent. Cycle Ended.")

    # Save images as PNG
    for i, img in enumerate(images_captured):
        if img is not None:
            filename = os.path.join(save_path, f"camera_{i+1}_image.png")
            cv2.imwrite(filename, img)
            print(f"Saved: {filename}")                
                
                

def start_capture():
    global cameras, root, Captura
    Referencia()
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

    arduino_port = find_arduino_port()
    if arduino_port:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
    ser.write(b'F')
    print("Command 'F' sent. Cycle started.")
    start_button.config(state=DISABLED)
    start_button.destroy()
    toggle_captura()  # Start capturing images automatically

def stop_capture():
    global cameras, Captura
    Captura = False
    cameras = []  # Reset the cameras list
    for label in image_labels:
        label.configure(image="")
    arduino_port = find_arduino_port()
    if arduino_port:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
    ser.write(b'P')
    print("Command 'P' sent. Cycle ended.")
    root.destroy()

def setup_camera(camera):
    camera.ExposureTime.SetValue(5000)
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Line1")
    
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

def process_contours(img1, img2, ksize=3):
    diff = calculate_difference(img1, img2)
    closed = denoise_binary_image(diff)
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tam_max_defecto = 5

    # Process contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore small defects
        if w < tam_max_defecto and h < tam_max_defecto:
            cant_contours_small += 1
        else:
            # Draw square for non-small defects
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 1)
    
def toggle_captura():
    global Captura
    Captura = not Captura
    if Captura:
        for camera in cameras:
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        update_images()
    else:
        for camera in cameras:
            camera.StopGrabbing()
root = Tk()
root.title("Camera Capture")

Captura = False
cameras = []
image_labels = []

for i in range(4):
    img = Image.new('RGB', (320, 320), color=(0, 0, 0))
    img = ImageTk.PhotoImage(img)
    label = Label(root, image=img)
    label.pack(side=LEFT)
    image_labels.append(label)


captura_button = Button(root, text="Pausar", command=toggle_captura)
captura_button.pack(side=LEFT)

start_button = Button(root, text="Start Capture", command=start_capture)
start_button.pack(side=TOP)

stop_button = Button(root, text="Stop Capture", command=stop_capture)
stop_button.pack(side=TOP)

root.mainloop()