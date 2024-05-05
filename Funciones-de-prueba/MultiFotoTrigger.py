import pypylon.pylon as py
import cv2
import serial
import time
import serial.tools.list_ports
def Camara_Luces ():
    available_ports = serial.tools.list_ports.comports()

    # Loop through the available ports and print their details
    for port in available_ports:
        print(port.device, "-", port.description)

    arduino_ports = [
        port.device
        for port in available_ports
        if 'ttyACM0' in port.description  # Adjust the keyword to match your Arduino's description
    ]

    if arduino_ports:
        arduino_port = arduino_ports[0]  # Automatically select the first detected Arduino port
        print("Arduino found on port:", arduino_port)

        # Open the serial connection
        ser = serial.Serial(arduino_port, 9600)  # Use the correct baud rate
        # Now you have a serial connection established with the detected Arduino port
        # You can use 'ser' for communication with the Arduino
    else:
        print("No Arduino found")
    time.sleep(2)

    NUM_CAMERAS = 4  # Number of cameras you want to use simultaneously

    tl_factory = py.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    for device in devices:
        print(device.GetFriendlyName())

    if len(devices) >= NUM_CAMERAS:  # Ensure at least one camera is available
        cameras = []
        
        ser.write(b'S')  # Sending 'S' as a signal to Arduino
        for i in range(NUM_CAMERAS):
            camera = py.InstantCamera(tl_factory.CreateDevice(devices[i]))
            camera.Open()
            camera.UserSetSelector = "Default"
            camera.UserSetLoad.Execute()

            camera.LineSelector = "Line1"
            camera.LineMode = "Input"
            camera.TriggerSelector = "FrameStart"
            camera.TriggerSource = "Line1"
            camera.TriggerMode = "On"
            camera.TriggerActivation.Value
            camera.ExposureMode.SetValue('TriggerWidth')
            camera.ExposureTime.SetValue(5000)
            camera.StartGrabbing()
            cameras.append(camera)  # Append each camera to the list

            num_photo = 0
            while num_photo < 10:
                grabbed_images = []
                for camera in cameras:
                    grab_result = camera.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
                    if grab_result.GrabSucceeded():
                        image_data = grab_result.Array
                        converted_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_BG2BGR)  # Replace with appropriate color conversion
                        grabbed_images.append(converted_image)

                # Save grabbed images
                for i, image in enumerate(grabbed_images):
                    filename = f"Camera_{i + 1}_Image_{num_photo}.png"
                    cv2.imwrite(filename, image)
                    print(f"Saved: {filename}")

                num_photo += 1
                print("Va por la foto número: ", num_photo)
                if num_photo == 10: 
                    ser.write(b'B')  # Sending 'B' as a signal to Arduino
                    ser.close()  # Close the serial connection
                
            # After image capturing loop
            
            for camera in cameras:
                camera.StopGrabbing()
                print("Se detienen las cámaras" )
                camera.Close()
    else:
        print("Número insuficiene de cámaras.")


Camara_Luces()
print("Funca")