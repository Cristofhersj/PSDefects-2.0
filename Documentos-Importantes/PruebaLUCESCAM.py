#!/home/vioro/tr/.venv/bin python3
# The rest of your Python code follows

import serial
import time
import serial.tools.list_ports
import pypylon.pylon as py
import cv2

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

def cam_control():
    arduino_port = find_arduino_port()
    NUM_CAMERAS = 1  # Number of cameras you want to use simultaneously

    tl_factory = py.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    for device in devices:
        print(device.GetFriendlyName())

    if len(devices) >= NUM_CAMERAS:  # Ensure at least one camera is available
        cameras = []

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
            camera.ExposureMode.SetValue('TimedExposure')  # Change to TimedExposure
            camera.ExposureTime.SetValue(5000)  # Increase exposure time to 5000 us
            camera.Gain.SetValue(0)  # Set gain to 0
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

            # After image capturing loop
            
            for camera in cameras:
                camera.StopGrabbing()
                print("Se detienen las cámaras" )
                camera.Close()
    else:
        print("Número insuficiene de cámaras.")
        
def send_signal():
    arduino_port = find_arduino_port()

    if arduino_port:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)

            while True:
                user_input = input("Enter command (F/P): ")

                if user_input.lower() == 'exit':
                    # Send 'P' to stop the cycles
                    ser.write(b'P')
                    print("Cycles stopped.")
                    break

                # Send 'F' or 'P' command to the Arduino
                ser.write(user_input.encode())
                print(f"Command '{user_input}' sent.")
                #cam_control()
               

        except Exception as e:
            print(f"Error: {e}")

        finally:
            if ser.is_open:
                ser.close()
                print("Serial connection closed.")


def Obtener_Referencia():

    arduino_port = find_arduino_port()
    NUM_CAMERAS = 1  # Number of cameras you want to use simultaneously

    tl_factory = py.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    for device in devices:
        print(device.GetFriendlyName())

    if len(devices) >= NUM_CAMERAS:  # Ensure at least one camera is available
        cameras = []
        
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
            camera.ExposureTime.SetValue(1500)
            camera.StartGrabbing()
            cameras.append(camera)  # Append each camera to the list

            num_photo = 0

    if arduino_port:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2) #Siempre poner un delay después de abrir una coneión por serial.
            Captura = True
            while Captura:
                # Send 'F' to start the cycle
                ser.write(b'F')
                print("Command 'F' sent. Cycle started.")
                time.sleep(5)  # Wait for 5 seconds                
                while num_photo < 1:
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
                    
                # After image capturing loop
                
                for camera in cameras:
                    camera.StopGrabbing()
                    print("Se detienen las cámaras" )
                    camera.Close()
                Captura= False
                
            # Send 'P' to stop the cycle
            ser.write(b'P')
            print("Command 'P' sent. Cycle stopped.")
            time.sleep(5)  # Wait for 5 seconds

        except Exception as e:
            print(f"Error: {e}")

        finally:
            if ser.is_open:
                ser.close()
                print("Serial connection closed.")

# Make sure to define or import your find_arduino_port() function correctly

def Ciclo_Trabajo():
    arduino_port = find_arduino_port()
    NUM_CAMERAS = 1  # Number of cameras you want to use simultaneously

    tl_factory = py.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()

    for device in devices:
        print(device.GetFriendlyName())

    if len(devices) >= NUM_CAMERAS:  # Ensure at least one camera is available
        cameras = []
        
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
            camera.ExposureTime.SetValue(1500)
            camera.StartGrabbing()
            cameras.append(camera)  # Append each camera to the list

            num_photo = 0

    if arduino_port:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2) #Siempre poner un delay después de abrir una coneión por serial.
            Captura = True
            while Captura:
                # Send 'F' to start the cycle
                ser.write(b'F')
                print("Command 'F' sent. Cycle started.")            
               
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

                    print("Va por la foto número: ", num_photo)
                    
                # After image capturing loop
                
                for camera in cameras:
                    camera.StopGrabbing()
                    print("Se detienen las cámaras" )
                    camera.Close()
                Captura= False
                
            # Send 'P' to stop the cycle
            ser.write(b'P')
            print("Command 'P' sent. Cycle stopped.")
            time.sleep(5)  # Wait for 5 seconds

        except Exception as e:
            print(f"Error: {e}")

        finally:
            if ser.is_open:
                ser.close()
                print("Serial connection closed.")

    




send_signal()
#Obtener_Referencia()
#Ciclo_Trabajo()