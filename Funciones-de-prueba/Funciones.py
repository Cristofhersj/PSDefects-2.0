#Librerías importantes
import time
import serial
import serial.tools.list_ports
from pypylon import pylon
import cv2
import numpy as np
from PIL import Image  # Pillow library for image handling
import matplotlib.pyplot as plt


#Variables Globales
carpeta_proyecto = ".ESS_Vinyl_Inspector"
usuario = "vioro"
arduinoPort = "/dev/ttyACM0"
triggerMode = "ON"
lineSelector = "Line1"
exposureTime = 5000
userSet = "Default"
program_start = True
sum_HSV_patron = np.zeros(3)
average_HSV_patron = np.zeros(3)
contador_frames_HSV = 0
cant_frames_patronHSV = 1

#Fucnión para localizar y preparar el arduino para su utilizacción
def find_arduino_port():
    available_ports = serial.tools.list_ports.comports()

    # Realiza un ciclo entre los distintos dispositivos conectados para detectar y utilizar el arduino
    for port in available_ports:
        print(port.device, "-", port.description)

    arduino_ports = [
        port.device
        for port in available_ports
        if 'ttyACM0' in port.description  # Descripción dada por el tipo de arduino, actualice según el tipo de arduino utilizado
    ]

    if arduino_ports:
        arduino_port = arduino_ports[0]  # Selecciona de forma auomática el primer arduino detectado
        print("Arduino found on port:", arduino_port)
        return arduino_port
    else:
        print("No Arduino found")
        return None

def ess_alarm(msj):
    arduino_port = find_arduino_port()

    if arduino_port:
        # Open the serial connection
        with serial.Serial(arduino_port, 9600) as ser:  # Use the correct baud rate
            buffer = f"{msj}\n"
            print("Se  va a enviar")
            ser.write(buffer.encode())
            print("Se envió")

#Función Encargada de leer el documento txt que contiene el estado actual de la máquina de estados.
def revision_maquina_estados(valor_estado):
    file_estado = "/home/" + usuario + "/" + carpeta_proyecto + "/Data/Estado.txt"

    with open(file_estado, 'r') as archivo_texto:
        letra = archivo_texto.read(1)
        if letra == 'C':
            valor_estado = -1
            print("El valor del estado es: ", valor_estado)
        elif letra == 'I':
            valor_estado = -2
            print("El valor del estado es: ", valor_estado)
        elif letra == 'P':
            ess_alarm('P')
            time.sleep(3)
            print("Se envió la señal de la letra: ",letra)
            
            while letra != 'R':
                if letra == 'C':
                    valor_estado = -1
                    break
                archivo_texto.close()
                time.sleep(0.5)
                archivo_texto = open(file_estado, 'r')
                letra = archivo_texto.read(1)
            ess_alarm('R')
            time.sleep(3)
            
def write_to_estado(input_text):
    file_name = f"/home/{usuario}/{carpeta_proyecto}/Data/BarraEstado.txt"  #Revisar por qué barra estado
    with open(file_name, 'w') as archivo_texto:
        archivo_texto.write(input_text)


def capture_and_process_images(camera):
    try:
        NUM_CAMERAS = 1  # Number of cameras you want to use simultaneously

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()

        for device in devices:
            print(device.GetFriendlyName())

        if len(devices) >= NUM_CAMERAS:  # Ensure at least one camera is available
            cameras = []
            
            for i in range(NUM_CAMERAS):
                camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[i]))
                camera.Open()
                camera.UserSetSelector = "Default"
                camera.UserSetLoad.Execute()
                camera.LineSelector = "Line1"
                camera.LineMode = "Input"
                camera.TriggerSelector = "FrameStart"
                camera.TriggerSource = "Line1"
                camera.TriggerMode = "On"
                camera.TriggerActivation.Value
                camera.Width.Value = 1280
                camera.Height.Value = 480
                camera.OffsetX.Value = 360 # Adjust this value as needed
                camera.OffsetY.Value = 360 # Adjust this value as needed
                camera.ExposureMode.SetValue('TriggerWidth')
                camera.ExposureTime.SetValue(5000)
                camera.StartGrabbing()
                cameras.append(camera)  # Append each camera to the list

                num_photo = 0
                while num_photo < 10:
                    grabbed_images = []
                    for camera in cameras:
                        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if grab_result.GrabSucceeded():
                            image_data = grab_result.Array
                            converted_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_BG2BGR)  # Replace with appropriate color conversion
                            grabbed_images.append(converted_image)
                    # Convert to BGR format (assuming img_base_patron is in RGB)
                    img_base2_patron = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("img_base2_patron.jpg", img_base2_patron)
                    
                    # Convert to BGR format (assuming img_base_patron is in RGB)
                    img_base_patron = cv2.cvtColor(img_base2_patron, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("img_base_patron.jpg", img_base_patron)

                    # Convert to grayscale
                    img_gris_patron = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite("img_gris_patron.jpg", img_gris_patron)
                    
                    # Apply Gaussian blur
                    filter_size = 3  # Adjust this value as needed
                    img_gauss_patron = cv2.GaussianBlur(img_gris_patron, (filter_size, filter_size), 0)
                    cv2.imwrite("img_gauss_patron.jpg", img_gris_patron)
                    
                    img_mirror_patron = cv2.flip(img_base2_patron, 1)
                    cv2.imwrite("img_mirror_patron.jpg", img_mirror_patron)
                    
                    # Save grabbed images
                    for i, image in enumerate(grabbed_images):
                        filename = f"Camera_{i + 1}_Image_{num_photo}.png"
                        cv2.imwrite(filename, image)
                        print(f"Saved: {filename}")

                    num_photo += 1
                    
                    
                # After image capturing loop
                
                for camera in cameras:
                    camera.StopGrabbing()
                    print("Se detienen las cámaras" )
                    camera.Close()
        else:
            print("Número insuficiene de cámaras.")

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        # Stop grabbing and close the camera
        camera.StopGrabbing()
        camera.Close()

def create_instant_camera():
    # Create a pylon.TlFactory instance
    tl_factory = pylon.TlFactory.GetInstance()

    # Get all available devices
    devices = tl_factory.EnumerateDevices()

    if not devices:
        print("No devices found.")
        return None

    # Create an InstantCamera object
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

    return camera


# Example usage:
# Assuming 'camera' is an InstantCamera object
#cam = create_instant_camera()
#capture_and_process_images(cam)


from PIL import Image
from skimage import io
from skimage import feature
"""img_arr = np.array(Image.open('/home/vioro/tr/img_gauss_patron.jpg'), dtype=np.uint8)
result = cv2.convertScaleAbs(img_arr, alpha=2, beta=0)
print(img_arr.shape)
edges1 = feature.canny(result, sigma=1)
io.imsave('Edges.png', edges1)
io.imsave('contrast.png', result)"""


def draw_rectangles_on_contours(image_path):
    # Read the image
    original_image = cv2.imread(image_path)
    alpha = 1.5  # Contrast control (1.0 means no change)
    beta = 0    # Brightness control (0 means no change)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 0, 130)  # You can adjust the parameters as needed
    
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    

    # Draw rectangles around the contours on the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save or display the result
    cv2.imwrite('Defectos.png', original_image)
    cv2.imshow("Contraste",adjusted_image)
    cv2.imshow("Defects",edges)
    cv2.imshow("grey",gray_image)
    cv2.imshow("Image with Rectangles", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
image_path = '/home/vioro/tr/resized.png'
#draw_rectangles_on_contours(image_path)
img = cv2.imread(image_path)

# Calculate histogram
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Normalize the histogram
hist = hist / hist.sum()

# Create an image for displaying the histogram
hist_image = np.zeros((256, 256), dtype=np.uint8)

# Draw the histogram
for i, prob in enumerate(hist):
    cv2.line(hist_image, (i, 255), (i, int(255 - prob * 2000)), 255, 1)

# Display the histogram
cv2.imshow('Histogram', hist_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read the image

# Ensure the image is not None
if img is not None:
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract the Hue channel
    hue_channel = hsv_img[:, :, 0]

    # Set a threshold value for the Hue channel (adjust this according to your needs)
    threshold_value = 90

    # Binarize the Hue channel
    _, binary_image = cv2.threshold(hue_channel, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the original and binary images
    cv2.imshow('Original Image', img)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to read the image.")"""
    
def HSV_Mean_calc(source):
            # Convert BGR to HSV
            target = cv2.cvtColor(source, cv2.COLOR_BGR2HSV_FULL)

            # Split HSV channels
            HSV_H, HSV_S, HSV_V = cv2.split(target)

            # Estimate average values of the channels
            Hue_Mean = np.mean(HSV_H)
            Saturation_Mean = np.mean(HSV_S)
            Value_Mean = np.mean(HSV_V)

            return Hue_Mean, Saturation_Mean, Value_Mean
        
img_base_actual=cv2.imread('/home/vioro/tr/img_base_patron.jpg')
img_base_patron=cv2.imread('/home/vioro/tr/img_gauss_patron.jpg')
# Obtiene el valor promedio de las componentes HSV de n imagenes
if contador_frames_HSV < cant_frames_patronHSV:
    Hue_mean, Saturation_mean, Value_mean = HSV_Mean_calc(img_base_actual)

    # values_HSV[contador_frames_HSV][0] = Hue_mean
    # values_HSV[contador_frames_HSV][1] = Saturation_mean
    # values_HSV[contador_frames_HSV][2] = Value_mean
    sum_HSV_patron[0] += Hue_mean
    sum_HSV_patron[1] += Saturation_mean
    sum_HSV_patron[2] += Value_mean

    contador_frames_HSV += 1

    # Saca la media de las n imagenes (patron de color)
    if contador_frames_HSV == cant_frames_patronHSV:
        average_HSV_patron[0] = sum_HSV_patron[0] / cant_frames_patronHSV
        average_HSV_patron[1] = sum_HSV_patron[1] / cant_frames_patronHSV
        average_HSV_patron[2] = sum_HSV_patron[2] / cant_frames_patronHSV
        
print("Average HSV values:")
print("Hue: {:.2f}".format(average_HSV_patron[0]))
print("Saturation: {:.2f}".format(average_HSV_patron[1]))
print("Value: {:.2f}".format(average_HSV_patron[2]))

def check_color_defect(pattern_image, current_image, error_hsv_acceptable):
    hue_mean_pattern, saturation_mean_pattern, value_mean_pattern = HSV_Mean_calc(pattern_image)
    hue_mean_current, saturation_mean_current, value_mean_current = HSV_Mean_calc(current_image)

    # Check if the denominator is not zero before division
    hue_error = abs((hue_mean_pattern - hue_mean_current) / (hue_mean_pattern + 1e-10)) * 100
    saturation_error = abs((saturation_mean_pattern - saturation_mean_current) / (saturation_mean_pattern + 1e-10)) * 100
    value_error = abs((value_mean_pattern - value_mean_current) / (value_mean_pattern + 1e-10)) * 100

    # Alert for error in hue
    if hue_error >= error_hsv_acceptable or saturation_error >= error_hsv_acceptable:
        print(f"Flag de error por tono en la cámara")
        # Additional action on error if needed

# Example usage:
# Assuming img_base_patron is the pattern image, img_base_actual is the current image, and errorHSV_aceptable is the threshold
#check_color_defect(img_base_patron, img_base_patron, 10)

def find_first_interface_with_cameras():
    # Get all available devices (cameras)
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()

    for device in devices:
        try:
            # Create a camera for the current device
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
            
            # Check if the camera is connected
            if camera.IsPylonDeviceAttached():
                # Return the camera and device if connected
                return camera, device
        except Exception as e:
            print(f"Error while handling device: {str(e)}")

    # If no connected cameras are found, return None
    return None, None
"""
# Example usage
camera, device = find_first_interface_with_cameras()
if camera is not None:
    print(f"Found connected camera on interface: {device.GetFriendlyName()}")
    # Now you can use 'camera' for further operations
else:
    print("No connected cameras found.")"""
    
    
    
#Esta función se encarga de calcular la diferencia entre la imagen de base 
# y los posibles defectos encontrados.
def calculate_difference(img1, img2):
    diff = cv2.absdiff(img1, img2)
    _, diff = cv2.threshold(diff, 11, 255, cv2.THRESH_BINARY)
    return diff
import numpy as np

#Esta función disminuye la cantidad de ruido en la imagen en caso de haber una 
#iluminación no ideal
def denoise_binary_image(binary_image, kernel_size=3):
    # Apply erosion to remove small white regions and protrusions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    # Apply dilation to add small white regions and broaden protrusions
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    return dilated_image

#Esta función se encarga de dibujar un rectángulo en la posición en que se encuantran defectos
def draw_rectangle(img1, img2, ksize=3):
    diff = calculate_difference(img1, img2)
    closed = denoise_binary_image(diff)
    dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return img1

"""
img1 = cv2.imread('Cuadro2.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Cuadro1.png', cv2.IMREAD_GRAYSCALE)
result = draw_rectangle(img1, img2)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()"""


import cv2
import numpy as np
import time

def detect_defects(img_path, filter_size=3, Lim_Inf=100, Lim_Sup=255, tam_max_defecto=50):
    # Load the input image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the grayscale image
    img_gauss = cv2.GaussianBlur(img, (filter_size, filter_size), 0)

    # Apply Canny edge detection to the grayscale image
    img_bordes = cv2.Canny(img_gauss, Lim_Inf, Lim_Sup, 3, False)

    # Define the OpenCV element for dilation and erosion
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Dilate and erode the edge map to enhance the contours
    img_dilatacion = cv2.dilate(img_bordes, element, cv2.POINT(-1, -1), 4, 1, 1)
    img_erosion = cv2.erode(img_dilatacion, element, cv2.POINT(-1, -1), 2, 1, 1)

    # Find contours in the dilated/eroded edge map
    contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the rectangles around the defects
    rectangles = []

    # Loop through the detected contours and draw rectangles around them
    for c in contours:
        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(c)

        # Ignore small contours
        if w < tam_max_defecto and h < tam_max_defecto:
            continue

        # Draw a rectangle around the contour
        rectangles.append(cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2))

    # Return the input image with the rectangles around the defects
    return img, rectangles

#Genera el patrón HSV principalmente por el promedio de la imagen
def generate_hsv_pattern(image, n_frames, hsv_sum):
    """
    Generates an HSV pattern based on the average HSV values of n_frames frames.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_values = np.mean(hsv_image, axis=(0, 1))
    hsv_sum += hsv_values

    return hsv_sum

#Detecta el defecto si cambia de color
def check_color_defect(image, hsv_mean, error_hsv_acceptable):
    """
    Checks for color defects based on the difference between the average HSV values of the image and the HSV pattern.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_values = np.mean(hsv_image, axis=(0, 1))
    error_hsv = [abs((hsv_mean[i] - hsv_values[i]) / hsv_mean[i]) * 100 for i in range(3)]

    if any(e >= error_hsv_acceptable for e in error_hsv):
        print("Color defect detected!")


def process_frame(image, n_frames, hsv_sum, hsv_mean, error_hsv, error_hsv_acceptable, id):
    """
    Processes a frame and generates an HSV pattern if no defects are found.
    """
    hsv_sum = generate_hsv_pattern(image, n_frames, hsv_sum)
    check_color_defect(image, hsv_mean, error_hsv_acceptable)

    # Clear the error_hsv list every 60 frames
    if len(error_hsv) >= 60:
        error_hsv.clear()

    # Print the status of the frame
    if len(error_hsv) % 60 == 0:
        if not error_hsv:
            print(f"Frame {len(error_hsv)} in camera {id} - OK")
        else:
            print(f"Frame {len(error_hsv)} in camera {id} - Error detected")
            
def main():
    # Call Elige_Usuario and reseteaStringErrores functions here
    Elige_Usuario()
    reseteaStringErrores()

    # Get the current date and create a directory for storing image data
    date = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(Images_result, date)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)

    # Create CSV files for storing calibration and processing time data
    archivo_dia_calibracion = os.path.join(dir_ProcesoCalibracion, date + ".csv")
    archivo_dia_velocidad = os.path.join(dir_MetricaVelocidad, date + ".csv")
    Crea_ProcesoCalibracion(archivo_dia_calibracion, max_lim_inf, tam_kernel_Init, max_iteraciones)
    Crea_TiemposProcesamiento(archivo_dia_velocidad)

    # Initialize system and interface objects
    system = System.GetInstance()
    interfaceList = system.GetInterfaces()
    interfacePtr = None
    numInterfaces = interfaceList.GetSize()

    # Find interfaces that contain cameras
    for N_Int in range(numInterfaces):
        interfacePtr = interfaceList.GetByIndex(N_Int)
        interfacePtr.UpdateCameras()
        camList = interfacePtr.GetCameras()
        numCameras = camList.GetSize()
        if numCameras != 0:
            break

    # Open serial connection and turn on lights
    writeToEstado("Z")
    serial_stream = ess.open_serial()
    time.sleep(3)
    ess.alarm(serial_stream, 'R', 0)
    time.sleep(3)
    ess.alarm(serial_stream, 'L', 0)

    # Prepare for image transmission
    purgar()
    Revision_MaquinaEstados(Valor_Estado, serial_stream)

    # Start image transmission
    writeToEstado("W")
    result = ess.find_cameras(interfacePtr, camList, numCameras)
    if result < 0:
        return result

    # Initialize camera objects and threads
    List_pCams = [None]*numCameras
    NumCam = [0]*numCameras
    numCameras = min(4, numCameras)  # Limit number of cameras to 4
    camThreads = [None]*numCameras

    for id in range(numCameras):
        List_pCams[id] = camList.GetByIndex(id)
        nodeMap = List_pCams[id].GetNodeMap()
        ResetTrigger(nodeMap)
        setupTrigger(nodeMap)
        camThreads[id] = threading.Thread(target=Analizar_Imagenes, args=(id, List_pCams[id], NumCam[id]))
        camThreads[id].start()

    # Wait for camera threads to finish
    for id in range(numCameras):
        camThreads[id].join()

    # Clean up
    for id in range(numCameras):
        List_pCams[id] = None

    ess.alarm(serial_stream, 'P', 0)
    time.sleep(3)
    sp.close(port)
    ess.stop_transmission(numCameras, interfacePtr, camList, interfaceList, system)

    return 0