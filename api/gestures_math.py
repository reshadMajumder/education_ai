import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyDTbZeiy9xL_zgGktKmef0Vcw5JhzjtbO4")
model = genai.GenerativeModel('gemini-1.5-flash')

cap = cv2.VideoCapture(0)  
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, canvas):
    fingers, lmList = info
    if fingers == [0, 1, 0, 0, 0]:  
        current_pos = tuple(lmList[8][0:2])  
        cv2.circle(canvas, current_pos, 10, (255, 0, 255), -1)  
    return canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return None

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
output_text = ""

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture video frame.")
        break

    img = cv2.flip(img, 1)  

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        canvas = draw(info, canvas)
        result = sendToAI(model, canvas, fingers)
        if result:
            output_text = result

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    cv2.imshow("Live Feed", img)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Combined Output", image_combined)

    if output_text:
        print("AI Output:", output_text)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): 
        canvas = np.zeros_like(img)
    elif key == ord('s'):  
        canvas = np.zeros_like(img)
    elif key == ord('q'):  
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
