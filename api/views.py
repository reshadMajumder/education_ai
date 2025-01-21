from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector
import base64
import json

# Configure Gemini
genai.configure(api_key="AIzaSyDTbZeiy9xL_zgGktKmef0Vcw5JhzjtbO4")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.8, minTrackCon=0.5)

# Initialize global canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True)  # Set draw=True to show hand landmarks
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList, img
    return None, None, img

def draw(info, canvas):
    fingers, lmList, img = info
    if fingers == [0, 1, 0, 0, 0]:  
        current_pos = tuple(lmList[8][0:2])  
        cv2.circle(canvas, current_pos, 10, (255, 0, 255), -1)  
    return fingers, lmList, img

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return None

@api_view(['GET'])
def index(request):
    """Render the main page"""
    return render(request, 'api/gestures_math.html')

@api_view(['POST'])
def process_frame(request):
    """Process video frames and handle gesture recognition"""
    global canvas
    try:
        data = request.data
        # Decode frame image
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        response_data = {
            'success': True,
            'drawing_mode': False,
            'position': None,
            'ai_response': None,
            'processed_frame': None,
            'last_position': data.get('last_position')  # Get last position from frontend
        }
        
        # Get hand information
        info = getHandInfo(img)
        if info:
            fingers, lmList, processed_img = info
            
            # Drawing mode (index finger up)
            if fingers == [0, 1, 0, 0, 0]:
                response_data['drawing_mode'] = True
                current_pos = tuple(lmList[8][0:2])
                
                # Draw line if we have a last position
                if response_data['last_position']:
                    last_x, last_y = response_data['last_position']
                    cv2.line(canvas, (int(last_x), int(last_y)), current_pos, (255, 0, 255), 4)
                else:
                    # If no last position, just draw a point
                    cv2.circle(canvas, current_pos, 2, (255, 0, 255), -1)
                
                response_data['position'] = current_pos
            
            # AI analysis mode (four fingers up)
            elif fingers == [1, 1, 1, 1, 0]:
                pil_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                ai_response = model.generate_content(["Solve this math problem", pil_image])
                response_data['ai_response'] = ai_response.text

            # Convert processed frame to base64
            _, buffer = cv2.imencode('.jpg', processed_img)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data['processed_frame'] = f'data:image/jpeg;base64,{frame_base64}'
            
            # Convert canvas to base64
            _, buffer = cv2.imencode('.jpg', canvas)
            canvas_base64 = base64.b64encode(buffer).decode('utf-8')
            response_data['canvas'] = f'data:image/jpeg;base64,{canvas_base64}'
        
        return Response(response_data)
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        })

@api_view(['POST'])
def clear_canvas(request):
    """Clear the canvas"""
    global canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    return Response({'success': True})

def gestures_math_view(request):
    return render(request, 'gestures_math.html')
