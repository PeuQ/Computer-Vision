import cv2
import requests
import time

# The URL where your FastAPI is running
API_URL = "http://localhost:8000/predict"
VIDEO_PATH = "comp_vis_general/detector_pets/playing_cat.mp4"

def test_api_with_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    print(f"Connecting to API at {API_URL}...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 1. Encode frame as JPG
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        # 2. Prepare the file for the POST request
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

        # 3. Send to API and measure time
        start = time.time()
        response = requests.post(API_URL, files=files)
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            # Draw the API's results back onto the frame
            for det in data['detections']:
                x1, y1, x2, y2 = map(int, det['coordinates'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{det['species']} ({latency:.0f}ms)", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show the "Stream" coming back from the API
        cv2.imshow("API Video Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_api_with_video()
    