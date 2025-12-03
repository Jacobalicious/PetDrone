import time
import cv2
import numpy as np
from djitellopy import Tello
import os

# --- Parameters & Configuration ---

# --- Proportional Gains (P-Controller) ---
YAW_KP = 0.08
UD_KP = 0.25
FB_KP = 0.0009

# --- Target Values ---
TARGET_X = 960 / 2
TARGET_Y = 720 / 2
TARGET_AREA = 80000

# --- Safety & Speed Limits ---
YAW_SPEED_LIMIT = 50
UD_SPEED_LIMIT = 60
FB_SPEED_LIMIT = 40

# --- Face Detection ---
HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade classifier.")
    print(f"Searched at: {HAAR_CASCADE_PATH}")
    print("Please ensure OpenCV is installed correctly.")
    exit()

# --- Main Function ---
def main():
    # --- Connect to Tello ---
    tello = Tello()
    tello.connect()

    # Check battery
    battery = tello.get_battery()
    print(f"--- Tello Battery: {battery}% ---")
    if battery < 5:
        print("!!! BATTERY LOW. PLEASE CHARGE. ABORTING. !!!")
        return

    # --- Initialize Video and State ---
    tello.streamon()
    
    # --- 
    # --- CHANGED: Use tello.get_frame_read() ---
    # --- This runs in a background thread to always get the LATEST frame ---
    # --- and prevents the video buffer from growing.
    # ---
    frame_read = tello.get_frame_read()
    time.sleep(1) # Give the background thread time to start and get a frame
    
    # --- DELETED: These lines caused the massive buffer delay ---
    # stream_url = tello.get_udp_video_address()
    # print(f"--- Connecting to video stream at: {stream_url} ---")
    # cap = cv2.VideoCapture(stream_url)
    # if not cap.isOpened(): ... (deleted error check)
    
    print("--- Video stream connected successfully! ---")

    drone_in_air = False
    
    # --- CHANGED: Initialize battery_status and loop_counter for battery check ---
    battery_status = battery 
    loop_counter = 0
    
    # --- Main Control Loop ---
    try:
        while True:
            # 1. Read Frame
            # --- CHANGED: Read from the background frame reader ---
            frame = frame_read.frame
            
            # --- CHANGED: Simpler check for a valid frame ---
            if frame is None:
                print("No frame received... retrying.")
                time.sleep(0.1) # Wait a moment
                continue
            
            # --- DELETED: No 'ret' variable from cap.read() ---
            # ret, frame = cap.read() # OLD
            # if not ret or frame is None: # OLD check

            # Resize for faster processing and flip
            img = cv2.resize(frame, (960, 720))
            img = cv2.flip(img, 1)  # Flip horizontally for a "mirror" view
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. Detect Faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Initialize control speeds to 0
            yaw_speed = 0
            up_down_speed = 0
            fwd_bwd_speed = 0

            # 3. If a face is found...
            if len(faces) > 0:
                # Find the largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]

                # Draw rectangle around the face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # --- Calculate Center and Area ---
                cx = x + w // 2
                cy = y + h // 2
                area = w * h

                # Draw center point
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                # --- 4. Calculate Errors ---
                error_yaw = cx - TARGET_X
                error_ud = TARGET_Y - cy
                error_fb = TARGET_AREA - area

                # --- 5. Calculate Control Signals (P-Controller) ---
                yaw_speed = YAW_KP * error_yaw
                yaw_speed = int(np.clip(yaw_speed, -YAW_SPEED_LIMIT, YAW_SPEED_LIMIT))

                up_down_speed = UD_KP * error_ud
                up_down_speed = int(np.clip(up_down_speed, -UD_SPEED_LIMIT, UD_SPEED_LIMIT))

                fwd_bwd_speed = FB_KP * error_fb
                fwd_bwd_speed = int(np.clip(fwd_bwd_speed, -FB_SPEED_LIMIT, FB_SPEED_LIMIT))

            # 6. Send Commands (if in air)
            if drone_in_air:
                tello.send_rc_control(0, fwd_bwd_speed, up_down_speed, yaw_speed)

            # 7. Display Video Feed
            # Draw target lines
            cv2.line(img, (int(TARGET_X), 0), (int(TARGET_X), 720), (255, 0, 0), 2)
            cv2.line(img, (0, int(TARGET_Y)), (960, int(TARGET_Y)), (255, 0, 0), 2)
            
            # --- 
            # --- CHANGED: Optimized battery check ---
            # --- Only ask the drone for its battery every 100 frames.
            # --- This stops the main loop from slowing down.
            # ---
            loop_counter += 1
            if loop_counter % 100 == 0:
                try:
                    battery_status = tello.get_battery()
                except:
                    battery_status = -1 # Indicate error
            
            # This line now uses the 'battery_status' variable, which is
            # only updated every 100 loops.
            cv2.putText(img, f"Battery: {battery_status}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # --- DELETED: Old, slow battery check ---
            # try:
            #     battery_status = tello.get_battery() ...

            if not drone_in_air:
                cv2.putText(img, "Press 'T' to Take Off", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Tello Face Tracker", img)

            # 8. Handle Keyboard Input
            key = cv2.waitKey(5) & 0xFF

            if key == ord('t') and not drone_in_air:
                print("Taking off...")
                tello.takeoff()
                tello.move_up(30)
                drone_in_air = True
            
            elif key == ord('l'):
                print("Landing...")
                if drone_in_air:
                    tello.send_rc_control(0, 0, 0, 0)
                    tello.land()
                    drone_in_air = False
                
            elif key == ord('q'):
                print("Quitting...")
                break

    except KeyboardInterrupt:
        print("Script interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # --- Cleanup ---
    finally:
        print("Cleaning up...")
        if drone_in_air:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        
        # --- DELETED: 'cap' no longer exists ---
        # cap.release() # NEW
        tello.streamoff()
        cv2.destroyAllWindows()
        tello.end()
        print("Script finished.")

# --- Run the main function ---
if __name__ == "__main__":
    main()