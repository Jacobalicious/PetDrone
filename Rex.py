from djitellopy import Tello
import time
import numpy as np
import cv2
import mediapipe as mp

# --- Constants ---
TARGET_HEIGHT_FEET = 5.0
FEET_TO_CM = 30.48
TARGET_HEIGHT_CM = TARGET_HEIGHT_FEET * FEET_TO_CM

# --- New Hand Tracking Constants ---
FRAME_WIDTH = 960  # Tello's default frame width
FRAME_HEIGHT = 720 # Tello's default frame height
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

# --- PID Gains (These will NEED tuning!) ---
# P_GAIN_VERTICAL = 0.7 # We'll reuse this idea
P_GAIN_YAW = 0.15          # Controls how fast it turns (left/right)
P_GAIN_UP_DOWN = 0.3     # Controls how fast it moves (up/down)

# --- Control Limits ---
MAX_SPEED_YAW = 70       # Max turning speed
MAX_SPEED_UP_DOWN = 70   # Max vertical speed

# --- Hand Detection Smoothing ---
# Simple moving average for hand position
SMOOTHING_FACTOR = 0.5
last_hand_cx = CENTER_X
last_hand_cy = CENTER_Y

def main():
    print("Connecting to Tello drone...")
    tello = Tello()
    tello.connect()
    print("✅ Connection successful.")

    # --- 🔋 Battery Check ---
    battery_level = tello.get_battery()
    print(f"🔋 Battery level: {battery_level}%")
    if battery_level < 20:
        print("!!! 🔴 Battery is too low for flight! Please charge the drone. 🔴 !!!")
        return

    # --- 🎥 Initialize MediaPipe & Video ---
    print("Initializing MediaPipe Hand Detection...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,              # Look for only one hand
        min_detection_confidence=0.7, # Minimum confidence to be "a hand"
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    
    print("Turning on video stream...")
    tello.streamon()
    # Set a fixed resolution for consistent center-point calculation
    frame_read = tello.get_frame_read()
    if frame_read.frame is None:
        print("!!! 🔴 Failed to get video frame! 🔴 !!!")
        tello.streamoff()
        return
    print("✅ Video stream active.")

    try:
        print("\n--- Taking off in 3 seconds... ---")
        time.sleep(3)
        tello.takeoff()
        print("✈️ Takeoff complete. Holding position for 2s...")
        time.sleep(2) 
        
        print("\n--- 🤖 STARTING HAND TRACKING ---")
        print("Show your open hand to the camera to begin tracking.")
        print("Press 'q' in the video window to land.")

        global last_hand_cx, last_hand_cy # Use global for smoothing

        while True:
            frame = frame_read.frame
            if frame is None:
                print("...waiting for frame...")
                time.sleep(0.1)
                continue
            
            # Resize for consistent processing, flip for "mirror" view
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame_flipped = cv2.flip(frame, 1) # 1 = horizontal flip
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            
            # Process the frame to find hands
            results = hands.process(rgb_frame)

            # --- Initialize control speeds to 0 (hover) ---
            yaw_velocity = 0
            up_down_speed = 0

            if results.multi_hand_landmarks:
                # --- Hand Detected ---
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get palm center (avg of 4 key points)
                palm_points = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                ]
                
                hand_cx = np.mean([p.x for p in palm_points]) * FRAME_WIDTH
                hand_cy = np.mean([p.y for p in palm_points]) * FRAME_HEIGHT

                # Apply smoothing
                hand_cx = (hand_cx * SMOOTHING_FACTOR) + (last_hand_cx * (1 - SMOOTHING_FACTOR))
                hand_cy = (hand_cy * SMOOTHING_FACTOR) + (last_hand_cy * (1 - SMOOTHING_FACTOR))
                last_hand_cx, last_hand_cy = hand_cx, hand_cy

                # --- 🎛️ Calculate Control Error ---
                error_x = hand_cx - CENTER_X
                error_y = hand_cy - CENTER_Y # Y is inverted (0 is top)
                
                # --- P-Controller Logic ---
                # 1. Yaw (Turning) Control
                yaw_velocity = int(P_GAIN_YAW * error_x)
                yaw_velocity = int(np.clip(yaw_velocity, -MAX_SPEED_YAW, MAX_SPEED_YAW))

                # 2. Altitude (Up/Down) Control
                # We use a negative gain because a *positive* error_y
                # (hand below center) means we need to move *down* (- speed)
                up_down_speed = -int(P_GAIN_UP_DOWN * error_y)
                up_down_speed = int(np.clip(up_down_speed, -MAX_SPEED_UP_DOWN, MAX_SPEED_UP_DOWN))

                # --- Drawing / Debugging ---
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame_flipped, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Draw center of hand
                cv2.circle(frame_flipped, (int(hand_cx), int(hand_cy)), 10, (0, 0, 255), -1)
                # Print debug info
                print(f"\r HAND: ({hand_cx:.0f}, {hand_cy:.0f}) | ERR: ({error_x:.0f}, {error_y:.0f}) | CMD: (Yaw: {yaw_velocity:3d}, UD: {up_down_speed:3d})", end="")

            else:
                # --- No Hand Detected ---
                # Stop moving (hover) and reset smoothing
                print(f"\r...Searching for hand... Sending (0, 0, 0, 0)", end="")
                last_hand_cx = CENTER_X # Reset smoother
                last_hand_cy = CENTER_Y

            # --- Send Final Command to Drone ---
            # send_rc_control(left_right, forward_back, up_down, yaw)
            tello.send_rc_control(0, 0, up_down_speed, yaw_velocity)

            # --- Display the Video Feed ---
            # Draw crosshairs at center
            cv2.circle(frame_flipped, (CENTER_X, CENTER_Y), 10, (0, 255, 0), 2)
            cv2.imshow("Tello Hand Tracker", frame_flipped)

            # --- Quit Condition ---
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("\n'q' pressed. Landing...")
                break
            
    except Exception as e:
        print(f"\n!!! 🔴 An error occurred: {e} 🔴 !!!")
        print("!!! Attempting emergency land... !!!")

    finally:
        # --- 🧹 Cleanup ---
        print("\n\n--- Routine finished. Preparing to land. ---")
        cv2.destroyAllWindows() # Close the video window
        hands.close()           # Close the MediaPipe module
        tello.streamoff()       # Turn off video stream
        tello.send_rc_control(0, 0, 0, 0) # Send hover command
        tello.land()            # Land
        print("✅ Landed safely. Goodbye!")


if __name__ == '__main__':
    main()