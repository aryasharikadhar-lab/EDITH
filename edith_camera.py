import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera unavailable. Check camera permissions.")
else:
    print("EDITH vision is online!")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        cv2.imshow("EDITH Vision", frame)

        # Press q to close the camera window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

