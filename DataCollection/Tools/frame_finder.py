import cv2

video = '2024-05-23-20-36-20.mkv'

video_capture = cv2.VideoCapture(video)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
while(True):
        ret, frame = video_capture.read()
        if not ret:
            break

        count += 1

print (video_length, count)


video_capture.release()
cv2.destroyAllWindows()

video_capture = cv2.VideoCapture(video)

video_length = count

count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    count += 1
    
    if count % 10 == 0 or count == video_length:
        cv2.putText(frame, f"Frame: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        # Wait for the ‘q’ key to be pressed to quit
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()