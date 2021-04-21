import cv2
import numpy as np
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

CNN_INPUT_SIZE = 128


def stable_point2d(point_2d):
    diff = point_2d[0][0] - point_2d[1][0]
    if 10 > diff > -10:
        return True
    else:
        return False


def direction_func(point_2d):

    direction = "Normal"

    if point_2d[0][0] <= point_2d[5][0] or point_2d[1][0] <= point_2d[6][0]:
        # print("Head Right")
        direction = "Head Right"
    elif point_2d[2][0] >= point_2d[7][0] or point_2d[3][0] >= point_2d[8][0]:
        # print("Head Left")
        direction = "Head Left"
    elif point_2d[1][1] >= point_2d[6][1] - 10 or point_2d[2][1] >= point_2d[7][1] - 10:
        # print("Head Up")
        direction = "Head Up"
    elif point_2d[0][1] <= point_2d[5][1] or point_2d[3][1] <= point_2d[8][1]:
        # print("Head Down")
        direction = "Head Down"
    else:
        # print("Normal")
        pass

    return direction


def main():
    """MAIN"""

    video_src = 0
    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()
    previous_direction = "Normal"
    prev_face = "Yes"

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        # img_queue.put(frame)

        # Get face from box queue.
        # facebox = box_queue.get()
        facebox = mark_detector.extract_cnn_facebox(frame)
        if facebox is None:
            no_face = "No face detected"
            if no_face != prev_face:
                print(no_face)
                prev_face = no_face

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            prev_face = "Yes"
            face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            point_2d = pose_estimator.draw_annotation_box(frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))
            x = stable_point2d(point_2d)

            if x:
                direction = direction_func(point_2d)
                if direction != previous_direction:
                    print(direction)
                    previous_direction = direction



            # Uncomment following line to draw head axes on frame.
            # pose_estimator.draw_axes(frame, stabile_pose[0], stabile_pose[1])

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break

    # Clean up the multiprocessing process.
    # box_process.terminate()
    # box_process.join()


if __name__ == '__main__':
    main()
