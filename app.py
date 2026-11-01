import argparse
import math

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose


def findDistance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def findAngle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    if distance == 0:
        return 0.0
    return abs(math.degrees(math.atan2(dx, -dy)))


def sendWarning():
    print("Warning: bad posture threshold exceeded.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Posture Monitor with MediaPipe")
    parser.add_argument(
        "--video",
        default="0",
        help="Path to the input video file. Use 0 for the default webcam.",
    )
    parser.add_argument(
        "--view",
        choices=("auto", "front", "side"),
        default="auto",
        help="Posture view mode. Auto detects front or side view.",
    )
    parser.add_argument("--offset-threshold", type=int, default=100)
    parser.add_argument("--neck-angle-threshold", type=int, default=25)
    parser.add_argument("--torso-angle-threshold", type=int, default=10)
    parser.add_argument("--time-threshold", type=int, default=180)
    return parser.parse_args()


def open_video(video_path):
    if video_path is None or str(video_path).strip() == "0":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(video_path)


def draw_text(image, text, position, color, scale=0.6, thickness=2):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main(
    video_path="0",
    offset_threshold=100,
    neck_angle_threshold=25,
    torso_angle_threshold=10,
    time_threshold=180,
    view_mode="auto",
):
    good_frames = 0
    bad_frames = 0
    red = (60, 60, 235)
    green = (70, 210, 70)
    yellow = (0, 220, 255)
    white = (245, 245, 245)
    panel = (25, 25, 25)

    cap = open_video(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video source: {video_path}")

    try:
        with mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as pose:
            while True:
                success, image = cap.read()
                if not success:
                    print("Unable to read a frame; stopping.")
                    break

                height, width = image.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps > 0 else 30.0
                results = pose.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )
                landmarks = results.pose_landmarks

                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (width, 126), panel, -1)
                cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)

                if landmarks is None:
                    good_frames = 0
                    bad_frames = 0
                    draw_text(image, "NO PERSON DETECTED", (18, 35), yellow, 0.8)
                    draw_text(
                        image,
                        "Move into frame so shoulders and hips are visible",
                        (18, 70),
                        white,
                        0.55,
                        1,
                    )
                    draw_text(image, "View: waiting", (18, 105), yellow, 0.55, 1)
                else:
                    pose_landmark = mp_pose.PoseLandmark

                    def point(landmark):
                        value = landmarks.landmark[landmark]
                        return int(value.x * width), int(value.y * height)

                    def visible(landmark):
                        return landmarks.landmark[landmark].visibility >= 0.2

                    required = (
                        pose_landmark.LEFT_SHOULDER,
                        pose_landmark.RIGHT_SHOULDER,
                        pose_landmark.LEFT_HIP,
                    )
                    if not all(visible(landmark) for landmark in required):
                        good_frames = 0
                        bad_frames = 0
                        draw_text(image, "PARTIAL POSE", (18, 35), yellow, 0.8)
                        draw_text(
                            image,
                            "Keep both shoulders and hips visible",
                            (18, 70),
                            white,
                            0.6,
                            1,
                        )
                    else:
                        left_shoulder = point(pose_landmark.LEFT_SHOULDER)
                        right_shoulder = point(pose_landmark.RIGHT_SHOULDER)
                        left_hip = point(pose_landmark.LEFT_HIP)
                        right_hip = (
                            point(pose_landmark.RIGHT_HIP)
                            if visible(pose_landmark.RIGHT_HIP)
                            else left_hip
                        )
                        nose = point(pose_landmark.NOSE)
                        left_ear = point(pose_landmark.LEFT_EAR)
                        right_ear = point(pose_landmark.RIGHT_EAR)

                        shoulder_center = (
                            (left_shoulder[0] + right_shoulder[0]) // 2,
                            (left_shoulder[1] + right_shoulder[1]) // 2,
                        )
                        hip_center = (
                            (left_hip[0] + right_hip[0]) // 2,
                            (left_hip[1] + right_hip[1]) // 2,
                        )
                        shoulder_width = findDistance(
                            *left_shoulder, *right_shoulder
                        )
                        hip_width = findDistance(*left_hip, *right_hip)
                        torso_height = max(
                            findDistance(*shoulder_center, *hip_center), 1
                        )
                        shoulder_tilt = abs(
                            math.degrees(
                                math.atan2(
                                    right_shoulder[1] - left_shoulder[1],
                                    right_shoulder[0] - left_shoulder[0],
                                )
                            )
                        )
                        hip_tilt = abs(
                            math.degrees(
                                math.atan2(
                                    right_hip[1] - left_hip[1],
                                    right_hip[0] - left_hip[0],
                                )
                            )
                        )
                        torso_lean = abs(
                            math.degrees(
                                math.atan2(
                                    hip_center[0] - shoulder_center[0],
                                    torso_height,
                                )
                            )
                        )
                        front_view = (
                            view_mode == "front"
                            or (
                                view_mode == "auto"
                                and visible(pose_landmark.RIGHT_HIP)
                                and shoulder_width / max(torso_height, 1) > 0.5
                                and hip_width / max(shoulder_width, 1) > 0.45
                            )
                        )
                        current_view = (
                            front_view if view_mode == "auto" else view_mode == "front"
                        )

                        if current_view:
                            view_label = "FRONT"
                            neck_metric = abs(
                                nose[0] - shoulder_center[0]
                            ) / max(shoulder_width, 1) * 100
                            neck_ok = neck_metric < neck_angle_threshold * 2
                            torso_metric = torso_lean
                            torso_ok = (
                                torso_metric < torso_angle_threshold
                                and shoulder_tilt < torso_angle_threshold * 2
                                and hip_tilt < torso_angle_threshold * 2
                            )
                            metric_text = (
                                f"Head offset: {neck_metric:.1f}% | "
                                f"Body lean: {torso_metric:.1f} deg"
                            )
                        else:
                            view_label = "SIDE"
                            ear = left_ear if visible(pose_landmark.LEFT_EAR) else right_ear
                            neck_metric = findAngle(
                                *left_shoulder, *ear
                            )
                            torso_metric = findAngle(
                                *left_hip, *left_shoulder
                            )
                            neck_ok = neck_metric < neck_angle_threshold
                            torso_ok = torso_metric < torso_angle_threshold
                            metric_text = (
                                f"Neck: {neck_metric:.1f} deg | "
                                f"Torso: {torso_metric:.1f} deg"
                            )

                        good_posture = neck_ok and torso_ok
                        if good_posture:
                            good_frames += 1
                            bad_frames = 0
                        else:
                            good_frames = 0
                            bad_frames += 1

                        status = "GOOD POSTURE" if good_posture else "ADJUST POSTURE"
                        status_color = green if good_posture else red
                        draw_text(image, status, (18, 35), status_color, 0.8)
                        draw_text(
                            image,
                            f"View: {view_label} | {metric_text}",
                            (18, 70),
                            white,
                            0.55,
                            1,
                        )
                        draw_text(
                            image,
                            "Front checks: head centering, shoulder level, hip level"
                            if current_view
                            else "Side checks: neck and torso inclination",
                            (18, 105),
                            white,
                            0.5,
                            1,
                        )

                        for landmark in (
                            left_shoulder,
                            right_shoulder,
                            left_hip,
                            right_hip,
                            nose,
                        ):
                            cv2.circle(image, landmark, 5, status_color, -1)
                        cv2.line(
                            image, left_shoulder, right_shoulder, status_color, 2
                        )
                        cv2.line(image, left_hip, right_hip, status_color, 2)
                        cv2.line(
                            image, shoulder_center, hip_center, status_color, 2
                        )

                good_time = good_frames / fps
                bad_time = bad_frames / fps
                time_value = good_time if good_time > 0 else bad_time
                time_label = "GOOD TIME" if good_time > 0 else "BAD TIME"
                time_color = green if good_time > 0 else red
                draw_text(
                    image,
                    f"{time_label}: {time_value:.1f}s  |  ALERT AFTER: {time_threshold}s",
                    (18, height - 22),
                    time_color,
                    0.65,
                )
                if bad_time > time_threshold:
                    sendWarning()

                cv2.imshow("Posture Monitor - press Q to quit", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_arguments()
    print("Arguments:")
    print(f"Video: {args.video}")
    print(f"View: {args.view}")
    print(f"Offset Threshold: {args.offset_threshold}")
    print(f"Neck Angle Threshold: {args.neck_angle_threshold}")
    print(f"Torso Angle Threshold: {args.torso_angle_threshold}")
    print(f"Time Threshold: {args.time_threshold}")
    main(
        args.video,
        args.offset_threshold,
        args.neck_angle_threshold,
        args.torso_angle_threshold,
        args.time_threshold,
        args.view,
    )
