from utils import read_video, save_video
from trackers import PlayerTracker


def main():
    input_video_path = "input_video/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect player
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames)

    # Draw output
    # Draw player bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    # Save video
    save_video(output_video_frames, "output_video/output_video.avi")


if __name__ == "__main__":
    main()
