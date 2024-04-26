import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo
import os
from pyvirtualdisplay import Display

#display = Display(visible=0, size=(1400, 900))
#display.start()


def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    for mp4 in Path(path).glob("*.mp4"):
        os.system(f"start {mp4}")