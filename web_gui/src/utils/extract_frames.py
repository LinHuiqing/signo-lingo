import numpy as np
import av

def extract_frames(video_path, frames_cap, transforms=None):
  """Extract and transform video frames

  Parameters:
  video_path (str): path to video file
  frames_cap (int): number of frames to extract, evenly spaced
  transforms (torchvision.transforms, optional): transformations to apply to frame

  Returns:
  list of numpy.array: vid_arr

  """
  frames = []
  with av.open(video_path) as container:
    stream = container.streams.video[0]
    n_frames = stream.frames
    remainder = n_frames % frames_cap
    interval = n_frames // frames_cap
    take_frame_idx = 0
    for frame_no, frame in enumerate(container.decode(stream)):
      if frame_no == take_frame_idx:
        img = frame.to_image()
        if transforms:
          img = transforms(img)
        frames.append(np.array(img))
        if remainder > 0:
          take_frame_idx += 1
          remainder -= 1
        take_frame_idx += interval
  if len(frames) < frames_cap:
    raise ValueError("Ensure video has >={} frames".format(frames_cap))
  return frames