import sys
import os
import re
import numpy as np
import torch
from torchvision import transforms
# services
from services.get_test_labels import get_test_video_name_to_class_id
from services.get_class_id_labels import get_class_id_to_label
# paths to models and utils
module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
  sys.path.append(module_path)
from models.models import CNN_LSTM
from utils.extract_frames import extract_frames

def format_list_to_torch(frames):
  """Convert list of numpy.array to numpy.array of format (batch_size, timesteps, C, H, W)

  Parameters:
  frames (numpy.array of numpy.array): list of frames

  Returns:
  numpy.array: vid_arr

  """
  formatted_frames = torch.from_numpy(frames).float()
  # formatted_frames = formatted_frames.permute(0, 3, 2, 1)
  formatted_frames = formatted_frames.unsqueeze(0)
  return formatted_frames

def get_prediction(video_name, video_path, model_path, test_label_path, label_meta_path):
  """Get predictions and ground truth (if video is in the test dataset) from the input video

  Parameters:
  video_name (str): video name without .mp4 extension
  video_path (str): path to video
  model_path (str): path to trained model
  test_label_path (str): path to csv that has class ids for test videos 
  label_meta_path (str): path to csv that has label values for class ids

  Returns:
  str: ground_truth
  str: output_label

  """
  test_video_name_to_class_id = get_test_video_name_to_class_id(test_label_path)
  class_id_to_label, num_classes = get_class_id_to_label(label_meta_path)
  # transforms_compose = transforms.Compose([transforms.Resize(256)])
  transforms_compose = transforms.Compose([transforms.Resize(256), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5], std=[0.5])])
  frames = extract_frames(video_path, 30, transforms=transforms_compose)
  # normalize the frames
  frames = np.array(frames)
  frames = frames / 255
  formatted_frames = format_list_to_torch(frames)

  model = CNN_LSTM(10, 
                 latent_size=512, 
                 n_cnn_layers=6, 
                 n_rnn_layers=1, 
                 n_rnn_hidden_dim=512, 
                 cnn_bn=True, 
                 bidirectional=True, 
                 dropout_rate=0.8, 
                 attention=True)
  checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()

  output = model.forward(formatted_frames)
  output = torch.max(output, 1)[1]
  output = output.item()

  output_label = class_id_to_label[output]
  try:
    video_name = re.search("signer\d+_sample\d+", video_name).group()
    ground_truth = test_video_name_to_class_id[video_name]
    ground_truth = class_id_to_label[ground_truth]
  except:
    ground_truth = ""
  return ground_truth, output_label