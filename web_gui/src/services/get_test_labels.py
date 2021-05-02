import pandas as pd

def get_test_video_name_to_class_id(test_label_path):
  """Get dictionary of video name to class id

  Parameters:
  test_label_path (str): path to csv that has class ids for test videos

  Returns:
  (dict of str: int): test_video_name_to_class_id

  """
  test_label_df = pd.read_csv(test_label_path, header=None)
  test_video_name_to_class_id = {k[0]: k[1] for k in test_label_df.values.tolist()}
  return test_video_name_to_class_id