import pandas as pd

def get_class_id_to_label(label_meta_path):
  """Get dictionary of class_id to label

  Parameters:
  label_meta_path (str): path to csv that has label values for class ids

  Returns:
  (dict of int: str): class_id_to_label
  int: num_classes

  """
  label_meta = pd.read_csv(label_meta_path)
  num_classes = len(label_meta["ClassId"].unique())
  class_id_to_label = {k[0]: k[2] for k in label_meta.values.tolist()}
  return class_id_to_label, num_classes