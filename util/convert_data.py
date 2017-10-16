import os
import json
DATA_PATH = '/home/zxy/Datasets/ActiveVision/ActiveVisionDataset/'
SCENE_NAMES = ['Home_001_1','Home_001_2','Home_002_1','Home_003_1','Home_003_2',
               'Home_004_1','Home_004_2','Home_005_1','Home_005_2','Home_006_1','Home_008_1',
               'Home_014_1','Home_014_2','Office_001_1']

#SCENE_IDS = [00011,00012,00021,00031,00032,00041,00042,00051,00052,00061,00081,00141,00142,10011]
DIRS = ['forward', 'backward', 'left', 'right', 'rotate_cw', 'rotate_ccw']

for i, scene in enumerate(SCENE_NAMES):
  path = DATA_PATH + scene
  annot_path = '{}/annotations.json'.format(path)
  move_dir = path + '/moves'
  annot_dir = path + '/annotations'
  if not os.path.exists(move_dir):
    os.mkdir(move_dir)
  if not os.path.exists(annot_dir):
    os.mkdir(annot_dir)
  data = json.load(open(annot_path))

  for img_name, annot in data.items():
    scene_id = img_name[:5]
    #assert int(scene_id) == SCENE_IDS[i]
    img_id = int(img_name[5:11])
    img_tag = img_name[:15]
    move_file = '{}/{}_moves.txt'.format(move_dir, img_tag)
    f = open(move_file, 'w')
    for m in DIRS:
      target_name = annot[m]
      if len(target_name) > 0:
        target = int(target_name[5:11])
      else:
        target = 0
      f.write('{} {}\n'.format(scene_id, target))
    f.close()
    
    annot_file = '{}/{}_boxes.txt'.format(annot_dir, img_tag)
    f = open(annot_file, 'w')
    boxes = annot['bounding_boxes']
    for j in range(len(boxes)):
      f.write('{} {} {} {} {}\n'.format(boxes[j][4], boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]))
    f.close()
    
  


