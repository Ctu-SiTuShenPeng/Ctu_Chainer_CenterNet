import os, cv2, random
import numpy as np
import xml.etree.ElementTree as ET
from data.data_sliceable import GetterDataset

def CreateDataList(IMGDir,XMLDir,train_split):
    ImgList = os.listdir(IMGDir)
    XmlList = os.listdir(XMLDir)
    classes = []
    dataList=[]
    for each_jpg in ImgList:
        each_xml = each_jpg.split('.')[0] + '.xml'
        if each_xml in XmlList:
            dataList.append([os.path.join(IMGDir,each_jpg),os.path.join(XMLDir,each_xml)])
            with open(os.path.join(XMLDir,each_xml), "r", encoding="utf-8") as in_file:
                tree = ET.parse(in_file)
                root = tree.getroot()
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        classes.append(cls)
    random.shuffle(dataList)
    if train_split <=0 or train_split >=1:
        train_data_list = dataList
        val_data_list = dataList
    else:
        train_data_list = dataList[:int(len(dataList)*train_split)]
        val_data_list = dataList[int(len(dataList)*train_split):]
    return train_data_list, val_data_list, classes

class VOCBboxDataset(GetterDataset):
    def __init__(self, data_list, classes_names, use_difficult=False, return_difficult=False):
        super(VOCBboxDataset, self).__init__()

        self.data_list = data_list
        self.classes_names = classes_names
        self.use_difficult = use_difficult

        self.add_getter(('img', 'bbox', 'label', 'difficult'), self._get_data)

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.data_list)
    
    def _get_data(self, i):
        img_path = self.data_list[i][0]
        lab_path = self.data_list[i][1]
        
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        img = img[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        
        anno = ET.parse(lab_path)
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue
            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1 for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.classes_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool)
        
        return img, bbox, label, difficult