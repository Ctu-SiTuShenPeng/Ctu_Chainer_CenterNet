import chainer,os,json,cv2,time,random
import numpy as np
from chainer import datasets
import matplotlib.font_manager as fm
from chainer.datasets import TransformDataset, ConcatenatedDataset
from chainer.optimizers import Adam, SGD
from chainer.training import StandardUpdater, Trainer, extensions, triggers
from data.data_loader import CreateDataList,VOCBboxDataset
from data.data_transforms import CenterDetectionTransform,DataAugmentationTransform
from nets.center_detector import CenterDetector,CenterDetectorTrain
from nets.backbone.hourglassnet import HourglassNet
from nets.backbone.simplenet import SimpleCNN
from core.detection_voc_evaluator import DetectionVOCEvaluator
from PIL import Image,ImageFont,ImageDraw

class Ctu_CenterNet:
    def __init__(self,USEGPU='0',image_size=512) -> None:
        if USEGPU == '-1':
            self.gpu_devices = -1
        else:
            self.gpu_devices = int(USEGPU)
            chainer.cuda.get_device_from_id(self.gpu_devices).use()
        self.image_size = image_size

    def InitModel(self, DataDir, train_split=0.9,batch_size=4,Pre_Model=None,alpha=1):
        self.alpha = alpha
        self.batch_size = batch_size
        train_data_list, val_data_list, self.classes_names = CreateDataList(os.path.join(DataDir,'DataImage'),os.path.join(DataDir,'DataLabel'),train_split)

        train_data = VOCBboxDataset(train_data_list,self.classes_names)
        test_data = VOCBboxDataset(val_data_list, self.classes_names, use_difficult=True, return_difficult=True)
        data_augmentation_transform = DataAugmentationTransform(self.image_size)
        center_detection_transform = CenterDetectionTransform(self.image_size, len(self.classes_names), 4, dtype=np.float32)
        train_data = TransformDataset(train_data, data_augmentation_transform)
        train_data = TransformDataset(train_data, center_detection_transform)
        # self.train_iter = chainer.iterators.MultiprocessIterator(train_data, self.batch_size)
        self.train_iter = chainer.iterators.SerialIterator(train_data, self.batch_size)
        self.test_iter = chainer.iterators.SerialIterator(test_data, self.batch_size, repeat=False, shuffle=False)

        detector = CenterDetector(HourglassNet,self.image_size, len(self.classes_names), dtype=np.float32)
        self.model = CenterDetectorTrain(detector, 1, 0.1, 1)
        # detector = CenterDetector(SimpleCNN, self.image_size, len(self.classes_names))
        # self.model = CenterDetectorTrain(detector, 1, 0, 0)
        if self.gpu_devices >= 0:
            self.model.to_gpu(self.gpu_devices)
        
    def train(self,TrainNum=150,learning_rate=0.0001,ModelPath='result_Model'):
        optimizer = Adam(alpha=1.25e-4)
        optimizer.setup(self.model)

        updater = chainer.training.updaters.StandardUpdater(self.train_iter, optimizer, device=self.gpu_devices)
        trainer = chainer.training.Trainer(updater, (TrainNum, 'epoch'), out=ModelPath)
        
        trainer.extend(
            extensions.snapshot_object(self.model.center_detector, 'Ctu_final_Model.npz'),
            trigger=chainer.training.triggers.ManualScheduleTrigger([each for each in range(1,TrainNum)], 'epoch')
        )

        trainer.extend(
            extensions.snapshot_object(self.model.center_detector, 'Ctu_best_Model.npz'),
            trigger=chainer.training.triggers.MaxValueTrigger(
                'validation/main/map',
                trigger=chainer.training.triggers.ManualScheduleTrigger([each for each in range(1,TrainNum)], 'epoch')
            ),
        )
        
        trainer.extend(
            DetectionVOCEvaluator(self.test_iter, self.model.center_detector, use_07_metric=True,label_names=self.classes_names),
            trigger=chainer.training.triggers.ManualScheduleTrigger([each for each in range(1,TrainNum)], 'epoch')
        )
        
        log_interval = 0.1, 'epoch'
        trainer.extend(extensions.LogReport(filename='ctu_log.json',trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.dump_graph("main/loss", filename='ctu_net.net'))
        
        print_loss = ['main/loss', 'main/hm_loss', 'main/wh_loss', 'main/offset_loss', 'main/hm_mae', 'main/hm_pos_loss', 'main/hm_neg_loss']
        print_acc = ['validation/main/map']
        trainer.extend(extensions.PrintReport(['epoch', 'lr']+print_loss+print_acc), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))

        os.makedirs(ModelPath,exist_ok=True)
        ClassDict = {
            "classes_names": self.classes_names,
            "image_size": self.image_size,
            'alpha':self.alpha
        }
        with open(os.path.join(ModelPath, 'ctu_params.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(ClassDict, sort_keys=True, indent=4, separators=(',', ': ')))

        trainer.run()
        
    def LoadModel(self,ModelPath):
        params_file = os.path.join(ModelPath,'ctu_params.json')
        with open(params_file, 'r', encoding='utf-8') as f:
            ClassDict = json.load(f)
        self.classes_names = ClassDict["classes_names"]
        self.image_size = ClassDict["image_size"]
        self.alpha = ClassDict["alpha"]

        if os.path.exists(os.path.join(ModelPath,'Ctu_best_Model.npz')):
            ModelFile = os.path.join(ModelPath,'Ctu_best_Model.npz')
        elif os.path.exists(os.path.join(ModelPath,'Ctu_final_Model.npz')):
            ModelFile = os.path.join(ModelPath,'Ctu_final_Model.npz')
        else:
            raise "NoModelFile"

        self.model = CenterDetector(HourglassNet,self.image_size, len(self.classes_names), dtype=np.float32)
        chainer.serializers.load_npz(ModelFile, self.model)
        if self.gpu_devices>=0:
            self.model.to_gpu()
        
        self.colors = []
        for _ in range(0,len(self.classes_names)):
            self.colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    
    def predict(self,img_cv,predict_score=0.5):
        start_time = time.time()
        pre_img = []
        for each_img in img_cv:
            # image = resize(img_cv, (self.image_size, self.image_size))
            image = cv2.cvtColor(each_img, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            pre_img.append(image)
        with chainer.using_config('train', False):
            predicted = self.model.predict(pre_img, detail=True)
        
        predict_num = len(pre_img)
        
        bboxes_result=[]
        imges_result = []
        for each_index in range(0,predict_num):
            b=[]
            thresh_idx = predicted[2][each_index] > predict_score
            bboxes = predicted[0][each_index][thresh_idx]
            labels = predicted[1][each_index][thresh_idx]
            scores = predicted[2][each_index][thresh_idx]

            origin_img_pillow = self.cv2_pillow(img_cv[each_index])
            font = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * np.shape(origin_img_pillow)[1] + 0.5).astype('int32'))
            thickness = max((np.shape(origin_img_pillow)[0] + np.shape(origin_img_pillow)[1]) // self.image_size, 1)

            bbox, bb_label, score = bboxes.tolist(), labels.tolist(), scores.tolist()
            for each_bbox in range(len(bbox)):
                bbox_score = float(score[each_bbox])
                if predict_score > bbox_score:
                    continue

                ymin, xmin, ymax, xmax = [int(x) for x in bbox[each_bbox]]
                # xmin = int(xmin / self.image_size * base_imageSize[1])
                # xmax = int(xmax / self.image_size * base_imageSize[1])
                # ymin = int(ymin / self.image_size * base_imageSize[0])
                # ymax = int(ymax / self.image_size * base_imageSize[0])
                class_name = self.classes_names[int(bb_label[each_bbox])]
                b.append([(xmin,xmax,ymin,ymax),class_name,bbox_score])

                top, left, bottom, right = ymin, xmin, ymax, xmax
                label = '{}-{:.3f}'.format(class_name, bbox_score)
                draw = ImageDraw.Draw(origin_img_pillow)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[int(bb_label[each_bbox])])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[int(bb_label[each_bbox])])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            
            bboxes_result.append(b)
            imges_result.append(self.pillow_cv2(origin_img_pillow))

        result_data={
            'img_num':predict_num,
            'imges_result':imges_result,
            'bboxes_result':bboxes_result,
            'time':time.time()-start_time
        }
        return result_data
    
    def read_image(self,filename):
        cv_img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), 1)
        return cv_img
    
    def cv2_pillow(self,img):
         return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    
    def pillow_cv2(self,img):
        return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    
if __name__ == '__main__':
    # ctu = Ctu_CenterNet(USEGPU='0',image_size=512)
    # ctu.InitModel(r'/home/ctu/Ctu_Project/DL_Project/DataDir/DataSet_Detection_YaoPian',train_split=0.9, batch_size=1,Pre_Model=None,alpha=1)
    # ctu.train(TrainNum=150,learning_rate=0.0001, ModelPath='result_Model')
    
    ctu = Ctu_CenterNet(USEGPU='0')
    ctu.LoadModel('result_Model')
    predictNum=1
    predict_cvs = []
    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 480)
    for root, dirs, files in os.walk(r'/home/ctu/Ctu_Project/DL_Project/DataDir/DataSet_Detection_YaoPian/DataImage'):
        for f in files:
            if len(predict_cvs) >= predictNum:
                predict_cvs.clear()
            img_cv = ctu.read_image(os.path.join(root, f))
            if img_cv is None:
                continue
            predict_cvs.append(img_cv)
            if len(predict_cvs) == predictNum:
                result = ctu.predict(predict_cvs,0.3)
                print(result['time'])
                for each_id in range(result['img_num']):
                    for each_bbox in result['bboxes_result'][each_id]:
                        print(each_bbox)
                    cv2.imshow("result", result['imges_result'][each_id])
                    cv2.waitKey()
    
    # import cv2
    # from chainercv.visualizations import vis_bbox
    # import matplotlib.pyplot as plt
    # import numpy as np
    # voc_bbox_label_names = ['黑圆', '黑椭圆', '土圆', '赭圆', '鱼肝油', '棕白胶囊', '棕棕胶囊', '蓝白胶囊', '白椭圆', '白圆']
    # image = cv2.imread("/home/ctu/Ctu_Project/DL_Project/DataDir/DataSet_Detection_YaoPian/DataImage/pill_bag_002.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.transpose((2, 0, 1))
    # size = 512
    # # transform = CenterDetectionTransform(size, len(voc_bbox_label_names), 4)
    # with chainer.using_config('train', False):
    #     num_class = len(voc_bbox_label_names)
    #     detector = CenterDetector(HourglassNet, size, num_class)
    #     chainer.serializers.load_npz('./result_Model/Ctu_final_Model.npz', detector)
    #     predicted = detector.predict([image], detail=True)
    #     thresh_idx = predicted[2][0] > 0.3
    #     ax = vis_bbox(
    #         image,
    #         predicted[0][0][thresh_idx],
    #         predicted[1][0][thresh_idx],
    #         predicted[2][0][thresh_idx],
    #         label_names=voc_bbox_label_names,
    #     )
    #     plt.show()

    #     output = predicted[3]
    #     resized_image = cv2.resize(image.transpose((1, 2, 0)), (size, size))
    #     for cls in range(num_class):
    #         print(cls)
    #         plt.imshow(resized_image)
    #         hm = output['hm'].data[0, cls]
    #         if hm.max() > 0.3:
    #             hm_img = cv2.resize(hm, (size, size))
    #             plt.title(voc_bbox_label_names[cls])
    #             plt.imshow(hm_img, alpha=0.8, cmap=plt.cm.jet)
    #             plt.colorbar()
    #             plt.show()