import sys,os
sys.path.append('Packages')
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget,QFileDialog
from PyQt5.QtCore import QDir, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap,QImage
import time
import yaml
import cv2
import numpy as np

from UI.PadXS import *
from UI.Load_model import *
from UI.Warning import *
from UI.Select_model import *
from server.server_main import *
from client.client_core import *
# 加载应用配置Setting.yaml文件的参数
class APP_Config:
    def __init__(self, conf_file):
        with open(conf_file) as fp:
            self.yaml_data = yaml.load(fp, Loader=yaml.FullLoader)

class Model_config:
    '''
    读取yml文件模型的配置参数
    '''
    def __init__(self,conf_file):
        with open(conf_file) as fp:
            self.yml_data = yaml.load(fp, Loader=yaml.FullLoader)

class Serving_thread(QThread):
    signal = pyqtSignal(str)    #设置触发信号传递的参数数据类型,这里是字符串
    signal_stop = pyqtSignal(bool,str)  #serving线程停止信号
    signal_stop_finished = pyqtSignal(str)  #线程停止完成信号
    def __init__(self,channel):
        super(Serving_thread,self).__init__()
        self.channel = channel
        self.keep_running = True

    def stop_cmd(self,signal_stop,new_channel):
        #改变信号，并且更新端口号
        self.channel = new_channel
        self.keep_running = signal_stop

    def run(self):
        #初始化服务端
        self.server = grpc_server(self.channel)
        #启动服务
        self.server.start()
        #启动服务完成打开信号
        self.signal.emit('serving_running')
        #每3s监听一次
        while self.keep_running:
            time.sleep(3)
        #服务线程完毕之后返回信号
        self.signal_stop_finished.emit('stop_finished')

class Predictor_threads(QThread):
    '''
    预测线程
    '''
    signal_img = pyqtSignal(QImage,int) #预测时返回的图片
    signal_str = pyqtSignal(list)    #预测完成之后返回信号
    signal_img_finished = pyqtSignal(int)   #预测完成之后返回信号
    def __init__(self,config,Predicter,save_flag,Predict_cmd):
        super(Predictor_threads,self).__init__()
        self.config = config
        self.Predicter = Predicter
        self.save_flag = save_flag
        self.Predict_cmd = Predict_cmd

    def run(self):
        #图片预测
        if self.config['predict_way'] == 'Image':
            Img = cv2.imread(self.config['picture_path'])
            Result = getattr(self.Predicter,self.Predict_cmd)(Img)
            #结果处理
            if self.config['model_type'] == 'classifier':
                self.signal_str.emit(Result)    #如果是图像分类则返回label
            #检测结果显示
            else:
                QtImg = QImage(Result.data,
                                        Result.shape[1],
                                        Result.shape[0],
                                        Result.shape[1] * 3,
                                        QImage.Format_BGR888)
                self.signal_img.emit(QtImg,1)
                #保存预测结果到文件夹
                if self.save_flag:
                    cv2.imwrite('output.jpg',Result)
        #视频预测
        if self.config['predict_way'] == 'Video':
            cap = cv2.VideoCapture(self.config['video_path'])
            if self.save_flag:
                width  = int(cap.get(3)) # float
                height = int(cap.get(4)) # float
                #写入视频
                fps = 25          # 视频帧率
                size = (width, height) # 需要转为视频的图片的尺寸
                print(size)
                video = cv2.VideoWriter("./output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps,size)      
            self.ret = True #这里设置ret为T最后空帧会是F
            frame = cv2.imread(self.config['face_picture'])#先挂一帧，作为预读
            while self.ret:
                QtImg = QImage(frame.data,
                                        frame.shape[1],
                                        frame.shape[0],
                                        frame.shape[1] * 3,
                                        QImage.Format_BGR888)
                self.signal_img.emit(QtImg,0)
                Result = getattr(self.Predicter,self.Predict_cmd)(frame)
                #检测结果显示
                if self.config['model_type'] == 'classifier':
                    self.signal_str.emit(Result)    #如果是图像分类则返回label
                else:
                    QtImg = QImage(Result.data,
                                            Result.shape[1],
                                            Result.shape[0],
                                            Result.shape[1] * 3,
                                            QImage.Format_BGR888)
                    self.signal_img.emit(QtImg,1)
                    #保存预测结果到文件夹
                    if self.save_flag:
                        Result = Result.astype(np.uint8)
                        video.write(Result)
                    #因为最后一帧是空帧把这个放到最后来这样最后一帧就跳出循环不用读了
                    self.ret, frame = cap.read()
        self.signal_img_finished.emit(1)
class PaddleX_Serving:
    def __init__(self):
        super(PaddleX_Serving, self).__init__()
        #提示框
        self.warning = QWidget()
        self.warning_ui = Ui_Waring()
        self.warning_ui.setupUi(self.warning)
        #读取配置文件
        self.APP_config_init()
        #serving线程
        self.serving = Serving_thread(self.config['server_channel']+self.config['default_port'])
        self.serving.signal.connect(self.finished_serving_start)    #完成启动serving信号
        self.serving.signal_stop.connect(self.serving.stop_cmd)     #退出serving线程信号
        self.serving.signal_stop_finished.connect(self.finished_stop_serving)   #退出serving线程完成的信号
        #主窗口
        self.mainwindows = QMainWindow()
        self.mainwindows_ui = Ui_PaddleX_Serving()
        self.mainwindows_ui.setupUi(self.mainwindows)
        log_Qpixmap = QPixmap(self.config['face_picture'])
        self.mainwindows_ui.Img_display.setPixmap(log_Qpixmap)
        self.mainwindows_ui.Display_Result.setPixmap(log_Qpixmap)
        #选择模型窗口
        self.select_model = QWidget()
        self.select_model_ui = Ui_Set_Model_select()
        self.select_model_ui.setupUi(self.select_model)
        self.select_model_ui.Bt_select_model.clicked.connect(self.select_model_bt)
        #点击打开选择模型
        self.mainwindows_ui.Start_serving.clicked.connect(self.open_select)
        # 加载模型的窗口
        self.load_model = QWidget()
        self.load_model_ui = Ui_Set_Model_load()
        self.load_model_ui.setupUi(self.load_model)
        #open加载模型w
        self.select_model_ui.next_set.clicked.connect(self.open_load)
        #点击加载模型按钮
        self.load_model_ui.Bt_load_model.clicked.connect(self.run_serving)
        #点击完成配置按钮
        self.load_model_ui.finish_set.clicked.connect(self.finished_load)
        #点击选择一张图片的按钮
        self.mainwindows_ui.Select_img.clicked.connect(self.add_picture_path)
        #点击选择一张视频的按钮
        self.mainwindows_ui.Bt_select_video.clicked.connect(self.add_video_path)
        #预测按钮
        self.mainwindows_ui.Bt_Predict.clicked.connect(self.run_predict)


    def APP_config_init(self):
        '''
        读取应用配置文件参数
        '''
        if not os.path.exists('./APPsetting/Setting.yaml'):
            self.warning_ui.label.setText("缺少配置文件APPsetting/Setting.yaml                  程序即将关闭")
            self.warning.show()
            QApplication.processEvents()
            time.sleep(5)
            sys.exit()
        yaml_config = APP_Config('./APPsetting/Setting.yaml')
        self.config ={
            'Author':'Reatris',
            'serving_status':'serving_stop',
            'model_dir':None,
            'default_port':yaml_config.yaml_data['default_port'],
            'server_channel':yaml_config.yaml_data['server_channel'],
            'client_channel':yaml_config.yaml_data['client_channel'],
            'use_gpu': False,
            'gpu_id': '0',
            'load_model_result': 'no_load',
            'predict_way':'Image',
            'face_picture': './source/image/Main_picture.jpg'
        }

            
    def open_select(self):
        '''
        打开模型选择界面
        '''
        if self.config['load_model_result'] == 'finished load model':
            # 关闭线程
            self.mainwindows_ui.Start_serving.setEnabled(False)
            self.serving.signal_stop.emit(False,self.config['real_server_channel'])
            return
        self.serving.keep_running = True
        self.select_model.show()

    def open_load(self):
        '''
        点击 启动服务按钮 打卡模型加载界面
        '''
        if self.config['model_dir'] == None:
            self.warning_ui.label.setText("请先选择模型文件！")
            self.warning.show()
            return
        Device_select = self.select_model_ui.combobox_use_gpu.currentIndex()
        if Device_select == 1:
            self.config['use_gpu'] = True
        else:
            self.config['use_gpu'] = False
        self.load_model_ui.progressBar.hide()   #先隐藏进度条
        self.load_model_ui.progressBar.setProperty("value", 1)
        self.load_model_ui.Bt_load_model.show() #显示加载按钮
        self.load_model_ui.Load_Model_tip.setText('未加载模型')
        self.load_model_ui.Load_Model_tip.setStyleSheet("background-color: rgb(170, 170, 127);")
        self.load_model.show()
        
    def select_model_bt(self):
        '''
        点击选择模型时加载文件夹选择
        '''
        # absolute_path is a QString object
        directory = QtWidgets.QFileDialog.getExistingDirectory(self.select_model, "选择模型所在的文件夹", "./") 
        # absolute_path = QFileDialog.getOpenFileName(self.select_model, 'Open file') #这是选择文件的
        if directory:
            cur_path = QDir('.')
            relative_path = cur_path.relativeFilePath(directory)
            #进行文件验证
            file_list = os.listdir(relative_path)
            flag = 0
            for f in file_list:
                if f == "model.yml":
                    flag +=1
                if f == 'score.yaml':
                    flag += 1
                if f == "__model__":
                    flag += 1
                if f == "__params__":
                    flag += 1
            if flag < 4:
                self.warning_ui.label.setText('模型文件不正确，必须包含导出的model.yml,__model__,__params__和score.yaml文件')
                self.warning.show()
                return
            
            #读取模型yml配置文件
            model_config = Model_config(relative_path+'/'+'model.yml')
            self.config['model_name'] = model_config.yml_data['Model']
            #这里是为了区分模型类别
            if model_config.yml_data['_Attributes']['model_type'] == 'detector' and len(model_config.yml_data['_ModelInputsOutputs']['test_outputs']) == 2:
                self.config['model_type'] = 'detector_seg'  #实例分割模型
            else:
                self.config['model_type'] = model_config.yml_data['_Attributes']['model_type']

            self.config['model_dir'] = relative_path
            self.select_model_ui.model_dir.setText('模型名称：' + self.config['model_name']+' 模型类型：'+self.config['model_type'])
            self.select_model_ui.model_dir.setStyleSheet("background-color: rgba(84, 252, 0, 200);")

    def run_serving(self):
        '''
        启动预测线程
        '''
        self.load_model_ui.Bt_load_model.hide()
        self.load_model_ui.progressBar.show()
        self.load_model_ui.Load_Model_tip.setText("正在加载模型")
        self.load_model_ui.Load_Model_tip.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.serving.start()
        #模拟读条
        count = 1
        for i in range(4):
            time.sleep(0.5)
            self.load_model_ui.progressBar.setProperty("value", count*19)
            count += 1


    def finished_serving_start(self,signal):
        '''
        接收服务启动信号，更新服务状态
        '''
        #更新端口号，因为重启服务端口号不能一样
        self.config['real_client_channel'] = self.config['client_channel']+self.config['default_port']
        self.config['default_port'] = str(int(self.config['default_port'])+10)
        self.config['real_server_channel'] = self.config['server_channel']+self.config['default_port']
        
        self.config['serving_status'] = signal
        #配置预测器
        self.Predicter = Predict_det(
            channel = self.config['real_client_channel'],
            model_dir = self.config['model_dir'],
            use_gpu = self.config['use_gpu'],
            gpu_id = self.config['gpu_id']
        )
        self.config['load_model_result'] = self.Predicter.load_model()
        self.load_model_ui.progressBar.setProperty("value", 100)
        self.load_model_ui.Load_Model_tip.setText(self.config['load_model_result']+ ":" +self.config['model_name'] +'端口：'+self.config['real_client_channel'])
        self.load_model_ui.Load_Model_tip.setStyleSheet("background-color: rgb(71, 255, 43);")

    def finished_load(self):
        '''
        配置完成按钮
        '''
        if self.config['load_model_result'] != "finished load model":
            return
        self.mainwindows_ui.Model_tips.setText('服务已启动')
        self.mainwindows_ui.Model_tips.setStyleSheet("background-color: rgb(71, 255, 43);")
        self.mainwindows_ui.Start_serving.setText('关闭服务')
        self.load_model.close()

    def finished_stop_serving(self,signal_stop_finished):
        '''
        服务线程关闭之后初始化界面和config
        '''
        self.config['load_model_result'] = 'no_load'
        self.config['serving_status'] = 'serving_stop'
        self.mainwindows_ui.Model_tips.setStyleSheet("background-color: rgb(255, 85, 0);")
        self.mainwindows_ui.Model_tips.setText("服务未启动")
        self.mainwindows_ui.Start_serving.setText( "启动服务")
        self.mainwindows_ui.Start_serving.setEnabled(True)
        self.mainwindows_ui.Bt_Predict.setEnabled(False)


    def add_picture_path(self):
        '''
        选择一张图片预测
        '''
        flag = 0
        if self.config['serving_status'] == 'serving_running':
            flag += 1
        if self.config['load_model_result'] == 'finished load model':
            flag += 1
        if flag <2 :
            self.warning_ui.label.setText('服务未启动，请启动服务')
            self.warning.show()
            return
        picture_absolute_path = QFileDialog.getOpenFileName(self.mainwindows, '选择一张图片') #这是选择文件的
        if picture_absolute_path:
            cur_path = QDir('.')
            relative_path = cur_path.relativeFilePath(picture_absolute_path[0])
            if not (relative_path.endswith('.jpg') or relative_path.endswith('.png')):
                self.warning_ui.label.setText('请选择jpg或者png图片')
                self.warning.show()
                return
            self.config['picture_path'] = relative_path
            QPixmap_ = QPixmap(self.config['picture_path'])
            self.mainwindows_ui.Img_display.setPixmap(QPixmap_)
            self.mainwindows_ui.Select_img.setText('已选择图片')
            self.mainwindows_ui.Output.setText('已选择图片'+self.config['picture_path'])
            self.mainwindows_ui.Bt_select_video.setText('选择一段视频')
            self.mainwindows_ui.Bt_Predict.setEnabled(True)
            self.config['predict_way'] = 'Image'

    def add_video_path(self):
        '''
        添加段视频的path用来预测
        '''
        flag = 0
        if self.config['serving_status'] == 'serving_running':
            flag += 1
        if self.config['load_model_result'] == 'finished load model':
            flag += 1
        if flag <2 :
            self.warning_ui.label.setText('服务未启动，请启动服务')
            self.warning.show()
            return
        picture_absolute_path = QFileDialog.getOpenFileName(self.mainwindows, '选择一段视频') #这是选择文件的
        if picture_absolute_path:
            cur_path = QDir('.')
            relative_path = cur_path.relativeFilePath(picture_absolute_path[0])
            if not (relative_path.endswith('.mp4') or relative_path.endswith('.avi')):
                self.warning_ui.label.setText('请选择.mp4或者.avi格式视频')
                self.warning.show()
                return
            self.config['video_path'] = relative_path
            self.mainwindows_ui.Select_img.setText('选择一张图片')
            self.mainwindows_ui.Output.setText('已选择视频'+self.config['video_path'])
            self.mainwindows_ui.Bt_select_video.setText('已选择视频')
            self.mainwindows_ui.Bt_Predict.setEnabled(True)
            self.config['predict_way'] = 'Video'


    def run_predict(self):
        '''
        点击预测按钮完成预测
        '''
        #读取图片
        self.mainwindows_ui.Bt_Predict.setEnabled(False)
        self.mainwindows_ui.Bt_Predict.setText('预测中')
        #按钮锁定
        self.mainwindows_ui.Start_serving.setEnabled(False)
        self.mainwindows_ui.Select_img.setEnabled(False)
        self.mainwindows_ui.Bt_select_video.setEnabled(False)
        #检测模式处理器
        #目标检测
        if self.config['model_type'] == 'detector':
            Predict_cmd = 'pdx_predict_det'
        #实例分割
        if self.config['model_type'] == 'detector_seg':
            Predict_cmd = 'pdx_predict_det_seg'
        #图像分类
        if self.config['model_type'] == 'classifier':
            Predict_cmd = 'pdx_predict_cls'
        #语义分割
        if self.config['model_type'] == 'segmenter':
            Predict_cmd = 'pdx_predict_seg'

        #预测器线程
        self.Predicter_thread = Predictor_threads(
            self.config,
            self.Predicter,
            self.mainwindows_ui.checkBox_save.isChecked(),
            Predict_cmd
        )
        self.Predicter_thread.signal_img.connect(self.draw_img)
        self.Predicter_thread.signal_str.connect(self.draw_label)
        self.Predicter_thread.signal_img_finished.connect(self.after_predicted_img)
        self.Predicter_thread.start()
       
        
        

    def draw_img(self,img,flag):
        '''
        接收线程预测的图片并且放在正确的窗口
        '''
        if self.config['predict_way'] == 'Image':
            self.mainwindows_ui.Display_Result.setPixmap(QPixmap.fromImage(img))
        if self.config['predict_way'] == 'Video':
            if flag == 0:
                self.mainwindows_ui.Img_display.setPixmap(QPixmap.fromImage(img))
            if flag == 1:
                self.mainwindows_ui.Display_Result.setPixmap(QPixmap.fromImage(img))

    def draw_label(self,signal_str):
        '''
        返回图片分类的标签，显示在输出框
        '''
        self.mainwindows_ui.Output.setText('预测结果:'+ str(signal_str))

    def after_predicted_img(self,flag):
        #组件解锁
        self.mainwindows_ui.Bt_Predict.setEnabled(True)
        self.mainwindows_ui.Start_serving.setEnabled(True)
        self.mainwindows_ui.Select_img.setEnabled(True)
        self.mainwindows_ui.Bt_select_video.setEnabled(True)
        self.mainwindows_ui.Bt_Predict.setText('预测')

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    PaddleX_Serving = PaddleX_Serving()
    PaddleX_Serving.mainwindows.show()
    sys.exit(app.exec_())
