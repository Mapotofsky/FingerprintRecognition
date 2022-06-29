import os
import tkinter as tk
import serial
import cv2
from tkinter.constants import BOTH, BOTTOM, E, FLAT, HORIZONTAL, LEFT, N, NW, RIGHT, S, TOP, W, X, YES
from tkinter.filedialog import *
from tkinter import messagebox
from PIL import Image, ImageTk
from getFingerprint import GET_FINGERPRINT
from Fingerprint import get_all_imgs
from savenpz import save_npz


serialPort_Finger = "COM8"  # 串口号
serialPort_Control = "COM8"  # 串口号
baudRate = 57600  # 波特率


def getFingerPic():
    """获取指纹图像并保存到本地
    """
    ser = serial.Serial(serialPort_Finger, baudRate, timeout=0.2)
    print("参数设置: 串口= %s ，波特率= %d" % (serialPort_Finger, baudRate))

    GET_FINGERPRINT(ser, 'pic/origin.png')
    ser.close()


def recognition():
    getFingerPic()
    score = get_all_imgs('pic/origin.png')
    if (score >= 0.69):
        print('指纹验证通过')
    else:
        print('指纹验证未通过')


class index(object):
    """选择功能
    """
    
    def goto_selectPic(self):
        self.root.destroy()
        selectPic()
        
    def goto_recognition(self):
        recognition()
        
        for root, dirs, files in os.walk('img', topdown=False):
            for name in files:
                if (name != 'p.txt'):
                    os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        
    def goto_fingerprintIn(self):
        getFingerPic()
        
        get_all_imgs('pic/origin.png')
        
        name = len(os.listdir('pic/samples'))//2
        img = cv2.imread('pic/origin.png', cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f'pic/samples/{name}.png', img)
        save_npz(f'pic/samples/{name}.npz', img)
        print('指纹录入成功，编号:', name)
        
        for root, dirs, files in os.walk('img', topdown=False):
            for name in files:
                if (name != 'p.txt'):
                    os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))        
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("+600+300")
        self.root.title('指纹识别系统')
        
        tk.Label(self.root, text='选择功能', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        tk.Button(self.root, text='录入指纹', font=('等线', 16), command=self.goto_fingerprintIn).pack(side=LEFT, padx=70, pady=15)
        tk.Button(self.root, text='识别模式', font=('等线', 16), command=self.goto_recognition).pack(side=LEFT, pady=15)
        tk.Button(self.root, text='教学模式', font=('等线', 16), command=self.goto_selectPic).pack(side=RIGHT, padx=70, pady=15)
        
        self.root.mainloop()


class selectPic(object):
    """选择图片
    """
    picPath = ''
    
    def getpic(self):
        getFingerPic()
        global picPath
        picPath = 'pic/origin.png'
        img_open = Image.open(picPath)
        img = ImageTk.PhotoImage(img_open)
        self.image_label.config(image=img)
        self.image_label.image = img
    
    def choosepic(self):
        path_ = askopenfilename()
        self.path.set(path_)
        global picPath
        picPath = self.file_entry.get()
        img_open = Image.open(picPath)
        img = ImageTk.PhotoImage(img_open)
        self.image_label.config(image=img)
        self.image_label.image = img
    
    def prev(self):
        self.root.destroy()
        index()
        
    def next(self):
        self.root.destroy()
        global picPath
        get_all_imgs(picPath)
        segmentation()
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.frame1 = tk.Frame(self.root)
        self.frame2 = tk.Frame(self.root)
        
        tk.Label(self.root, text='选择图片', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        tk.Button(self.frame1, text='录入指纹', font=('等线', 12), command=self.getpic).pack(side=LEFT, padx=5)
        tk.Button(self.frame1, text='打开路径', font=('等线', 12), command=self.choosepic).pack(side=LEFT, padx=5)
        self.file_entry = tk.Entry(self.frame1, state='readonly', text=self.path)
        self.file_entry.pack(side=RIGHT, expand=YES, fill=X)
        self.frame1.pack(fill=X)
        
        tk.Button(self.frame2, text='返回', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=15)
        tk.Button(self.frame2, text='开始', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=15)
        self.image_label = tk.Label(self.frame2)
        self.image_label.pack()
        self.frame2.pack(fill=BOTH, expand=YES)
        
        self.root.mainloop()

   
class segmentation(object):
    """指纹分割
    """
        
    def prev(self):
        if (self.count == 0):
            self.root.destroy()
            index()
        else:
            if (self.count == 1):
                self.text.config(text='此乃原图')
                img_open = Image.open('img/ori/0.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 2):
                self.text.config(text=f'利用sobel算子粗略估计边缘情况\n\n该图为x方向的梯度')
                img_open = Image.open('img/sobel/1.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 3):
                self.text.config(text=f'利用sobel算子粗略估计边缘情况\n\n该图为y方向的梯度\nxy方向的梯度在之后预估局部脊线的方向时会用到')
                img_open = Image.open('img/sobel/2.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 4):
                self.text.config(text=f'对sobel算子的结果进行平方，方便观察\n\n该图为x方向的梯度')
                img_open = Image.open('img/sobel_squared/3.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 5):
                self.text.config(text=f'对sobel算子的结果进行平方，方便观察\n\n该图为y方向的梯度')
                img_open = Image.open('img/sobel_squared/4.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 6):
                self.text.config(text=f'对sobel算子的结果进行平方，方便观察\n\n该图为二者叠加的结果')
                img_open = Image.open('img/sobel_squared/5.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 7):
                self.text.config(text=f'使用一个积分算子，对原图进行模糊，方便分割')
                img_open = Image.open('img/integral/6.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            self.count -= 1
            
        
    def next(self):
        if (self.count == 0):
            self.text.config(text=f'利用sobel算子粗略估计边缘情况\n\n该图为x方向的梯度')
            img_open = Image.open('img/sobel/1.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 1):
            self.text.config(text=f'利用sobel算子粗略估计边缘情况\n\n该图为y方向的梯度\nxy方向的梯度在之后预估局部脊线的方向时会用到')
            img_open = Image.open('img/sobel/2.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 2):
            self.text.config(text=f'对sobel算子的结果进行平方，方便观察\n\n该图为x方向的梯度')
            img_open = Image.open('img/sobel_squared/3.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 3):
            self.text.config(text=f'对sobel算子的结果进行平方，方便观察\n\n该图为y方向的梯度')
            img_open = Image.open('img/sobel_squared/4.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 4):
            self.text.config(text=f'对sobel算子的结果进行平方，方便观察\n\n该图为二者叠加的结果')
            img_open = Image.open('img/sobel_squared/5.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 5):
            self.text.config(text=f'使用一个积分算子(与平滑相比像素间差别更大)，对图片进行模糊，方便分割')
            img_open = Image.open('img/integral/6.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 6):
            self.text.config(text=f'分割结果如下')
            img_open = Image.open('img/threshold/9.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 7):
            self.root.destroy()
            ridgeOrientation()
        self.count += 1
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='指纹分割', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/ori/0.png')
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text='此乃原图', font=('等线', 11), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()


class ridgeOrientation(object):
    """预估局部脊线的方向
    """
    
    def prev(self):
        self.root.destroy()
        segmentation()
        
    def next(self):
        self.root.destroy()
        ridgeFrequency()
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='预估脊线方向', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/Orientation/10.png')
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text=f'基于梯度信息，使用cv2.phase计算边缘的方向场，可得左图结果\n\n方向场信息将用于后续的图像增强处理', font=('等线', 11), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()


class ridgeFrequency(object):
    """预估局部脊线的频率
    """
    
    def prev(self):
        self.root.destroy()
        ridgeOrientation()
        
    def next(self):
        self.root.destroy()
        picEnhance()
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='预估脊线频率', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/plot1/plot1.png')
        img_open = img_open.resize((256, 480*256//640), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text=f'首先假设指纹频率处处相同!\n\n在这个假设下，我们可以只取指纹的一个方向进行频率的计算\n简单起见，这里只在y轴方向上计算脊线像素的累计分布\n\n频率信息将用于后续的图像增强处理', font=('等线', 11), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()
        
        
class picEnhance(object):
    """指纹图像增强
    """
    
    def prev(self):
        if (self.count == 0):
            self.root.destroy()
            ridgeFrequency()
        else:
            if (self.count == 1):
                self.text.config(text='基于前面计算得到的频率和方向场，可以构建一组Gabor-filter来对原图进行各方向的卷积融合，从而达到比普通的增强算子更好的效果')
                img_open = Image.open('img/after_filters/20.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                self.filter.config(image='')
                self.filter.image = ''
            elif (self.count == 2):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/21.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/12.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 3):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/22.png')
                img = ImageTk.PhotoImage(img_open)
                self.filter.config(image=img)
                self.filter.image = img
                filter_open = Image.open('img/8_filters/13.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 4):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/23.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/14.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 5):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/24.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/15.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 6):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/25.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/16.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 7):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/26.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/17.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 8):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/27.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/18.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            elif (self.count == 9):
                self.text.config(text=f'各方向卷积核及其卷积结果如下')
                img_open = Image.open('img/after_filters/28.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
                filter_open = Image.open('img/8_filters/19.png')
                filter = ImageTk.PhotoImage(filter_open)
                self.filter.config(image=filter)
                self.filter.image = filter
            self.count -= 1
            
        
    def next(self):
        if (self.count == 0):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/21.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/12.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 1):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/22.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/13.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 2):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/23.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/14.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 3):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/24.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/15.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 4):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/25.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/16.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 5):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/26.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/17.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 6):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/27.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/18.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 7):
            self.text.config(text=f'各方向卷积核及其卷积结果如下')
            img_open = Image.open('img/after_filters/28.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            filter_open = Image.open('img/8_filters/19.png')
            filter = ImageTk.PhotoImage(filter_open)
            self.filter.config(image=filter)
            self.filter.image = filter
        elif (self.count == 8):
            self.text.config(text=f'对于每个像素，找到与其脊线方向最接近的filter，并将相应的卷积结果合并到最后的图像中，如此组装所有的像素点，得到左图结果')
            img_open = Image.open('img/enhanced/30.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
            self.filter.config(image='')
            self.filter.image = ''
        elif (self.count == 9):
            self.root.destroy()
            getMinutiae()
        self.count += 1
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='指纹图像增强', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/after_filters/20.png')
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text='基于前面计算得到的频率和方向场，可以构建一组Gabor-filter来对原图进行各方向的卷积融合，从而达到比普通的增强算子更好的效果', font=('等线', 11), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        self.filter = tk.Label(self.root, image='')
        self.filter.pack()
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()


class getMinutiae(object):
    """特征点提取
    """
        
    def prev(self):
        if (self.count == 0):
            self.root.destroy()
            picEnhance()
        else:
            if (self.count == 1):
                self.text.config(text='增强后的指纹图像仍保留了灰度信息，为方便后续的形态学处理，先将其二值化')
                img_open = Image.open('img/binarization/32.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 2):
                self.text.config(text=f'将指纹细化，提取骨架')
                img_open = Image.open('img/crossing/37.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 3):
                self.text.config(text=f'筛选出8邻域内像素由黑变白1次或3次的点，分别标记为终结点(terminations)和分叉点(bifurcations)')
                img_open = Image.open('img/crossing/38.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            elif (self.count == 4):
                self.text.config(text=f'筛选出8邻域内像素由黑变白1次或3次的点，分别标记为终结点(terminations)和分叉点(bifurcations)\n\n在原图上看更清楚点')
                img_open = Image.open('img/crossing/36.png')
                img = ImageTk.PhotoImage(img_open)
                self.image_label.config(image=img)
                self.image_label.image = img
            self.count -= 1
            
        
    def next(self):
        if (self.count == 0):
            self.text.config(text=f'将指纹细化，提取骨架')
            img_open = Image.open('img/crossing/37.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 1):
            self.text.config(text=f'筛选出8邻域内像素由黑变白1次或3次的点，分别标记为终结点(terminations)和分叉点(bifurcations)')
            img_open = Image.open('img/crossing/38.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 2):
            self.text.config(text=f'筛选出8邻域内像素由黑变白1次或3次的点，分别标记为终结点(terminations)和分叉点(bifurcations)\n\n在原图上看更清楚点')
            img_open = Image.open('img/crossing/36.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 3):
            self.text.config(text=f'对特征点进行筛选，剔除距离指纹边缘太近的点')
            img_open = Image.open('img/minutiae/41.png')
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.image_label.image = img
        elif (self.count == 4):
            self.root.destroy()
            minutiaeOrientation()
        self.count += 1
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='特征点提取', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/binarization/32.png')
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text='增强后的指纹图像仍保留了灰度信息，为方便后续的形态学处理，先将其二值化', font=('等线', 11), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()


class minutiaeOrientation(object):
    """计算特征点处方向
    """
    
    def prev(self):
        self.root.destroy()
        getMinutiae()
        
    def next(self):
        self.root.destroy()
        minutiaeExpress()
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='计算特征点方向', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/draw_minutiae/44.png')
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text=f'为了保证指纹特征的旋转不变性，在进行特征表示前需要对指纹方向进行归一化处理，因此这里先计算各特征点的方向\n\n标准定义请参阅ISO/IEC 19794-2, 2005', font=('等线', 11), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()


class minutiaeExpress(object):
    """Minutia Cylinder-Code (MCC)
    """
    
    def prev(self):
        self.root.destroy()
        minutiaeOrientation()
        
    def next(self):
        self.root.destroy()
        minutiaeCompare()
           
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='Minutia Cylinder-Code', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
        
        img_open = Image.open('img/xijiedian1/45.png')
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        self.text = tk.Label(self.root, text=f'原文:\nMinutia Cylinder-Code: a new representation and matching technique for fingerprint recognition", IEEE tPAMI 2010', font=('等线', 10), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        
        
        def print_selection(v):
            """滑块组件的触发函数
            """
            filename = self.files[int(v) - 1]
            img_open = Image.open(f'img/xijiedian1/{filename}')
            global img  # 设置为全局变量，否则不会显示
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.root.update()
        
        
        self.files = os.listdir('img/xijiedian1')
        pic_num = len(self.files)
        self.s = tk.Scale(self.root, label='select minutiae point', from_=1, to=pic_num, orient=tk.HORIZONTAL, length=200, showvalue=1, tickinterval=pic_num//10, resolution=1, command=print_selection)
        self.s.pack()
        
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='下一步', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        self.root.mainloop()


class minutiaeCompare(object):
    """指纹匹配
    """
    
    def prev(self):
        self.root.destroy()
        minutiaeExpress()
    
    def next(self):
        if (self.p >= 0.69):
            print('指纹验证通过')
        else:
            print('指纹验证未通过')
           
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('500x420+600+300')
        self.root.title('指纹识别系统')
        
        self.path = tk.StringVar()
        self.count = 0
        self.frame = tk.Frame(self.root)
        
        tk.Label(self.root, text='特征点匹配', font=('等线', 20), bg='green', width=20, height=2).pack(pady=10)
                
        self.files = os.listdir('img/xijiedian2')
        pic_num = len(self.files)
        filename = self.files[0]
        img_open = Image.open(f'img/xijiedian2/{filename}')
        img_open = img_open.resize((256, 364*256//640), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img_open)
        self.image_label = tk.Label(self.root, image=img)
        self.image_label.pack(side=LEFT)
        
        file = open('img/p.txt')
        file_data = file.readlines()
        self.p = float(file_data[0])
        self.text = tk.Label(self.root, text='匹配率: {:.2f}'.format(self.p), font=('等线', 14), anchor=NW, width=190, wraplength=190, justify='left')
        self.text.pack(padx=20, pady=35)
        
        
        def print_selection(v):
            """滑块组件的触发函数
            """
            filename = self.files[int(v) - 1]
            img_open = Image.open(f'img/xijiedian2/{filename}')
            img_open = img_open.resize((256, 364*256//640), Image.ANTIALIAS)
            global img  # 设置为全局变量，否则不会显示
            img = ImageTk.PhotoImage(img_open)
            self.image_label.config(image=img)
            self.root.update()
        

        self.s = tk.Scale(self.root, label='select minutiae point', from_=1, to=pic_num, orient=tk.HORIZONTAL, length=200, showvalue=1, tickinterval=1, resolution=1, command=print_selection)
        self.s.pack()
        
        tk.Button(self.frame, text='上一步', font=('等线', 14), command=self.prev).pack(side=LEFT, padx=20)
        tk.Button(self.frame, text='发命令', font=('等线', 14), command=self.next).pack(side=RIGHT, padx=20)
        self.frame.pack(fill=X, expand=YES, anchor=S, pady=30)
        
        
        def on_closing():
            if messagebox.askokcancel("Quit", "确定退出?"):
                self.root.destroy()
                for root, dirs, files in os.walk('img', topdown=False):
                    for name in files:
                        if (name != 'p.txt'):
                            os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                

        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        
        self.root.mainloop()


if __name__ == '__main__':
    index()
