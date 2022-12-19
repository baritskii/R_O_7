import cv2
import numpy as np
from tkinter import *
from PIL import *
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "SimHei"# Прямое изменение словаря конфигурации и установка шрифта по умолчанию
    
def Point():   
    img = cv2.imread("img2.jpg")
    height, width =  img.shape[:2]
    height = int(height/3)
    width = int(width/2)

    img = cv2.resize(img, (height, width), cv2.INTER_LINEAR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = img.copy

    # Поскольку оператору Harris требуется входное изображение float32, преобразуйте формат данных
    gray = np.float32(gray)

    # Позвоните оператору Харриса
    R = cv2.cornerHarris(gray, 2, 3, 0.06)

    R = cv2.dilate(R, None)

    # 0,01 - это порог, установленный человеком. Чем меньше значение, тем больше углов
    img[R > 0.01 * R.max()] = [255, 255, 0]

    #plt.subplot(121),plt.imshow(R, cmap='gray')
    #plt.title("R серого изображения")
    #plt.subplot(122),plt.imshow(img[:,:,::-1])
    #plt.title('Угол обнаружения изображения')
    cv2.imshow("img", img)
    cv2.waitKey(0)

def Flow():

    query_img = cv2.imread('img.jpg')

    original_img = cv2.imread('img1.jpg') 


    query_img_bw = cv2.cvtColor(query_img, cv2.IMREAD_GRAYSCALE)
    original_img_bw = cv2.cvtColor(original_img, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()

    queryKP, queryDes = orb.detectAndCompute(query_img_bw,None)
    trainKP, trainDes = orb.detectAndCompute(original_img_bw,None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(queryDes,trainDes)

    matches = sorted(matches, key = lambda x:x.distance)

    final_img = cv2.drawMatches(query_img, queryKP, original_img, trainKP, matches[:20],None)
    
    final_img = cv2.resize(final_img, (1000,650))

    cv2.imshow("Matches", final_img)
    cv2.waitKey()


def Menu():
    window = Tk()

    
    window.title("Menu")

    w = window.winfo_screenwidth()
    h = window.winfo_screenheight()
    w = w//2 # середина экрана
    h = h//2 
    w = w - 200 # смещение от середины
    h = h - 200
    window.geometry('300x200+{}+{}'.format(w, h))
    window.configure(bg='#D0FBFF')

    btn = Button(window, text="Нахождение точек", padx=10, pady=7, command =Point, bg='#7CFFA8')  
    btn.pack(anchor="center", padx=10, pady=20)

    btn = Button(window, text="Сравнение точек", padx=10, pady=7, command =Flow, bg='#7CFFA8')  
    btn.pack(anchor="center", padx=10, pady=15)

    btn1 = Button(window, text="Выход", padx=10, pady=7, command =exit, bg='#7CFFA8')  
    btn1.pack(anchor="center", padx=10, pady=10)
    


    window.mainloop()

Menu()

