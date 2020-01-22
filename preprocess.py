import cv2
import numpy as np
import sys
import os

def open_img(path):
	#не используем сv2.imread, т.к. там поддерживается только ascii
	#если есть кириллица в пути - чтение произойдёт правильно
	f = open(path, "rb")
	chunk = f.read()
	chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
	return cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)

#матожидание
def calc_mean(input_dir):
	mean = np.zeros((1,3))
	cnt = 0
	for filename in os.listdir(input_dir):
		image = open_img(input_dir+'\\'+filename)
		mean += np.mean(image, axis=tuple(range(image.ndim-1)))
		cnt +=1
	print("\nmean is", mean/cnt)
	return mean/cnt
	
#стандартное отклонение	
def calc_std(input_dir,mean):
	std = np.zeros((1,3))
	cnt = 0
	for filename in os.listdir(input_dir):
		image = open_img(input_dir+'\\'+filename)
		std += np.mean(np.subtract(image, mean)**2, axis=tuple(range(image.ndim-1)))
		cnt +=1
	print("std is", np.sqrt(std/cnt),"\n")
	return np.sqrt(std/cnt)
	
def preprocess(input_dir, output_dir, mean, std):
	print("Preprocessing...\n")
	for filename in os.listdir(input_dir):
		image = open_img(input_dir+'\\'+filename)
		#нормализация
		image = (image - mean)/std
		
		width,height = np.shape(image)[0],np.shape(image)[1]
		
		#После поворота изображения размером AxB на 45 градусов результат - квадрат со стороной A/sqrt(2) + B/sqrt(2)
		sq_side = width/np.sqrt(2) + height/np.sqrt(2)
		
		#вычисляем значения для дополнения изображения отражением
		brdr_add_width = int((0.75*sq_side - width/2))
		brdr_add_height = int((0.75*sq_side - height/2))
		
		#дополняем изображение отражениями 
		image = cv2.copyMakeBorder(image,brdr_add_width,brdr_add_width,brdr_add_height,brdr_add_height,cv2.BORDER_REFLECT)
		
		#поворот на 45
		M = cv2.getRotationMatrix2D((np.shape(image)[0]/2, np.shape(image)[1]/2), 45, 1)
		image = cv2.warpAffine(image, M, (np.shape(image)[0], np.shape(image)[1]))
		
		#выбираем центральный квадрат со стороной  A/sqrt(2) + B/sqrt(2)
		#изменяем размер на 300x300
		image = cv2.resize(image[int((sq_side)/4):int(1.25*(sq_side)),int((sq_side)/4):int(1.25*(sq_side)),:],(300,300))
		
		#пишем результат
		np.save(output_dir+'\\'+filename[:-4], image, allow_pickle=True)
		
	print("Done")
		
if __name__ == "__main__":
	if (len(sys.argv) != 3):
		print("Wrong number of arguments:", len(sys.argv)-1)
		print("Must be: 2")
		print("Usage: python preprocess.py <input_directory> <output_directory>")
		sys.exit()
	
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	
	mean = calc_mean(input_dir)
	std = calc_std(input_dir, mean)

	preprocess(input_dir, output_dir, mean, std)
