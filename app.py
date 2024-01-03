from flask import Flask, request, jsonify, send_file
#import mysql.connector

app = Flask(__name__)

import numpy as np
from sklearn.cluster import KMeans
import cv2

import json

num_colors = 6  # 指定顏色數量

# 定義函式用於判斷顏色類別
def get_color_category(color):
    r, g, b = color

    # 設定閥值範圍
    gray_threshold = 30
    cool_threshold = 60
    warm_threshold = 60
  
    # 判斷顏色類別
    if r == g == b:
        print(" gray") 
        return 'gray'
    elif g - r > cool_threshold or r < cool_threshold :
        print(" cool") 
        return 'cool'
    elif r  > warm_threshold :
        print(" warm") 
        return 'warm'
    else:
        print(" unknown") 
        return 'unknown'
    
# 判断图片是否为灰度图  
def checkGray(chip):
    chip_gray = cv2.cvtColor(chip,cv2.COLOR_BGR2GRAY)
    r,g,b = cv2.split(chip)
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    s_w, s_h = r.shape[:2]
    x = (r+b+g)/3
    # x = chip_gray
    r_gray = abs(r-x)
    g_gray = abs(g-x)
    b_gray=  abs(b-x)
    r_sum = np.sum(r_gray)/(s_w * s_h)
    g_sum = np.sum(g_gray)/(s_w * s_h)
    b_sum = np.sum(b_gray)/(s_w * s_h)
    gray_degree = (r_sum+g_sum+b_sum)/3
    if gray_degree <10:
        print("Gray")
        return 1
    else:
        print ("NOT Gray")
        return 2

def find_image_color(image_files, color_hex):
    selected_images = []
    for i in range(0, len(image_files)):        
        # 讀取單張照片
        image_path = image_files[i]  # 替換為實際的照片路徑

        # 讀取圖片
        image = cv2.imread(image_path)

        #cv2.imshow('My Image', image)
        #cv2.waitKey(0)  

        # 设置目标图像尺寸
        target_width = 200  # 替换为你想要的宽度
        target_height = int(image.shape[0] * (target_width / image.shape[1]))

        # 降采样/调整图像尺寸
        resized_image = cv2.resize(image, (target_width, target_height))

        # 判断图片是否为灰度图
        if checkGray(resized_image) == 1:
            # 灰度图像，直接分类为灰度
            main_color = 'gray'
        else:
            # 將圖片轉換為RGB色彩空間
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # 將圖片轉換為一維數組
            pixels = image_rgb.reshape(-1, 3)
            # 執行K-Means算法
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(pixels)
            # 取得每個像素的標籤
            labels = kmeans.labels_
            # 取得每個顏色的RGB值
            colors = kmeans.cluster_centers_
            # 計算每個色系的像素數量
            counts = np.bincount(labels)
            # 排序顏色和像素數量
            sorted_colors = colors[np.argsort(counts)][::-1]
            sorted_counts = np.sort(counts)[::-1]
            # 計算冷色調和暖色調的比例加總
            cool_sum = 0
            warm_sum = 0
            print(f"Kmeans 分群後6個色系:")
            for i in range(num_colors):
                color = sorted_colors[i]
                print(i, end=' ')
                print(":", end=' ')
                category = get_color_category(color)
                if category == 'cool':
                    cool_sum += sorted_counts[i]
                elif category == 'warm':
                    warm_sum += sorted_counts[i]                
            
            # 計算冷色調和暖色調的比例
            total_sum = cool_sum + warm_sum
            if total_sum == 0 :
                main_color = 'gray'
            else:
                cool_ratio = cool_sum / total_sum
                warm_ratio = warm_sum / total_sum
                # 判斷最高比例的色調類別
                if cool_ratio > warm_ratio:
                    main_color = 'cool'
                    main_ratio = cool_ratio
                else:
                    main_color = 'warm'
                    main_ratio = warm_ratio

        # 印出結果
        print(f"")
        print(f"圖片名稱：{image_path}")
        print(f"主要色調：{main_color}")
        
        if(color_hex == main_color):
            selected_images.append(image_path)

    return selected_images

from PIL import Image
import numpy as np

def find_closest_image(image_files, color_hex):
    # 將 hex 顏色轉換為 RGB
    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))

    min_diff = float('inf')
    closest_image = None

    for image_file in image_files:
        # 讀取圖片並轉換為 numpy array
        image = Image.open(image_file)
        image_array = np.array(image)

        # 計算圖片中每個像素與指定顏色的差異
        diff = np.sqrt(np.sum((image_array - color_rgb)**2, axis=-1))

        # 找出與指定顏色最相近的圖片
        if np.mean(diff) < min_diff:
            min_diff = np.mean(diff)
            closest_image = image_file

    return "[\"" + closest_image + "\"]"

def sql_connect():
    return mysql.connector.connect(
        host='localhost',
        user='aos_app',
        password='nP9VU@/bBIrHe)rI',
        database='aos_app'
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    # Init DB connection
    #cnx = sql_connect()
    #cursor = cnx.cursor()
    
    result = []
    
    # Check uploaded file
    if len(request.files) == 0:
        result.append({'status': 'error', 'reason': 'No file uploaded'})
    elif 'user_id' not in request.form:
        result.append({'status': 'error', 'reason': 'Missing user_id'})
    elif 'color' not in request.form:
        result.append({'status': 'error', 'reason': 'Missing color'})
    else:
        user_id = request.form['user_id']
        color = request.form['color']
        photos = []
        for file in request.files.getlist('file'):
            file.save(user_id + '_' + file.filename)
            photos.append(user_id + '_' + file.filename)
        # 執行 photo_handler() 並將結果存入資料庫
        # 資料: request.files.keys()
        print("File saved, now running algorithm with color =", color)
        if color == "cool" or color == "warm" or color == "gray":
            print("Using find_image_color()")
            best_photo = json.dumps(find_image_color(photos, color))
        else:
            print("Using find_closest_image()")
            best_photo = find_closest_image(photos, color)
        print('best photo:', best_photo)
        #cursor.execute("INSERT INTO files (filename, user_id) VALUES (%s, %s)", (file.filename, user_id))
        #cnx.commit()
        result.append({'status': 'success', 'reason': 'File upload successful', 'result': best_photo})
    #print('->', result)
    return jsonify(result)

@app.route('/picture', methods=['GET'])
def get_picture():
    filename = request.args.get('filename', '')
    if filename:
        return send_file(filename, mimetype='image/gif')
    else:
        return jsonify([])
'''
@app.route('/history', methods=['GET'])
def get_files():
    # Init DB connection
    #cnx = sql_connect()
    #cursor = cnx.cursor()
    
    result = []
    
    # get user_id
    user_id = request.args.get('user_id', '')
    #print('get_files(): Get user_id =', user_id, 'type =', type(user_id))
    
    # issue sql command to get result
    if user_id:
        cursor.execute("SELECT * FROM files WHERE user_id = %s", (user_id,))
        rows = cursor.fetchall()
        for row in rows:
            result.append({'id': row[0], 'filename': row[1], 'user_id': row[2]})
    #else:
    #    print('get_files(): user_id is null!')

    return jsonify(result)
'''
if __name__ == '__main__':
    app.run(debug=False)