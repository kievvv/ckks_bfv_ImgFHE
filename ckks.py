import tenseal as ts
import sqlite3
from PIL import Image
import numpy as np
import os
from time import time

def restored_image():
    # 加密并将每个图像存储到数据库中
    for i in range(1, 101):  # 读取图片
        image_path = f"./data/{i}.jpg"
        values_red, values_green, values_blue = encrypt_image(image_path, context, ts.SCHEME_TYPE.CKKS)

        # 将加密向量的值序列化
        serialized_red = values_red.serialize()
        serialized_green = values_green.serialize()
        serialized_blue = values_blue.serialize()

        # 将序列化的加密向量插入到数据库中
        cursor.execute('''
        INSERT OR REPLACE INTO EncryptedImages (id, encrypted_red, encrypted_green, encrypted_blue)
        VALUES (?, ?, ?, ?)
    ''', (i, sqlite3.Binary(serialized_red), sqlite3.Binary(serialized_green), sqlite3.Binary(serialized_blue)))
        print(f"加密存储第{i}张图片")

def encrypt_image(image_path, context_public, encryption_algorithm):
    image = Image.open(image_path)
    image = image.resize((64, 64))  # 调整大小
    image_array = np.array(image)

    # 分离每个通道
    red_channel = image_array[:, :, 0].flatten().tolist()
    green_channel = image_array[:, :, 1].flatten().tolist()
    blue_channel = image_array[:, :, 2].flatten().tolist()

    # 加密每个通道，使用公钥
    encrypted_red = ts.ckks_vector(context_public, red_channel)
    encrypted_green = ts.ckks_vector(context_public, green_channel)
    encrypted_blue = ts.ckks_vector(context_public, blue_channel)

    return encrypted_red, encrypted_green, encrypted_blue

def calculate_cosine_similarity(vector1, vector2):
    # 分子
    numerator = vector1.dot(vector2)
    numerator = numerator.pow(2)
    # 分母
    denominator1 = vector1.dot(vector1)
    denominator2 = vector2.dot(vector2)
    denominator = denominator1 * denominator2

    numerator = numerator.decrypt(ckks_secret_key)
    denominator = denominator.decrypt(ckks_secret_key)

    numerator_array = np.array(numerator)
    denominator_array = np.array(denominator)
    similarity = np.sum(numerator_array) / np.sum(denominator_array)

    return similarity

#######################################################################################
def decrypt_all_image():
    # 创建或连接到 SQLite 数据库
    conn = sqlite3.connect("ckks_encrypted_images.db")
    cursor = conn.cursor()

    # 查询数据库以获取加密向量
    cursor.execute("SELECT encrypted_red, encrypted_green, encrypted_blue FROM EncryptedImages")
    rows = cursor.fetchall()

    # 创建保存目录
    save_directory = "./ckks_restored_images/"
    os.makedirs(save_directory, exist_ok=True)

    # 解密并还原每个图像
    for i, row in enumerate(rows, start=1):
        # 反序列化加密向量的值
        serialized_red, serialized_green, serialized_blue = row

        # 反序列化加密向量
        deserialized_red = ts.ckks_vector_from(context, serialized_red)
        deserialized_green = ts.ckks_vector_from(context, serialized_green)
        deserialized_blue = ts.ckks_vector_from(context, serialized_blue)

        # 使用私钥解密
        decrypted_red = deserialized_red.decrypt(ckks_secret_key)
        decrypted_green = deserialized_green.decrypt(ckks_secret_key)
        decrypted_blue = deserialized_blue.decrypt(ckks_secret_key)

        # 转换为 NumPy 数组
        red_channel = np.array(decrypted_red)
        green_channel = np.array(decrypted_green)
        blue_channel = np.array(decrypted_blue)

        # 由于解密后是一维数组，需要重新调整为图像形状
        shape = (int(len(decrypted_red) ** 0.5), int(len(decrypted_red) ** 0.5), 1)
        red_channel = red_channel.reshape(shape)
        green_channel = green_channel.reshape(shape)
        blue_channel = blue_channel.reshape(shape)

        # 合并通道并进行值的缩放和截断
        restored_image_array = np.clip(np.concatenate([red_channel, green_channel, blue_channel], axis=2), 0, 255).astype('uint8')


        # 将还原后的图像数组转换为图像对象
        restored_image = Image.fromarray(restored_image_array.astype('uint8'))

        # 保存还原后的图像为 JPG 文件
        restored_image.save(f"./ckks_restored_images/{i}_restored.jpg")
        print(f"解密还原第{i}张图片")

    # 关闭连接
    conn.close()
    print("解密完成")

if __name__ == "__main__":
    # 定义 CKKS context，并分离公私钥
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40  # 设置全局缩放因子
    ckks_secret_key = context.secret_key()
    context.make_context_public()

    # 创建或连接到 SQLite 数据库
    conn = sqlite3.connect("ckks_encrypted_images.db")
    cursor = conn.cursor()

    # 创建表以存储加密向量
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS EncryptedImages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encrypted_red BLOB,
            encrypted_green BLOB,
            encrypted_blue BLOB
        )
    ''')

    t_start = time()
    # 加密并将每个图像存储到数据库中
    restored_image()
    t_end = time()
    print("ckks同态加密用时: {} ms".format((t_end - t_start) * 1000))

    # 提交更改并关闭连接
    conn.commit()
    conn.close()
    print("加密存储完成")

    # 读取要查询的图片并加密
    query_image_path = "./data/5.jpg"
    encrypted_query_red, encrypted_query_green, encrypted_query_blue = encrypt_image(query_image_path, context, ts.SCHEME_TYPE.CKKS)

    # 连接到 SQLite 数据库
    conn = sqlite3.connect("ckks_encrypted_images.db")
    cursor = conn.cursor()

    # 查询数据库以获取加密向量
    cursor.execute("SELECT encrypted_red, encrypted_green, encrypted_blue FROM EncryptedImages")
    rows = cursor.fetchall()

    t_start = time()
    # 计算余弦相似度并排序
    similarities = []

    for i, row in enumerate(rows, start=1):
        # 反序列化数据库中的密文向量
        serialized_red, serialized_green, serialized_blue = row
        encrypted_red = ts.ckks_vector_from(context, serialized_red)
        encrypted_green = ts.ckks_vector_from(context, serialized_green)
        encrypted_blue = ts.ckks_vector_from(context, serialized_blue)

        # 计算余弦相似度
        similarity_red = calculate_cosine_similarity(encrypted_query_red, encrypted_red)
        similarity_green = calculate_cosine_similarity(encrypted_query_green, encrypted_green)
        similarity_blue = calculate_cosine_similarity(encrypted_query_blue, encrypted_blue)

        # 将三个通道的相似度取平均作为整体相似度
        average_similarity = (similarity_red + similarity_green + similarity_blue) / 3

        similarities.append((i, average_similarity))
        print(f"计算第{i}张图片的相似度")

    # 对相似度进行排序
    print("相似度排序中...")
    similarities.sort(key=lambda x: abs(x[1] - 1))

    # 输出相似度最接近1的图像编号和相似度值
    # print(similarities)
    closest_result = similarities[0]

    t_end = time()
    print("ckks密态搜索用时: {} ms".format((t_end - t_start) * 1000))
    print(f"最相似的图片为: {closest_result[0]}, 二者余弦相似度为: {closest_result[1]}")

    # 关闭数据库连接
    conn.close()

    t_start = time()
    # 解密并还原图像
    decrypt_all_image()
    t_end = time()
    print("bfv同态解密用时: {} ms".format((t_end - t_start) * 1000))
