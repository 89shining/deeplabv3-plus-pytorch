#----------------------------------------------------#
#   DeeplabV3+ 批量预测（医学图像 2D切片）
#----------------------------------------------------#
import os
from tqdm import tqdm
from PIL import Image
from deeplab import DeeplabV3

if __name__ == "__main__":
    #=============================================#
    #   加载模型
    #=============================================#
    deeplab = DeeplabV3()
    # 修改为你要使用的权重文件
    deeplab.load_weights("logs/last_epoch_weights.pth")

    # 启用灰度mask输出（背景=0，目标=255）
    deeplab.gray_output = True

    #=============================================#
    #   模式选择为批量预测
    #=============================================#
    mode = "dir_predict"

    #=============================================#
    #   输入/输出路径设置
    #=============================================#
    dir_origin_path = r"D:/project/deeplabv3-plus-pytorch/img"      # 测试集切片
    dir_save_path   = r"D:/project/deeplabv3-plus-pytorch/output"   # 结果保存路径
    os.makedirs(dir_save_path, exist_ok=True)

    #=============================================#
    #   遍历测试集切片进行预测
    #=============================================#
    if mode == "dir_predict":
        img_names = [
            f for f in os.listdir(dir_origin_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ]
        for img_name in tqdm(img_names, desc="Predicting"):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)

            #-----------------------------------------#
            # detect_image 输出二值mask (灰度)
            #-----------------------------------------#
            r_image = deeplab.detect_image(image)

            save_path = os.path.join(dir_save_path, img_name)
            r_image.save(save_path)

        print(f" 批量预测完成！结果保存在: {dir_save_path}")
