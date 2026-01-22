import os
import numpy as np
import matplotlib.pyplot as plt
# -------- 讀取單張影像並畫出 bounding box --------
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from PIL import Image
import os

# 範例影像與 XML
img_path = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\影像檔\248.JPG"
xml_path = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\標註檔\248.xml"

# 讀影像
img = Image.open(img_path)

# 讀 XML
tree = ET.parse(xml_path)
root = tree.getroot()

# 建立 figure
fig, ax = plt.subplots(1, figsize=(12,12))
ax.imshow(img)
# 繪製 bounding box
for obj in root.findall("object"):
    name = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    # 繪製矩形框
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=4,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # 標註文字
    ax.text(
        xmin, ymin - 10, name,
        color='red',
        fontsize=20,
        weight='bold'
    )

# 迴圈結束後再呼叫
plt.axis('off')
plt.show()
plt.close()

def visualize_batch(img_dir, xml_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        img_name = xml_file.replace(".xml", ".JPG")  # 對應影像檔名
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"⚠ 找不到影像：{img_name}")
            continue
        # 讀影像
        img = Image.open(img_path)
        # 讀 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 建立 figure
        fig, ax = plt.subplots(1, figsize=(12,12))
        ax.imshow(img)
        obj_count = 0
        for obj in root.findall("object"):
            obj_count += 1
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            # 繪製矩形框
            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=5,        # 線寬
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            # 標註文字
            ax.text(
                xmin, max(ymin - 20, 0), name,
                color='red',
                fontsize=22,
                weight='bold'
            )
        ax.axis('off')
        # 存檔
        out_path = os.path.join(out_dir, img_name)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✔ {img_name} | 物件數量: {obj_count}")
    print("🎉 全部影像 bounding box 繪製完成")

# -------- 設定資料夾 --------
img_dir = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\影像檔"
xml_dir = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\標註檔"
out_dir = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害物件框視覺化"

# 執行批次
visualize_batch(img_dir, xml_dir, out_dir)
