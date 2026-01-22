import os
import xml.etree.ElementTree as ET
import pandas as pd

rice_disease_xml_dir = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\標註檔"
# tea_disease_xml_dir = r"C:\Users\user\Downloads\茶病害徵狀影像資料集\茶病害徵狀影像資料集\標註檔"
def create_df(xml_dir):
    rows = []

    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # -------- 影像層級資訊 --------
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)

        objects = root.findall("object")
        object_count = len(objects)

        # -------- 物件層級 --------
        for idx, obj in enumerate(objects, start=1):
            row = {
                "xml_file": file,          # 來源 xml
                "filename": filename,      # 影像名稱
                "width": width,
                "height": height,
                "depth": depth,
                "object_id": idx,          # 第幾個物件
                "object_count": object_count,
                "name": obj.find("name").text,
                "pose": obj.find("pose").text,
                "truncated": int(obj.find("truncated").text),
                "difficult": int(obj.find("difficult").text),
            }

            # -------- Basic_Info_1 ~ 9 --------
            for i in range(1, 10):
                tag = f"Basic_Info_{i}"
                elem = obj.find(tag)
                row[tag] = int(elem.text) if elem is not None else None

            # -------- Bounding Box --------
            bndbox = obj.find("bndbox")
            row["xmin"] = int(bndbox.find("xmin").text)
            row["ymin"] = int(bndbox.find("ymin").text)
            row["xmax"] = int(bndbox.find("xmax").text)
            row["ymax"] = int(bndbox.find("ymax").text)
            rows.append(row)
    # -------- 建立 DataFrame --------
    df = pd.DataFrame(rows)
    print(df.head())
    print(f"總列數（bbox 數量）: {len(df)}")
    return df

rice_disease_df = create_df(rice_disease_xml_dir)

# tea_disease_df = create_df(tea_disease_xml_dir)

# 轉換成多標籤 (multi-label)
rice_disease_df 
rice_disease_label = rice_disease_df[["filename", "name"]].drop_duplicates()
multi_label_df = pd.crosstab(rice_disease_label["filename"], rice_disease_label["name"])
multi_label_df

# 轉換成多類別 (multi-class)
multi_label_df["label"] = multi_label_df.drop(columns="num_labels", errors="ignore").apply(lambda row: row.idxmax(), axis=1) # 產生病害類別
labels = multi_label_df["label"].unique().tolist()
sorted_labels = sorted(labels, key=lambda x: int(x[-2:]))                                                                    # 根據標籤後兩碼排序標籤編號
label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}                                                        
multi_label_df["label_id"] = multi_label_df["label"].map(label_to_id)                                                        # 針對病害類別進行編碼
multi_label_df.to_csv(r"C:\Users\user\Desktop\rice_disease_recogintion_models\rice_disease_label_df.csv")