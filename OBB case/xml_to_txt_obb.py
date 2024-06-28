# https://zhuanlan.zhihu.com/p/391137600
import xml.etree.ElementTree as ET
import math
import glob
from pathlib import Path



def convert_robndbox(size, xmlbox, name):
    cx, cy, w, h = (
        float(xmlbox.find("cx").text),
        float(xmlbox.find("cy").text),
        float(xmlbox.find("w").text),
        float(xmlbox.find("h").text),
    )
    angle = float(xmlbox.find("angle").text)

    x1 = cx + (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle)
    y1 = cy + (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)
    x2 = cx - (w / 2) * math.cos(angle) - (h / 2) * math.sin(angle)
    y2 = cy - (w / 2) * math.sin(angle) + (h / 2) * math.cos(angle)
    x3 = cx - (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle)
    y3 = cy - (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle)
    x4 = cx + (w / 2) * math.cos(angle) + (h / 2) * math.sin(angle)
    y4 = cy + (w / 2) * math.sin(angle) - (h / 2) * math.cos(angle)

    # https://github.com/otamajakusi/yolov5_obb/blob/master/docs/GetStart.md
    if name == "type1":
        name = "0"
    elif name == "type2":
        name = "1"
    elif name == "type3":
        name = "2"
    elif name == "type4":
        name = "3"
    return f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {name} "


def convert_annotation(xml, classes):
    tree = ET.parse(xml)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    yolo = ""
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        
        while True:
            xmlbox = obj.find("robndbox")
            if xmlbox:
                yolo += convert_robndbox((w, h), xmlbox, cls)
                
            if int(difficult) == 1:
                yolo+= "1\n"
                break
            else:
                yolo+= "0\n"
                break

    return yolo


def voc2yolo(path: Path, class_file):
    if path.is_dir():
        xmls = [Path(f) for f in glob.glob(f"{path}/*.xml")]
    else:
        xmls = [path]
    classes = open(class_file).read().splitlines()
    for xml in xmls:
        with open(xml) as in_file:
            yolo = convert_annotation(in_file, classes)
            with open(f"{xml.parent / xml.stem}.txt", "w") as out_file:
                out_file.write(yolo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--path", required=True)
    parser.add_argument("--class-file", required=True)
    args = parser.parse_args()

    voc2yolo(Path(args.path), args.class_file)
