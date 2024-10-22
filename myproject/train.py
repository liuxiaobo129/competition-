import argparse

import torch
from transformers import DetrForObjectDetection, DetrFeatureExtractor
from torch.utils.data import DataLoader
import torchvision.transforms as T

from myproject.dataset.customdataset import CustomDataset

def collate_fn(batch):
    images, annotations = zip(*batch)  # 将批次中的图像和标签分别解压出来
    return list(images), list(annotations)

def main():

    def parse_args():
        parser = argparse.ArgumentParser(description='Train a detector')
        parser.add_argument("root_path", default='/',description='Train a detector')
        parser.add_argument("batch_size", default=4,description='Train a detector')
        args = parser.parse_args()
        return args

    # 定义一些参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "facebook/detr-resnet-50"

    # 加载 DETR 模型和特征提取器
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)
    feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)



    # 自定义数据集的变换
    transform = T.Compose([
        # 随机裁剪或缩放，避免图片过度失真，保证标签的准确性
        T.RandomResizedCrop(size=(800, 800), scale=(0.8, 1.0)),
        # 转换为 Tensor，并且归一化图像通道，使得图像值在 [-1, 1] 范围内
        T.ToTensor(),
    ])

    root_path = '/Users/liuxiaobo/Downloads/adownload/competition/training_anno.csv';

    train_dataset =  CustomDataset(root_path,transform)


    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 准备优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 训练循环
    model.train()
    count = 0
    for epoch in range(1):  # 训练5个Epoch
        for batch in train_dataloader:
            # 获取图像和标注
            images, annotations = batch
            # 将图像发送到设备 (GPU 或 CPU)
            pixel_values = [image.to(device) for image in images]
            # 处理标注
            labels_ = []

            annotations_to_labels(annotations, labels_)
            # 使用特征提取器进行处理
            encoding = feature_extractor(images=pixel_values, return_tensors="pt")

            outputs = model(pixel_values=encoding['pixel_values'].to(device),
                            labels=labels_)
            # 计算损失
            loss = outputs.loss
            print(f"Loss: {loss.item()}")
            count =count + 1
            print("train couting ---------- ----------", count );
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 保存模型
        save_path = f"path_to_save/detr_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved after epoch {epoch + 1} at {save_path}")


def annotations_to_labels(annotationList, labels_):
    for annotations in annotationList:
        for annotation in annotations:
            annotation2 = annotation['bbox']
            result = []
            label_arr = []
            for i in range(len(annotation2[0])):
                label_arr.append(0)
                sublist = []
                for tensor in annotation2:
                    sublist.append(tensor[i].item())
                result.append(sublist)
            labels_.append(
                {"class_labels": torch.tensor(label_arr), "boxes": torch.tensor(result, dtype=torch.float32)})



if __name__ == '__main__':
    main();



