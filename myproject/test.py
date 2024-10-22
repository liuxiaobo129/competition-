import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image


def max_row_dict():
    max_row_dict = {}
    max_values = np.max(probas, axis=1)  # 每一行的最大值
    max_indices = np.argmax(probas, axis=1)  # 每一行最大值的列索引
    # 创建字典，按列索引分组，并保留最大值最大的一行
    for i, (value, index) in enumerate(zip(max_values, max_indices)):
        if index not in max_row_dict:
            max_row_dict[index] = (value, i)  # 记录最大值和行号
        else:
            # 如果已经有相同列索引，比较最大值大小，保留较大的那行
            if value > max_row_dict[index][0]:
                max_row_dict[index] = (value, i)
    return max_row_dict
def load_model(model_path, device):
    # 定义模型并加载权重
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 切换到评估模式
    return model

# 将 [x_center, y_center, width, height] 转换为 [x_min, y_min, x_max, y_max]
def convert_to_coco_format(bboxes_scaled):
    bboxes_scaled_coco = np.zeros_like(bboxes_scaled)
    bboxes_scaled_coco[:, 0] = bboxes_scaled[:, 0] - 0.5 * bboxes_scaled[:, 2]  # x_min
    bboxes_scaled_coco[:, 1] = bboxes_scaled[:, 1] - 0.5 * bboxes_scaled[:, 3]  # y_min
    bboxes_scaled_coco[:, 2] = bboxes_scaled[:, 0] + 0.5 * bboxes_scaled[:, 2]  # x_max
    bboxes_scaled_coco[:, 3] = bboxes_scaled[:, 1] + 0.5 * bboxes_scaled[:, 3]  # y_max
    return bboxes_scaled_coco

# 根据数值生成红色的强度，数值越高红色越深
def generate_color():
    # 红色强度随着数值增加
    random_value = np.random.rand()
    return (random_value, 0, 0)  # 返回 RGB 颜色，红色为主

def predict(image_path, model, device):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    image = Image.open(image_path);
    width, height = image.size
    # 处理图像
    inputs = processor(images=image, return_tensors="pt")
    # 进行推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取结果
    logits = outputs.logits  # 预测类别
    bboxes = outputs.pred_boxes  # 预测边界框

    # 设置阈值
    threshold = 0.9
    # 取出概率
    probas = logits.softmax(-1)[0, :, :-1]  # 不包括背景类
    keep = probas.max(-1).values > threshold  # 选择高于阈值的索引

    # 选择保留的边界框
    bboxes_scaled = bboxes[0, keep].cpu().numpy()
    probas = probas[keep].cpu().numpy()

    # 转换边界框为 COCO 格式并转换为像素坐标
    bboxes_scaled_coco = convert_to_coco_format(bboxes_scaled)
    bboxes_scaled_pixel = bboxes_scaled_coco.copy()
    bboxes_scaled_pixel[:, [0, 2]] *= width  # x_min 和 x_max
    bboxes_scaled_pixel[:, [1, 3]] *= height  # y_min 和 y_max
    bboxes_scaled_pixel = np.round(bboxes_scaled_pixel).astype(int)

    # 可视化结果
    plt.imshow(image)

    def max_row_dict():
        global max_row_dict, index
        max_values = np.max(probas, axis=1)  # 每一行的最大值
        max_indices = np.argmax(probas, axis=1)  # 每一行最大值的列索引
        # 创建字典，按列索引分组，并保留最大值最大的一行
        max_row_dict = {}
        for i, (value, index) in enumerate(zip(max_values, max_indices)):
            if index not in max_row_dict:
                max_row_dict[index] = (value, i)  # 记录最大值和行号
            else:
                # 如果已经有相同列索引，比较最大值大小，保留较大的那行
                if value > max_row_dict[index][0]:
                    max_row_dict[index] = (value, i)

    # max_row_dict()

    # 提取保留的行
    rows_to_keep = [row_idx for _, row_idx in max_row_dict().values()]
    bboxes_scaled_pixel = bboxes_scaled_pixel[rows_to_keep]
    probas = probas[rows_to_keep]


    for box, prob in zip(bboxes_scaled_pixel, probas):
        p = prob.argmax(-1)
        m = prob.max()
        color = generate_color(p)
        x_min, y_min, x_max, y_max = box
        plt.gca().add_patch(
            plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color=color, linewidth=2))
        plt.text(x_min, y_min, f"{p}", fontsize=12, color=color)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载保存的模型
    model_path = "path_to_save/detr_epoch_5.pth"
    model = load_model(model_path, device)

    # 进行推理
    image_path = "/path/to/test/image.jpg"  # 测试图像路径
    outputs, predicted_boxes = predict(image_path, model, device)

    print("Predicted boxes:", predicted_boxes)
    # 根据需求解析并处理预测结果


