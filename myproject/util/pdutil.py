import json
import pandas as pd


def split_col_to_rows(pdb):
    # 初始化一个空的列表，用于存储拼接结果
    result_list = []
    # 遍历每一行的x列和y列
    for i, row in pdb.iterrows():
        # 分割 x 列，得到各个元素
        split_values = json.loads(row['Polygons'])
        # 将每个分割值与对应的 y 列值拼接，并添加到结果列表
        for value in split_values:
            result_list.append([row['Path'], str([value])])
    # 将结果列表转换为 DataFrame
    final_result = pd.DataFrame(result_list, columns=['Path', 'Polygons'])
    return final_result