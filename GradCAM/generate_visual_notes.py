import sys
import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import math

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

    # okvqa
    qs_path = "/data/fangwenlong/prophet-main/datasets/okvqa/OpenEnded_mscoco_val2014_questions_llava_1.json"
    image_folder = "/data/fangwenlong/prophet-main/datasets/coco2014/val2014/"
    knowledge_path = "/data/fangwenlong/LLaVA-NeXT-inference/LLaVA-NeXT-inference/result/llava_answer_8b_knowledge_and_image_summry.jsonl"
    val_captions = pd.read_csv(
        "/data/fangwenlong/A-Simple-Baseline-For-Knowledge-Based-VQA-main/A-Simple-Baseline-For-Knowledge-Based-VQA-main/annotations/ok_vqa/val_annots_fixed.csv.zip")
    # 指定你想要检查的文件夹路径
    folder_path = 'okvqa_heatmap_onlyknow_black_060/'
    answers_file = "okvqa_heatmap_onlyknow/okvqa_heatmap_onlyquestion_060.jsonl"
    answers_2 = []
    # print(args.answers_file_stage1)
    write = False
    if write == True:
        answers_file = os.path.expanduser(answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")


    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")

    questions = [json.loads(q) for q in open(os.path.expanduser(qs_path), "r")]
    query_instructions = get_chunk(questions, 1, 0)

    # load knowledge notes
    answers_1 = []
    with open(knowledge_path, "r") as file:
        for line in file:
            llava_caption = json.loads(line.strip())  # 解析每一行的JSON对象
            answers_1.append(llava_caption)  # 将JSON对象添加到列表中

    n = 1
    for index, line in enumerate(query_instructions):
        print(n)
        answer_s1 = answers_1[index]

        as_1 = answer_s1["text"]
        qs_id = line["question_id"]
        query_texts = line["question"]+ as_1  # line["question"]+ as_1   line["question"] +   knowledge_all_one_question
        # print(query_texts)


        if len(query_texts) > 250:
            query_texts = query_texts[:250]
        # print(query_texts,"\n")
        context_key = val_captions[val_captions.question_id == qs_id].iloc[0].image_path
        # 确保context_key是有效的文件路径，然后打开图片
        if os.path.isfile(image_folder + context_key):
            original_image = Image.open(image_folder + context_key).convert("RGB")
            # original_image = image.resize((224, 224))
        else:
            print(f"Warning: Image path {image_folder + context_key} does not exist and will be skipped.")

        raw_image = original_image
        question = query_texts
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)  #1,3,384,384
        question = txt_processors["eval"](question)

        samples = {"image": image, "text_input": [question]}
        samples = model.forward_itm(samples=samples) # gradcam[1,576]

        # Gradcam visualisation

        norm_img = np.float32(raw_image) / 255  #569,640,3
        gradcam = samples['gradcams'].reshape(24,24).numpy()

        avg_gradcam = getAttMap(norm_img, gradcam, blur=True)    #569,640,3
        red = True
        if red == True:
            red_channel = avg_gradcam[:, :, 0]  # 取红色通道 (569, 640)

            # 将 avg_gradcam 转换为灰度图（单通道），因为我们只需要关注程度，不需要颜色
            gray_heatmap = np.mean(avg_gradcam, axis=-1)  # 569, 640

            # 设置一个阈值，比如 0.5，生成掩码
            threshold = 0.60 * red_channel.max()
            #threshold = 0.6 * gray_heatmap.max()
            mask = (red_channel > threshold).astype(np.float32)  # 569, 640

            # 扩展掩码到三个通道
            mask = np.stack([mask] * 3, axis=-1)  # 569, 640, 3

            # 原图是浮点数（0~1），将掩码应用到原图上
            masked_image = norm_img * mask
            # 将不关注的区域设置为白色（255, 255, 255）
            white_background = np.ones_like(norm_img)  # 创建一个全白的背景 (569, 640, 3)
            masked_image = masked_image + (1 - mask) * white_background  # 非关注区域设置为白色
            # black_background = np.zeros_like(norm_img)  # 创建一个全黑的背景 (569, 640, 3)
            # masked_image = masked_image + (1 - mask) * black_background  # 非关注区域设置为黑色

            # 将 masked_image 转换回 0-255 范围的 uint8 类型，以便显示
            masked_image_uint8 = np.uint8(masked_image * 255)

        samples = model.forward_cap(samples=samples, num_captions=50, num_patches=20)
        # print(samples['captions'][0][:5])

        # save visual notes
        if write == True:
            ans_file.write(json.dumps({"question_id": qs_id,
                                       "text": samples['captions'][0][:5],
                                       }) + "\n")

        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig, ax = plt.subplots(figsize=(raw_image.size[0] / 100, raw_image.size[1] / 100), dpi=100)
        # ax.imshow(avg_gradcam)
        ax.imshow(masked_image_uint8)   #avg_gradcam   masked_image_uint8
        ax.set_yticks([])
        ax.set_xticks([])
        # plt.show()
        plt.savefig(folder_path + context_key, bbox_inches='tight', pad_inches=0,dpi=100)  # bbox_inches='tight', pad_inches=0
        plt.close(fig)

        n += 1