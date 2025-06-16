import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import copy
from Prompt.prompt import knowledge_notes_prompt
import pandas as pd

val_captions = pd.read_csv("/data/fangwenlong/A-Simple-Baseline-For-Knowledge-Based-VQA-main/A-Simple-Baseline-For-Knowledge-Based-VQA-main/annotations/ok_vqa/val_annots_fixed.csv.zip")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions,answers_1, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.answers_1 = answers_1
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        qs_id = line["question_id"]

        answer_s1 = self.answers_1[index]
        if answer_s1["question_id"] == qs_id:
            knowledges = []
            for knowledge_one in answer_s1["ctxs"]:
                knowledges.append(knowledge_one["text"])

        knowledge_all_one_question = ""
        for index,i in enumerate(knowledges):
            knowledge_all_one_question += "Knowledge "+ str(index+1)+":"+ i + '\n'

        context_key = val_captions[val_captions.question_id == qs_id].iloc[0].image_path
        # context_key = val_captions[val_captions.question_id == image_id]

        # add prompt
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs = knowledge_notes_prompt({"DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN, "knowledge_all_one_question": knowledge_all_one_question})
        # qs = line["question"] +"Here's some additional knowledge you can refer to." + knowledge_all_one_question +"\nAnswer the question using a single word or phrase."    #"\nWhat is shown in this image?" +"\nRequires answers of less than 100 words."

        conv_template = "llava_llama_3"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(self.image_folder + context_key)  # .convert('RGB')
        images = []
        images.append(image)

        image_size = []  # 初始化一个空列表
        for i in images:  # 假设 'images' 是一个包含PIL图像对象的列表
            # 将图像尺寸作为一个元组添加到列表中
            image_size.append((i.width, i.height))
        # 将列表转换为元组
        image_size = tuple(image_size)
        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        image_tensor = process_images(images, self.image_processor,self.model_config)  # 如果有[0]是【5，3，336，336】，没有就是1,5,3,336,336，其中1代表着几张图片

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image_size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, answers_1, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, answers_1, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Load Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # setting save files
    answers_file = os.path.expanduser(args.answers_file_stage2)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # loading retrieved knowledge
    # answers_1 = json.loads(args.answers_file_stage1)
    with open(args.answers_file_stage1, 'r', encoding='utf-8') as file:
        # 加载并解析JSON数据
        answers_1 = json.load(file)
    
    # data loader (add prompt)
    data_loader = create_data_loader(questions, answers_1, args.image_folder, tokenizer, image_processor, model.config)
    try_num = 1

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        try_num += 1
        idx = line["question_id"]
        cur_prompt = line["question"] # "The dog's paws would be in use if it were playing the game of frisbee. Dogs typically use their paws to catch and hold onto frisbees during the game."#line["question"]
        image_id = line["image_id"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        image_tensor = image_tensor[0]  # 本来一个tuple包住了多个图片，现在变成拆开了，变成了多个图
        image_sizes = image_sizes[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        # llm推理解析
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # save file
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "image_id":image_id,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/fangwenlong/lama3-llava-next-8b")   #facebook/opt-350m   /data/fangwenlong/llava-v1.6-mistral-7b    /data/fangwenlong/llava-v1.6-vicuna-13b
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/fangwenlong/prophet-main/datasets/coco2014/val2014/")
    parser.add_argument("--question-file", type=str, default="/data/fangwenlong/prophet-main/datasets/okvqa/OpenEnded_mscoco_val2014_questions_llava_1.json")
    parser.add_argument("--answers-file-stage1", type=str, default="/data/fangwenlong/knowledge/retriever_test.json")
    parser.add_argument("--answers-file-stage2", type=str, default="result/llava_answer_8b_knowledge_and_image_summry1.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")   #v1_mmtag   llava_v1     mistral_instruct    vicuna_v1  ok-vqa
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)    #0.2  0
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
