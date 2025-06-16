#1.Load pre-trained models
import os
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import ToPILImage
from transformers import AutoImageProcessor

from flmr import index_custom_collection
from flmr import FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval
import math
import json
from PIL import Image

# load models
checkpoint_path = "/data/fangwenlong/FLMR-main/PreFLMR_ViT-L"
image_processor_name = "/data/fangwenlong/clip-vit-large-patch14"
val_captions = pd.read_csv("/data/fangwenlong/A-Simple-Baseline-For-Knowledge-Based-VQA-main/A-Simple-Baseline-For-Knowledge-Based-VQA-main/annotations/a_ok_vqa/a_ok_vqa_val_fixed_annots.csv.zip")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == "__main__":
    # load model
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        checkpoint_path, subfolder="context_tokenizer"
    )
    model = FLMRModelForRetrieval.from_pretrained(
        checkpoint_path,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

    # 2.Create document collections
    num_items = 100
    feature_dim = 1664
    passage_contents = [f"This is test sentence {i}" for i in range(num_items)]
    filename = '/data/fangwenlong/RAGatouille-main/wikiweb/extracted_values.tsv'    ###数据库读取
    filename = "/data/fangwenlong/RAGatouille-main/RAGatouille-main/aokvqa_data/okvqa_full_clean_corpus.csv"
    my_documents = []

    # 使用 csv.DictReader 读取 TSV 文件，指定制表符为分隔符
    with open(filename, 'r') as file:
        for line in file:
            # 同上处理每一行
            comma_index = line.find(',')
            if comma_index != -1:
                text_content = line[comma_index + 1:].strip().strip('"')
                # print(text_content)
                my_documents.append(text_content)
    # Option 1. text-only documents
    custom_collection = my_documents

    # 1.Create toy query data
    qs_path = "/data/fangwenlong/prophet-main/datasets/aokvqa/aokvqa_v1p0_val1.json"
    image_folder = "/data/fangwenlong/prophet-main/datasets/coco2017/val2017/"
    questions = [json.loads(q) for q in open(os.path.expanduser(qs_path), "r")]
    query_instructions = get_chunk(questions, 1, 0)

    n = 1
    batch_size = 10  # 每批处理的line数量
    results = []  # 存储所有查询的结果
    query_texts = []
    query_images = []
    # 设置JSON文件的路径
    output_json_file_path = 'gs_query_results.json'

    # 检查JSON文件是否存在
    if os.path.exists(output_json_file_path):
        # 如果文件存在，清空其内容
        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            f.write("")  # 写入空字符串清空文件
    else:
        # 如果文件不存在，创建一个空文件用于后续追加
        open(output_json_file_path, 'w').close()
    
    for line in query_instructions:
        qs_id = line["question_id"]
        query_text = "Obtain documents that correspond to the inquiry alongside the provided image: " + line["question"]
        query_texts.append(query_text)
        context_key = val_captions[val_captions.question_id == qs_id].iloc[0].image_path
        # 确保context_key是有效的文件路径，然后打开图片
        if os.path.isfile(image_folder + context_key):
            image = Image.open(image_folder + context_key).convert("RGB")
            query_images.append(image)
        else:
            print(f"Warning: Image path {image_folder + context_key} does not exist and will be skipped.")

        # 每处理10个line，执行查询并保存结果
        if (n % batch_size == 0) or (n == len(query_instructions)):
            num_queries = len(query_texts)
            query_encoding = query_tokenizer(query_texts)
            query_pixel_values = image_processor(query_images, return_tensors="pt")['pixel_values']
            # 2.Obtain query embeddings with model
            inputs = dict(
                input_ids=query_encoding['input_ids'],
                attention_mask=query_encoding['attention_mask'],
                pixel_values=query_pixel_values,
            )

            # Run model query encoding
            res = model.query(**inputs)

            queries = {i: query_texts[i] for i in range(num_queries)}
            # print(queries)
            query_embeddings = res.late_interaction_output

            # 3.Search the collection
            from flmr import search_custom_collection, create_searcher

            # initiate a searcher
            searcher = create_searcher(
                index_root_path=".",
                index_experiment_name="gs_experiment",
                index_name="gs_index",
                nbits=8, # number of bits in compression
                use_gpu=True, # whether to enable GPU searching
            )
            # Search the custom collection
            ranking = search_custom_collection(
                searcher=searcher,
                queries=queries,
                query_embeddings=query_embeddings,
                num_document_to_retrieve=5, # how many documents to retrieve for each query
            )

            # Analyse retrieved documents
            ranking_dict = ranking.todict()

            for i in range(num_queries):
                print(f"Query {i} retrieved documents:")
                retrieved_docs = ranking_dict[i]
                retrieved_docs_indices = [doc[0] for doc in retrieved_docs]
                retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
                retrieved_doc_texts = [my_documents[doc_idx] for doc_idx in retrieved_docs_indices]

                data = {
                    "Question":query_texts[i],
                    "Confidence": retrieved_doc_scores,
                    "Content": retrieved_doc_texts,
                }

                #df = pd.DataFrame.from_dict(data)
                results.append(data)

            # 重置查询和图像列表
            query_texts = []
            query_images = []

        n+=1
    # 将结果列表转换为JSON格式并保存到文件
    with open(output_json_file_path, 'a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    results = []


