# NoteMR

This is the official implementation of the paper "[Notes-guided MLLM Reasoning: Enhancing MLLM with Knowledge and Visual Notes for Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_Notes-guided_MLLM_Reasoning_Enhancing_MLLM_with_Knowledge_and_Visual_Notes_CVPR_2025_paper.html)", which is accepted by CVPR 2025. 

## Abstract
The knowledge-based visual question answering (KB-VQA) task involves using external knowledge about the image to assist reasoning. Building on the impressive performance of multimodal large language model (MLLM), recent methods have commenced leveraging MLLM as an implicit knowledge base for reasoning. However, the direct employment of MLLM with raw external knowledge might result in reasoning errors due to misdirected knowledge information. Additionally, MLLM may lack fine-grained perception of visual features, which can result in hallucinations during reasoning. To address these challenges, we propose **Note**s-guided **M**LLM **R**easoning (**NoteMR**), a novel framework that guides MLLM in better reasoning by utilizing knowledge notes and visual notes. Specifically, we initially obtain explicit knowledge from an external knowledge base. Then, this explicit knowledge, combined with images, is used to assist the MLLM in generating knowledge notes. These notes are designed to filter explicit knowledge and identify relevant internal implicit knowledge within the MLLM. We then identify highly correlated regions between the images and knowledge notes, retaining them as image notes to enhance the model's fine-grained perception, thereby mitigating MLLM induced hallucinations. Finally, both notes are fed into the MLLM, enabling a more comprehensive understanding of the image-question pair and enhancing the model's reasoning capabilities. Our method achieves state-of-the-art performance on the OK-VQA and A-OKVQA datasets, demonstrating its robustness and effectiveness across diverse VQA scenarios.

## Model Architecture

<div align=center>
<img src=".\docs\NoteMR.jpg"/>
</div>
The framework of Notes-guided MLLM Reasoning (NoteMR).

## Environment Requirements
The experiments were conducted on NVIDIA RTX A6000 GPU with 48GB memory. 
* Python 3.10.14
* PyTorch 2.0.1
* CUDA 11.7

To run the MLLM reasoning code, you need to install the requirements:
``` 
pip install -r requirements.txt
```

## Data Download
We evaluate our model using two publicly available KB-VQA dataset. 
* OK-VQA

<a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf" target="_blank">
    <img alt="Paper OKVQA" src="https://img.shields.io/badge/%F0%9F%93%96%20Paper-OKVQA-C6E7FF?logoColor=white" />
</a>
<!-- <a href="https://github.com/allenai/aokvqa" target="_blank">
    <img alt="Github OKVQA" src="https://img.shields.io/badge/Github-OKVQA-F2F2F2?logo=github&logoColor=white" />
</a> -->

* A-OKVQA

<a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680141.pdf" target="_blank">
    <img alt="Paper AOKVQA" src="https://img.shields.io/badge/%F0%9F%93%96%20Paper-AOKVQA-C6E7FF?logoColor=white" />
</a>
<a href="https://github.com/allenai/aokvqa" target="_blank">
    <img alt="Github AOKVQA" src="https://img.shields.io/badge/Github-AOKVQA-F2F2F2?logo=github&logoColor=white" />
</a>


## Run Code


### Step. 1-1 Retrieval (FLMR/PreFLMR)
<a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf" target="_blank">
    <img alt="Paper FLMR" src="https://img.shields.io/badge/%F0%9F%93%96%20Paper-FLMR-C6E7FF?logoColor=white" />
</a>
<a href="https://github.com/linweizhedragon/FLMR" target="_blank">
    <img alt="Github FLMR" src="https://img.shields.io/badge/Github-FLMR-F2F2F2?logo=github&logoColor=white" />
</a>
<a href="https://aclanthology.org/2024.acl-long.289/" target="_blank">
    <img alt="Paper PreFLMR" src="https://img.shields.io/badge/%F0%9F%93%96%20Paper-PreFLMR-C6E7FF?logoColor=white" />
</a>
<a href="https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-G" target="_blank">
    <img alt="Hugging Face PreFLMR" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PreFLMR-ffc107?color=ffc107&logoColor=white" />
</a>

We extract the top-k passages related to the input image and the question with the knowledge retriever, using the pre-trained PreFLMR.

### Step. 1-2 Generate Knowledge Notes
```
python .\generate_knowledge_notes.py
```

### Step. 2 Generate Visual Notes (GradCAM)
<a href="https://ieeexplore.ieee.org/document/8237336" target="_blank">
    <img alt="Paper GradCAM" src="https://img.shields.io/badge/%F0%9F%93%96%20Paper-GradCAM-C6E7FF?logoColor=white" />
</a>
<a href="https://github.com/ramprs/grad-cam/" target="_blank">
    <img alt="Github GradCAM" src="https://img.shields.io/badge/Github-GradCAM-F2F2F2?logo=github&logoColor=white" />
</a>

### Step. 3 Generate Output
```
python .\generate_output.py
```


## Papers for the Project & How to Cite

If you use or extend our work, please cite the paper as follows:
```
@InProceedings{Fang_2025_CVPR,
    author    = {Fang, Wenlong and Wu, Qiaofeng and Chen, Jing and Xue, Yun},
    title     = {Notes-guided MLLM Reasoning: Enhancing MLLM with Knowledge and Visual Notes for Visual Question Answering},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {19597-19607}
}
```
