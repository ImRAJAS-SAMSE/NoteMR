# prompt in Equation 5
def knowledge_notes_prompt(input_dict):
    return '''Visual context: {DEFAULT_IMAGE_TOKEN}
Summary information about the retrieved knowledge linked to this image. If the retrieved knowledge is not relevant to the image, return the image captions.
Retrieved knowledge: {knowledge_all_one_question}
Return answers that are as concise as possible and do not exceed 100 words.'''.format(**input_dict)

# prompt in Equation 10
def candidate_output_prompt(input_dict):
    return '''Your task is to perform visual question answering (VQA). You will receive three parts of information:
1. The original image for full context, providing a complete scene.
2. The Region of Interest (ROI) image called visual notes, where colored areas indicate focus and white areas are irrelevant. Mark the important areas related to the problem. Note that the visual notes may contain noise.
3. Additional knowledge, providing background information relevant to the issue.
Please answer the questions in combination with the three parts, mainly based on the information of the original image, using the visual notes as an aid, and referring to additional knowledge to improve accuracy.
Analyzing both the visual notes and the original image. Focus on the areas in the original image that correspond to the colored areas of the visual notes.
Ignore irrelevant noise areas and generate answers that match the question.
Original Image: {DEFAULT_IMAGE_TOKEN}
Visual Notes: {DEFAULT_IMAGE_TOKEN_2}
Knowledge Notes: {knowledge}
Question: {question}
Answer the question using a single word or phrase.
Answer:'''.format(**input_dict)

# prompt in Equation 11
def best_candidate_output_prompt(input_dict):
    return '''Your task is to perform visual question answering (VQA). You will receive three parts of information:
1. The original image for full context, providing a complete scene.
2. The Region of Interest (ROI) image called visual notes, where colored areas indicate focus and white areas are irrelevant. Mark the important areas related to the problem. Note that the visual notes may contain noise.
3. The candidate outputs for you to use as a reference, note that the candidate outputs may contain noise.
4. Additional knowledge, providing background information relevant to the issue.
Please answer the questions in combination with the three parts, mainly based on the information of the original image, using the visual notes as an aid, and referring to additional knowledge and candidate outputs to improve accuracy.
Analyzing both the visual notes and the original image. Focus on the areas in the original image that correspond to the colored areas of the visual notes.
Ignore irrelevant noise areas and generate answers that match the question.
Original Image: {DEFAULT_IMAGE_TOKEN}
Visual Notes: {DEFAULT_IMAGE_TOKEN_2}
Candidate Outputs: {Candidate_Outputs}
Knowledge Notes: {knowledge}
Question: {question}
Answer the question using a single word or phrase.
Answer:'''.format(**input_dict)