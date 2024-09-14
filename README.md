# Deep Learning and Large Language Models for Audio and Text Analysis in Predicting Suicidal Acts in Chinese Psychological Support Hotlines

This is the data and code for the paper: Deep Learning and Large Language Models for Audio and Text Analysis in Predicting Suicidal Acts in Chinese Psychological Support Hotlines. The link is https://doi.org/10.48550/arXiv.2409.06164.


## Introduction

- We employ a memory stream conduction to summarize the caller's emotional state and potential suicidal tendencies, aiding the LLM in forming a chain of thought. In the process of constructing memory streams for long psychological support hotline conversations, we segment complex dialogue texts into more manageable small segments. For each segment, we rely on three main components to retrieve memory backgrounds and form new comprehensive summaries: Recency, Importance, and Relevance. The details of code link is: https://github.com/xiaoning317/Suicide_Risk_Prediction/blob/master/LLM/Summarization.py

- Our goal is to use an LLM to directly predict suicide risk based on these summary transcripts. We employ both zero-shot and few-shot prompting to perform the suicide risk prediction task with ChatGPT-4. In the first approach, ChatGPT-4 predicted the suicide risk based on summarization in a zero-shot prompt setting, without any examples provided. In the second approach, we used few-shot prompting, where the LLM was provided with several sample references before predicting the suicide risk. The details of code link is:https://github.com/xiaoning317/Suicide_Risk_Prediction/blob/master/LLM/Prediction.py



## Citation

- Y. Chen, J. Li, C. Song, Q. Zhao, Y. Tong, G. Fu, “Deep Learning and Large Language Models for Audio and Text Analysis in Predicting Suicidal Acts in Chinese Psychological Support Hotlines,” arXiv preprint arXiv:2409.06164, 2024.
