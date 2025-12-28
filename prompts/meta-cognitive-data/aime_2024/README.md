---
dataset_info:
- config_name: unhelpful
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  - name: year
    dtype: string
  - name: thought__DeepSeek_R1_Distill_Llama_70B
    dtype: string
  - name: thought__DeepSeek_R1_Distill_Qwen_32B
    dtype: string
  - name: thought__DeepSeek_R1_Distill_Qwen_14B
    dtype: string
  - name: thought__DeepSeek_R1_Distill_Llama_8B
    dtype: string
  - name: thought__DeepSeek_R1_Distill_Qwen_7B
    dtype: string
  - name: incorrect_thought
    dtype: string
  - name: uninformative_thought
    dtype: string
  - name: irrelevant_thought
    dtype: string
  - name: irrelevant_thought_10p
    dtype: string
  - name: irrelevant_thought_33p
    dtype: string
  - name: irrelevant_thought_66p
    dtype: string
  - name: hard_irrelevant_question
    dtype: string
  - name: hard_irrelevant_answer
    dtype: string
  - name: s1_hard_irrelevant_thought
    dtype: string
  - name: s1_hard_irrelevant_thought_10p
    dtype: string
  - name: s1_hard_irrelevant_thought_33p
    dtype: string
  - name: s1_hard_irrelevant_thought_66p
    dtype: string
  - name: exaone_hard_irrelevant_thought
    dtype: string
  - name: exaone_hard_irrelevant_thought_10p
    dtype: string
  - name: exaone_hard_irrelevant_thought_33p
    dtype: string
  - name: exaone_hard_irrelevant_thought_66p
    dtype: string
  - name: hard_irrelevant_thought
    dtype: string
  - name: hard_irrelevant_thought_10p
    dtype: string
  - name: hard_irrelevant_thought_33p
    dtype: string
  - name: hard_irrelevant_thought_66p
    dtype: string
  splits:
  - name: train
    num_bytes: 12152804
    num_examples: 30
  download_size: 5165106
  dataset_size: 12152804
- config_name: unhelpful_correct
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  - name: year
    dtype: string
  - name: correct_thought
    dtype: string
  splits:
  - name: train
    num_bytes: 601679
    num_examples: 25
  download_size: 280958
  dataset_size: 601679
configs:
- config_name: unhelpful
  data_files:
  - split: train
    path: unhelpful/train-*
- config_name: unhelpful_correct
  data_files:
  - split: train
    path: unhelpful_correct/train-*
---
