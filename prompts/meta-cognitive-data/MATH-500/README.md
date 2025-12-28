---
dataset_info:
- config_name: unhelpful
  features:
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: subject
    dtype: string
  - name: level
    dtype: int64
  - name: unique_id
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
  - name: test
    num_bytes: 74013468
    num_examples: 500
  download_size: 32538620
  dataset_size: 74013468
- config_name: unhelpful_correct
  features:
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: subject
    dtype: string
  - name: level
    dtype: int64
  - name: unique_id
    dtype: string
  - name: correct_thought
    dtype: string
  splits:
  - name: test
    num_bytes: 3574249
    num_examples: 485
  download_size: 1631832
  dataset_size: 3574249
configs:
- config_name: unhelpful
  data_files:
  - split: test
    path: unhelpful/test-*
- config_name: unhelpful_correct
  data_files:
  - split: test
    path: unhelpful_correct/test-*
---
