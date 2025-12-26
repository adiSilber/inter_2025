---
dataset_info:
- config_name: unhelpful
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - name: answerKey
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
    num_bytes: 461660
    num_examples: 1119
  - name: test
    num_bytes: 103666883
    num_examples: 1172
  - name: validation
    num_bytes: 126560
    num_examples: 299
  download_size: 87002808
  dataset_size: 104255103
- config_name: unhelpful_correct
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - name: answerKey
    dtype: string
  - name: correct_thought
    dtype: string
  splits:
  - name: train
    num_bytes: 354236
    num_examples: 1119
  - name: test
    num_bytes: 3946597
    num_examples: 1160
  - name: validation
    num_bytes: 97856
    num_examples: 299
  download_size: 2105295
  dataset_size: 4398689
configs:
- config_name: unhelpful
  data_files:
  - split: train
    path: unhelpful/train-*
  - split: test
    path: unhelpful/test-*
  - split: validation
    path: unhelpful/validation-*
- config_name: unhelpful_correct
  data_files:
  - split: train
    path: unhelpful_correct/train-*
  - split: test
    path: unhelpful_correct/test-*
  - split: validation
    path: unhelpful_correct/validation-*
---
