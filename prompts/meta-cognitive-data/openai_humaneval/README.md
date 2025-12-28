---
dataset_info:
- config_name: unhelpful
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: test
    dtype: string
  - name: entry_point
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
    num_bytes: 30676316
    num_examples: 164
  download_size: 12724195
  dataset_size: 30676316
- config_name: unhelpful_correct
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: canonical_solution
    dtype: string
  - name: test
    dtype: string
  - name: entry_point
    dtype: string
  - name: correct_thought
    dtype: string
  splits:
  - name: test
    num_bytes: 1506599
    num_examples: 160
  download_size: 634878
  dataset_size: 1506599
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
