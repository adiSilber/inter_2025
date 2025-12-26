---
dataset_info:
- config_name: unhelpful
  features:
  - name: Pre-Revision Question
    dtype: string
  - name: Pre-Revision Correct Answer
    dtype: string
  - name: Pre-Revision Incorrect Answer 1
    dtype: string
  - name: Pre-Revision Incorrect Answer 2
    dtype: string
  - name: Pre-Revision Incorrect Answer 3
    dtype: string
  - name: Pre-Revision Explanation
    dtype: string
  - name: Self-reported question-writing time (minutes)
    dtype: float64
  - name: Question
    dtype: string
  - name: Correct Answer
    dtype: string
  - name: Incorrect Answer 1
    dtype: string
  - name: Incorrect Answer 2
    dtype: string
  - name: Incorrect Answer 3
    dtype: string
  - name: Explanation
    dtype: string
  - name: Revision Comments (from Question Writer)
    dtype: string
  - name: Subdomain
    dtype: string
  - name: Writer's Difficulty Estimate
    dtype: string
  - name: Extra Revised Question
    dtype: string
  - name: Extra Revised Explanation
    dtype: string
  - name: Extra Revised Correct Answer
    dtype: string
  - name: Extra Revised Incorrect Answer 1
    dtype: string
  - name: Extra Revised Incorrect Answer 2
    dtype: string
  - name: Extra Revised Incorrect Answer 3
    dtype: string
  - name: Non-Expert Validator Accuracy
    dtype: float64
  - name: Majority Non-Expert Vals Incorrect
    dtype: float64
  - name: Expert Validator Accuracy
    dtype: float64
  - name: Record ID
    dtype: string
  - name: High-level domain
    dtype: string
  - name: Question Writer
    dtype: string
  - name: Feedback_EV_1
    dtype: string
  - name: Validator Revision Suggestion_EV_1
    dtype: string
  - name: Is First Validation_EV_1
    dtype: bool
  - name: Post hoc agreement_EV_1
    dtype: string
  - name: Sufficient Expertise?_EV_1
    dtype: string
  - name: Understand the question?_EV_1
    dtype: string
  - name: Question Difficulty_EV_1
    dtype: string
  - name: Validator Answered Correctly_EV_1
    dtype: int64
  - name: Self-reported time (minutes)_EV_1
    dtype: float64
  - name: Probability Correct_EV_1
    dtype: string
  - name: Manual Correctness Adjustment_EV_1
    dtype: string
  - name: Expert Validator_EV_1
    dtype: string
  - name: Feedback_EV_2
    dtype: string
  - name: Validator Revision Suggestion_EV_2
    dtype: string
  - name: Is First Validation_EV_2
    dtype: bool
  - name: Post hoc agreement_EV_2
    dtype: string
  - name: Sufficient Expertise?_EV_2
    dtype: string
  - name: Understand the question?_EV_2
    dtype: string
  - name: Question Difficulty_EV_2
    dtype: string
  - name: Validator Answered Correctly_EV_2
    dtype: int64
  - name: Self-reported time (minutes)_EV_2
    dtype: float64
  - name: Probability Correct_EV_2
    dtype: string
  - name: Manual Correctness Adjustment_EV_2
    dtype: string
  - name: Expert Validator_EV_2
    dtype: string
  - name: Feedback_NEV_1
    dtype: string
  - name: Validator Answered Correctly_NEV_1
    dtype: int64
  - name: Explanation_NEV_1
    dtype: string
  - name: Self-reported time (minutes)_NEV_1
    dtype: float64
  - name: Websites visited_NEV_1
    dtype: string
  - name: Probability Correct_NEV_1
    dtype: string
  - name: Manual Correctness Adjustment_NEV_1
    dtype: 'null'
  - name: Non-Expert Validator_NEV_1
    dtype: string
  - name: Feedback_NEV_2
    dtype: string
  - name: Validator Answered Correctly_NEV_2
    dtype: int64
  - name: Explanation_NEV_2
    dtype: string
  - name: Self-reported time (minutes)_NEV_2
    dtype: float64
  - name: Websites visited_NEV_2
    dtype: string
  - name: Probability Correct_NEV_2
    dtype: string
  - name: Manual Correctness Adjustment_NEV_2
    dtype: 'null'
  - name: Non-Expert Validator_NEV_2
    dtype: string
  - name: Feedback_NEV_3
    dtype: string
  - name: Validator Answered Correctly_NEV_3
    dtype: float64
  - name: Explanation_NEV_3
    dtype: string
  - name: Self-reported time (minutes)_NEV_3
    dtype: float64
  - name: Websites visited_NEV_3
    dtype: string
  - name: Probability Correct_NEV_3
    dtype: string
  - name: Manual Correctness Adjustment_NEV_3
    dtype: 'null'
  - name: Non-Expert Validator_NEV_3
    dtype: string
  - name: Expert Validator Disagreement Category
    dtype: float64
  - name: Canary String
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
    num_bytes: 68188260
    num_examples: 198
  download_size: 27849310
  dataset_size: 68188260
- config_name: unhelpful_correct
  features:
  - name: Pre-Revision Question
    dtype: string
  - name: Pre-Revision Correct Answer
    dtype: string
  - name: Pre-Revision Incorrect Answer 1
    dtype: string
  - name: Pre-Revision Incorrect Answer 2
    dtype: string
  - name: Pre-Revision Incorrect Answer 3
    dtype: string
  - name: Pre-Revision Explanation
    dtype: string
  - name: Self-reported question-writing time (minutes)
    dtype: float64
  - name: Question
    dtype: string
  - name: Correct Answer
    dtype: string
  - name: Incorrect Answer 1
    dtype: string
  - name: Incorrect Answer 2
    dtype: string
  - name: Incorrect Answer 3
    dtype: string
  - name: Explanation
    dtype: string
  - name: Revision Comments (from Question Writer)
    dtype: string
  - name: Subdomain
    dtype: string
  - name: Writer's Difficulty Estimate
    dtype: string
  - name: Extra Revised Question
    dtype: string
  - name: Extra Revised Explanation
    dtype: string
  - name: Extra Revised Correct Answer
    dtype: string
  - name: Extra Revised Incorrect Answer 1
    dtype: string
  - name: Extra Revised Incorrect Answer 2
    dtype: string
  - name: Extra Revised Incorrect Answer 3
    dtype: string
  - name: Non-Expert Validator Accuracy
    dtype: float64
  - name: Majority Non-Expert Vals Incorrect
    dtype: float64
  - name: Expert Validator Accuracy
    dtype: float64
  - name: Record ID
    dtype: string
  - name: High-level domain
    dtype: string
  - name: Question Writer
    dtype: string
  - name: Feedback_EV_1
    dtype: string
  - name: Validator Revision Suggestion_EV_1
    dtype: string
  - name: Is First Validation_EV_1
    dtype: bool
  - name: Post hoc agreement_EV_1
    dtype: string
  - name: Sufficient Expertise?_EV_1
    dtype: string
  - name: Understand the question?_EV_1
    dtype: string
  - name: Question Difficulty_EV_1
    dtype: string
  - name: Validator Answered Correctly_EV_1
    dtype: int64
  - name: Self-reported time (minutes)_EV_1
    dtype: float64
  - name: Probability Correct_EV_1
    dtype: string
  - name: Manual Correctness Adjustment_EV_1
    dtype: string
  - name: Expert Validator_EV_1
    dtype: string
  - name: Feedback_EV_2
    dtype: string
  - name: Validator Revision Suggestion_EV_2
    dtype: string
  - name: Is First Validation_EV_2
    dtype: bool
  - name: Post hoc agreement_EV_2
    dtype: string
  - name: Sufficient Expertise?_EV_2
    dtype: string
  - name: Understand the question?_EV_2
    dtype: string
  - name: Question Difficulty_EV_2
    dtype: string
  - name: Validator Answered Correctly_EV_2
    dtype: int64
  - name: Self-reported time (minutes)_EV_2
    dtype: float64
  - name: Probability Correct_EV_2
    dtype: string
  - name: Manual Correctness Adjustment_EV_2
    dtype: 'null'
  - name: Expert Validator_EV_2
    dtype: string
  - name: Feedback_NEV_1
    dtype: string
  - name: Validator Answered Correctly_NEV_1
    dtype: int64
  - name: Explanation_NEV_1
    dtype: string
  - name: Self-reported time (minutes)_NEV_1
    dtype: float64
  - name: Websites visited_NEV_1
    dtype: string
  - name: Probability Correct_NEV_1
    dtype: string
  - name: Manual Correctness Adjustment_NEV_1
    dtype: 'null'
  - name: Non-Expert Validator_NEV_1
    dtype: string
  - name: Feedback_NEV_2
    dtype: string
  - name: Validator Answered Correctly_NEV_2
    dtype: int64
  - name: Explanation_NEV_2
    dtype: string
  - name: Self-reported time (minutes)_NEV_2
    dtype: float64
  - name: Websites visited_NEV_2
    dtype: string
  - name: Probability Correct_NEV_2
    dtype: string
  - name: Manual Correctness Adjustment_NEV_2
    dtype: 'null'
  - name: Non-Expert Validator_NEV_2
    dtype: string
  - name: Feedback_NEV_3
    dtype: string
  - name: Validator Answered Correctly_NEV_3
    dtype: float64
  - name: Explanation_NEV_3
    dtype: string
  - name: Self-reported time (minutes)_NEV_3
    dtype: float64
  - name: Websites visited_NEV_3
    dtype: string
  - name: Probability Correct_NEV_3
    dtype: string
  - name: Manual Correctness Adjustment_NEV_3
    dtype: 'null'
  - name: Non-Expert Validator_NEV_3
    dtype: string
  - name: Expert Validator Disagreement Category
    dtype: float64
  - name: Canary String
    dtype: string
  - name: correct_thought
    dtype: string
  splits:
  - name: train
    num_bytes: 4245531
    num_examples: 165
  download_size: 1924958
  dataset_size: 4245531
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
