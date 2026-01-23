#!/specific/scratches/parallel/evyataroren-2025-12-31/inter_2025/.miniconda3/envs/inter2025_vllm/bin/python
#SBATCH --job-name=qwen_inf_play
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1   
#SBATCH --cpus-per-task=1 
#SBATCH --gpus=1 --constraint='l40s|a6000|a5000|geforce_rtx_3090' 
#SBATCH --mem=32G
#SBATCH --output=logs/qwen-infr_play_%j.out
#SBATCH --error=logs/qwen-infr_play_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable{
  "meta": {
    "description": "100 simple prompts for testing LLM capabilities with reference answers",
    "total_count": 100,
    "structure": "List of objects containing 'question' and 'answer' keys"
  },
  "data": [
    { "question": "What is the capital of France?", "answer": "Paris" },
    { "question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare" },
    { "question": "What is the chemical symbol for water?", "answer": "H2O" },
    { "question": "How many continents are there on Earth?", "answer": "Seven" },
    { "question": "What is the speed of light?", "answer": "Approximately 299,792,458 meters per second (approx. 300,000 km/s)" },
    { "question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci" },
    { "question": "What is the largest mammal in the world?", "answer": "The Blue Whale" },
    { "question": "In which year did World War II end?", "answer": "1945" },
    { "question": "What is the powerhouse of the cell?", "answer": "Mitochondria" },
    { "question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong" },
    { "question": "What is 2 + 2?", "answer": "4" },
    { "question": "What is 15 multiplied by 4?", "answer": "60" },
    { "question": "If x = 5 and y = 10, what is x + y?", "answer": "15" },
    { "question": "What is the square root of 64?", "answer": "8" },
    { "question": "Is 17 a prime number?", "answer": "Yes" },
    { "question": "Solve for x: 2x - 4 = 10.", "answer": "x = 7" },
    { "question": "Which is heavier: a pound of lead or a pound of feathers?", "answer": "They weigh the same (one pound)" },
    { "question": "Complete the sequence: 2, 4, 6, 8, ...", "answer": "10" },
    { "question": "If a train travels 60 miles in one hour, how far does it go in 30 minutes?", "answer": "30 miles" },
    { "question": "What is 10% of 200?", "answer": "20" },
    { "question": "Write a haiku about a robot.", "answer": "Metal heart beats cold,\nCircuits thinking deep inside,\nLife in wires and code." },
    { "question": "Tell me a joke about a programmer.", "answer": "Why do programmers prefer dark mode? Because light attracts bugs." },
    { "question": "Write a two-sentence horror story.", "answer": "I woke up to the sound of the baby monitor. Then I remembered I don't have a baby." },
    { "question": "Compose a tweet about coffee.", "answer": "Life happens, coffee helps. ☕ #MorningVibes #CoffeeLover" },
    { "question": "Write a short poem about the rain.", "answer": "Softly falls the rain,\nWashing dust from weary streets,\nLife is green again." },
    { "question": "Create a name for a fantasy dragon.", "answer": "Ignisorth" },
    { "question": "Write an opening sentence for a mystery novel.", "answer": "The clock struck thirteen, and the manor went silent." },
    { "question": "Suggest a title for a book about time travel.", "answer": "Echoes of Tomorrow" },
    { "question": "Write a compliment for a chef.", "answer": "This dish is a masterpiece; the flavors are perfectly balanced." },
    { "question": "Describe the color blue to someone who is blind.", "answer": "Blue feels like cool water running over your hands or the refreshing chill of ice." },
    { "question": "Write a 'Hello World' program in Python.", "answer": "print(\"Hello, World!\")" },
    { "question": "How do I create a bold text in Markdown?", "answer": "**text** or __text__" },
    { "question": "What is the HTML tag for a hyperlink?", "answer": "<a>" },
    { "question": "Write a SQL query to select all users from a table named 'users'.", "answer": "SELECT * FROM users;" },
    { "question": "What does JSON stand for?", "answer": "JavaScript Object Notation" },
    { "question": "What is the difference between a list and a tuple in Python?", "answer": "Lists are mutable (changeable), while tuples are immutable (unchangeable)." },
    { "question": "How do I comment out a line in C++?", "answer": "// This is a comment" },
    { "question": "What command is used to list files in Linux?", "answer": "ls" },
    { "question": "Write a JavaScript function to add two numbers.", "answer": "function add(a, b) { return a + b; }" },
    { "question": "What port does HTTP use by default?", "answer": "80" },
    { "question": "Translate 'Hello' to Spanish.", "answer": "Hola" },
    { "question": "What is the synonym of 'happy'?", "answer": "Joyful" },
    { "question": "What is the antonym of 'fast'?", "answer": "Slow" },
    { "question": "Define the word 'serendipity'.", "answer": "The occurrence of events by chance in a happy or beneficial way." },
    { "question": "How do you spell 'receive'?", "answer": "R-E-C-E-I-V-E" },
    { "question": "Translate 'Thank you' to French.", "answer": "Merci" },
    { "question": "What is a palindrome? Give an example.", "answer": "A word or phrase that reads the same backward as forward. Example: Racecar." },
    { "question": "Use the word 'ephemeral' in a sentence.", "answer": "The beauty of the sunset was ephemeral, fading into darkness within minutes." },
    { "question": "What is the plural of 'child'?", "answer": "Children" },
    { "question": "Translate 'Good morning' to German.", "answer": "Guten Morgen" },
    { "question": "Summarize 'Goldilocks and the Three Bears' in one sentence.", "answer": "A young girl breaks into a house owned by three bears, eats their food, breaks a chair, and sleeps in a bed before escaping." },
    { "question": "Explain quantum computing to a five-year-old.", "answer": "Regular computers use light switches that are either on or off. Quantum computers use magic switches that can be on and off at the same time to solve puzzles really fast." },
    { "question": "What is the main idea of the Declaration of Independence?", "answer": "That all people have rights to life, liberty, and the pursuit of happiness, and colonies should be free from British rule." },
    { "question": "Summarize the plot of the movie 'Titanic'.", "answer": "Two people from different social classes fall in love on an ill-fated voyage aboard the Titanic, which strikes an iceberg and sinks." },
    { "question": "Explain the concept of inflation.", "answer": "Inflation is when prices for goods and services go up over time, meaning your money buys less than it used to." },
    { "question": "How does a refrigerator work?", "answer": "It uses a fluid refrigerant to absorb heat from the inside of the fridge and release it outside, keeping the interior cool." },
    { "question": "What is the difference between weather and climate?", "answer": "Weather is the short-term state of the atmosphere (rainy today), while climate is the long-term average (tropical)." },
    { "question": "Why is the sky blue?", "answer": "The atmosphere scatters sunlight in all directions, and blue light is scattered more than other colors because it travels as shorter, smaller waves." },
    { "question": "How do airplanes fly?", "answer": "Wings are shaped to make air move faster over the top than the bottom, creating lower pressure above (lift) that pulls the plane up." },
    { "question": "What is photosynthesis?", "answer": "The process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water." },
    { "question": "Name three fruits that are yellow.", "answer": "Banana, Lemon, Pineapple" },
    { "question": "List five countries in Europe.", "answer": "France, Germany, Italy, Spain, United Kingdom" },
    { "question": "What are the primary colors?", "answer": "Red, Blue, Yellow" },
    { "question": "Name three types of renewable energy.", "answer": "Solar, Wind, Hydroelectric" },
    { "question": "List the planets in our solar system.", "answer": "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune" },
    { "question": "What are the ingredients in a standard cake?", "answer": "Flour, sugar, eggs, butter, baking powder, milk" },
    { "question": "Name three famous composers.", "answer": "Mozart, Beethoven, Bach" },
    { "question": "What are the four seasons?", "answer": "Spring, Summer, Autumn (Fall), Winter" },
    { "question": "List three programming languages.", "answer": "Python, Java, C++" },
    { "question": "Name five animals that live in the ocean.", "answer": "Shark, Whale, Dolphin, Octopus, Jellyfish" },
    { "question": "Is a tomato a fruit or a vegetable?", "answer": "Botanically a fruit, culinarily treated as a vegetable." },
    { "question": "Why do we sleep?", "answer": "To restore the body and brain, consolidate memory, and regulate metabolism." },
    { "question": "Can you tickle yourself?", "answer": "Generally no, because the cerebellum predicts the sensation and cancels the response." },
    { "question": "Why do leaves change color in the fall?", "answer": "Chlorophyll breaks down, revealing other pigments like carotenoids (yellow/orange) and anthocyanins (red)." },
    { "question": "Do fish drink water?", "answer": "Saltwater fish drink water to stay hydrated; freshwater fish absorb water through their skin and gills." },
    { "question": "Why is the ocean salty?", "answer": "Rain washes mineral ions from the land into water, which flows into the ocean where evaporation leaves the salt behind." },
    { "question": "How are rainbows formed?", "answer": "Light is refracted, reflected, and dispersed by water droplets in the atmosphere, creating a spectrum of light." },
    { "question": "Why do onions make you cry?", "answer": "Cutting them releases a gas (syn-propanethial-S-oxide) that reacts with eye moisture to form mild sulfuric acid, irritating the eyes." },
    { "question": "What causes earthquakes?", "answer": "The sudden release of energy in the Earth's crust that creates seismic waves, often caused by tectonic plates shifting." },
    { "question": "Why do birds migrate?", "answer": "To find better food sources and breeding grounds as seasons change." },
    { "question": "I am currently eating breakfast. What time of day is it likely to be?", "answer": "Morning" },
    { "question": "If it is raining outside, should I wear sunglasses or take an umbrella?", "answer": "Take an umbrella" },
    { "question": "A man walks into a room with a match. He sees a lamp, a candle, and a fireplace. What does he light first?", "answer": "The match" },
    { "question": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?", "answer": "An echo" },
    { "question": "What has keys but can't open locks?", "answer": "A piano" },
    { "question": "What has to be broken before you can use it?", "answer": "An egg" },
    { "question": "What goes up but never comes down?", "answer": "Age" },
    { "question": "What gets wet while drying?", "answer": "A towel" },
    { "question": "What can you catch, but not throw?", "answer": "A cold" },
    { "question": "What has many teeth, but can't bite?", "answer": "A comb" },
    { "question": "Classify this sentiment: 'I loved the movie!' (Positive/Negative/Neutral)", "answer": "Positive" },
    { "question": "Classify this sentiment: 'The service was terrible.' (Positive/Negative/Neutral)", "answer": "Negative" },
    { "question": "Extract the date from this sentence: 'The meeting is on January 25th.'", "answer": "January 25th" },
    { "question": "Extract the email from this text: 'Contact us at support@example.com.'", "answer": "support@example.com" },
    { "question": "Correct the grammar: 'Me goes to the store.'", "answer": "I go to the store." },
    { "question": "Correct the grammar: 'She dont like apples.'", "answer": "She doesn't like apples." },
    { "question": "Rewrite this professionally: 'Hey, give me the report now.'", "answer": "Could you please provide the report at your earliest convenience?" },
    { "question": "Convert this text to uppercase: 'hello world'.", "answer": "HELLO WORLD" },
    { "question": "Remove the spaces from this string: 'Data Science'.", "answer": "DataScience" },
    { "question": "What is the next day after Friday?", "answer": "Saturday" }
  ]
}



import os
import sys



PROJECT_HOME = os.getcwd()
CACHE_BASE = os.path.join(PROJECT_HOME, ".cache")
os.makedirs(CACHE_BASE, exist_ok=True)

os.environ["PROJECT_HOME"] = PROJECT_HOME
os.environ["CACHE_BASE"] = CACHE_BASE
os.environ["HF_HOME"] = os.path.join(CACHE_BASE, "huggingface")
# os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_BASE, "huggingface", "datasets")
os.environ["VLLM_USAGE_SOURCE"] = "production"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["FLASHINFER_WORKSPACE_DIR"] = os.path.join(CACHE_BASE, "flashinfer")
os.environ["PYTHONPATH"] = PROJECT_HOME + ":" + os.environ.get("PYTHONPATH", "")

# Add PROJECT_HOME to Python path so 'play' module can be imported
sys.path.insert(0, PROJECT_HOME)




import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)



import pickle
import torch
from typing import Dict, List
from datetime import datetime
from pipeline.interface import Experiment, ModelGenerationConfig, JudgeGenerationConfig, SamplingParams, DataPoint, ActivationCapturer
from pipeline.dataset_loaders import aggregated_dataset_loader, MATH500Loader, aggregate_shuffle_strategy
from pipeline.judge_correctness import run_judge_validation
from pipeline.generate_normal import GenerateSimple
from pipeline.generate_batched import GenerateBatched

class AttentionMapCapturer(ActivationCapturer):
    """Captures attention maps from all layers and heads during model forward pass."""
    
    def __init__(self):
        super().__init__()
        self.hooks = []
        self.model = None
    
    def bind(self, model: torch.nn.Module):
        """Analyze the model and prepare to capture attention maps."""
        self.model = model
        
        # Find all layers with attention modules
        # For transformers, typically model.layers[i].self_attn or similar
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Could not find model layers for attention capture")
        
        # Initialize activation storage for each layer
        for layer_idx in range(len(layers)):
            layer_name = f"layer_{layer_idx}_attention"
            self.activations[layer_name] = []
        # Ensure model forwards return attentions by default when possible.
        try:
            if hasattr(self.model, 'config'):
                # Prefer to enable attention outputs so hooks can see weights.
                setattr(self.model.config, 'output_attentions', True)
                # Some implementations expose an attn implementation flag; prefer eager.
                try:
                    setattr(self.model.config, 'attn_implementation', 'eager')
                except Exception:
                    pass
        except Exception:
            pass
    
    def __enter__(self):
        """Register hooks to capture attention outputs."""
        if self.model is None:
            raise ValueError("Must call bind() before using context manager")
        
        # Find layers and register hooks
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Could not find model layers")
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Qwen/DeepSeek attention modules typically return a tuple where
                # output[1] are the attention weights when `output_attentions=True`.
                attention_weights = None
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights = output[1]

                # If weights are missing, raise instructive error so caller can
                # ensure `output_attentions=True` or that model config was set.
                if attention_weights is None:
                    raise ValueError(
                        f"Layer {layer_idx}: attention weights are None. "
                        "Ensure the model is called with `output_attentions=True` "
                        "or set `model.config.output_attentions = True` and use an "
                        "attn implementation that exposes weights (e.g. 'eager')."
                    )

                # Offload to CPU and detach to save GPU memory and break graph
                try:
                    cpu_attn = attention_weights.detach().cpu()
                except Exception:
                    # If not a tensor, store as-is
                    cpu_attn = attention_weights

                self.activations[f"layer_{layer_idx}_attention"].append(cpu_attn)

            return hook
        
        # Register hooks on attention modules
        for layer_idx, layer in enumerate(layers):
            # Different model architectures use different names
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            else:
                continue
            
            hook_handle = attn_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook_handle)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks after capture."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# Define picklable functions for model generation
def should_stop_fn(tokens):
    return False

def get_injection_fn(tokens):
    return ""

def global_stop_fn(tokens):
    return any(tok in ['<|endoftext|>', '<|im_end|>', '<|end_of_sentence|>'] for tok in tokens)

def question_prompt_template(q):
    return f"<|begin_of_sentence|><|User|> {q}<|Assistant|><think>"

def judge_prompt_fn(question, correct_answer, generated_text):
    return (
        f"<|begin_of_sentence|><|User|> "
        f"You are an expert math problem evaluator. Compare the model's answer with the correct answer.\n\n"
        f"Question: {question}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Model's Response: {generated_text}\n\n"
        f"Does the model's response correctly answer the question? "
        f"Respond with ONLY 'yes' or 'no' at the end of your evaluation."
        f"<|Assistant|><think>"
    )


def create_math500_experiment() -> Experiment:
    """Create an experiment for MATH-500 dataset with attention capture."""
    
    # Model generation config (no injection)
    model_config = ModelGenerationConfig(
        model_name="qwen-7B",
        model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
        should_stop_fn=should_stop_fn,
        get_injection_fn=get_injection_fn,
        global_stop_fn=global_stop_fn,
        question_prompt_template=question_prompt_template,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            take_dumb_max=False,
            max_new_tokens=1024
        ),
        dtype=torch.bfloat16
    )
    
    # Judge config
    judge_config = JudgeGenerationConfig(
        judge_name="qwen-7B-judge",
        judge_model_path="/home/ADV_2526a/evyataroren/inter_2025/models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B",
        judge_prompt=judge_prompt_fn,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_k=None,
            top_p=None,
            take_dumb_max=True,
            max_new_tokens=256,
            
        ),
        dtype=torch.bfloat16
    )
    
    # Load MATH-500 dataset
    dataset = aggregated_dataset_loader(
        datasets=[MATH500Loader],
        seed=42,
        strategy=aggregate_shuffle_strategy.SEQUENTIAL,
        base_path="/home/ADV_2526a/evyataroren/inter_2025/datasets/datasets"
    )
    
    # Create attention capturer
    attention_capturer = AttentionMapCapturer()
    
    # Create experiment
    experiment = Experiment(
        dataset=dataset,
        model_generation_config=model_config,
        judge_generation_config=judge_config,
        seed=42,
        activation_capturer=attention_capturer
    )
    
    # Populate datapoints from dataset
    experiment.populate_datapoints()
    
    # Limit to first 10 samples for testing
    experiment.datapoints = experiment.datapoints[:10]
    
    # Mark all datapoints to capture activations
    for dp in experiment.datapoints:
        dp.should_capture_activations = True
    
    return experiment


def main():
    print("=" * 80)
    print("Starting MATH-500 Experiment (No Injection, Attention Capture)")
    print("=" * 80)
    
    # Create experiment
    print("\n1. Creating experiment configuration...")
    experiment = create_math500_experiment()
    print(f"   Loaded {len(experiment.datapoints)} datapoints from MATH-500")
    
    # Run generation
    print("\n2. Running model generation (capturing attention maps)...")
    generator = GenerateBatched(experiment, device='cuda')
    generator.generate(batch_size=8)
    print("   Generation complete!")
    
    # Unload model to free memory
    generator.unload_model()
    
    # Verify activations were captured
    print("\n2a. Verifying captured activations...")
    total_activations = 0
    for dp_idx, dp in enumerate(experiment.datapoints):
        activation_types = [
            ('activations_question', dp.activations_question),
            ('activations_upto_injection', dp.activations_upto_injection),
            ('activations_injection', dp.activations_injection),
            ('activations_after_injection', dp.activations_after_injection),
        ]
        
        print(f"\n   DataPoint {dp_idx} ({dp.question_id}):")
        for activation_name, activation_dict in activation_types:
            if activation_dict is not None:
                print(f"      {activation_name}:")
                print(f"         Keys: {list(activation_dict.keys())}")
                for key, tensor_list in activation_dict.items():
                    num_tensors = len(tensor_list)
                    # Build a robust shapes summary for reporting. Handles:
                    #  - None entries
                    #  - torch.Tensor entries
                    #  - nested lists/tuples of tensors
                    shapes = []
                    try:
                        for t in tensor_list:
                            if t is None:
                                shapes.append(None)
                            elif isinstance(t, torch.Tensor):
                                shapes.append(tuple(t.shape))
                            elif isinstance(t, (list, tuple)):
                                inner_shapes = []
                                for e in t:
                                    if isinstance(e, torch.Tensor):
                                        inner_shapes.append(tuple(e.shape))
                                    else:
                                        inner_shapes.append(type(e).__name__)
                                shapes.append(inner_shapes)
                            else:
                                shapes.append(type(t).__name__)
                    except Exception as e:
                        shapes = [f"<shape-error: {e}>"]

                    # Always print the number of entries and the shapes summary
                    print(f"            {key}: {num_tensors} tensors, shapes: {shapes}")
                    total_activations += num_tensors
            else:
                print(f"      {activation_name}: None")
    
    # # Run judge validation
    # print("\n3. Running judge validation...")
    # run_judge_validation(experiment)
    
    # # Count results
    # correct = sum(1 for dp in experiment.datapoints if dp.judge_decision)
    # total = len(experiment.datapoints)
    # print(f"   Judge results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    
    # Save experiment
    output_dir = "/home/ADV_2526a/evyataroren/inter_2025/artifacts"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"math500_no_injection_{timestamp}.pkl")
    
    print(f"\n4. Saving datapoints to {output_path} (only datapoints will be pickled)...")
    # Save only the datapoints list to avoid pickling non-serializable model/code objects
    with open(output_path, 'wb') as f:
        pickle.dump(experiment.datapoints, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("   Datapoints saved successfully!")
    
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Total questions: {len(experiment.datapoints)}")
    print(f"  - Correct answers: N/A (judge validation disabled)")
    print(f"  - Total tokens with activations: {total_activations}")
    print(f"  - Datapoints pickled to: {output_path}")
    

if __name__ == "__main__":
    main()
