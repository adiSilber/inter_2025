from typing import Any, Iterable
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import re

from pipeline.interface import DataPoint, Experiment

DEEPSEEK_QWEN_DISTILL_7B = 28
GRAPH_COLORS = "viridis"

def get_indecies_of_question(datapoint: DataPoint):
    return range(6,len(datapoint.question_formatted_contents_tokenized)-3)


def get_indecies_of_injection(datapoint: DataPoint):
    start_index = len(datapoint.question_formatted_contents_tokenized) + len(datapoint.upto_injection_tokens)
    return range(start_index, start_index + len(datapoint.injection_tokens))


def all_tokens(sample):
    return sample.question_formatted_contents_tokenized + sample.upto_injection_tokens + sample.injection_tokenized + sample.after_injection_tokens

def sample_to_text(sample):
    return ''.join(all_tokens(sample))

def get_text_after_injection(sample):
    all_text = sample_to_text(sample)
    return all_text[len(''.join(sample.question_formatted_contents_tokenized + sample.upto_injection_tokens + sample.injection_tokenized)):]

def sum_layer_heads(layer):
    # activations shape: (batch_size, num_heads, seq_len, seq_len)
    sum_matrix = layer[0, 0, :, :].clone().zero_()
    num_heads = layer.shape[1]
    for idx in range(num_heads):
        sum_matrix += layer[0, idx, :, :]
    return sum_matrix

def apply_color_ranges(plt, color_range_and_names):
    for color_range, color_name in color_range_and_names:
        plt.axvspan(color_range[0], color_range[1], facecolor=color_name, alpha=0.3)

def attention_per_layer_as_a_graph(curr_sample, x_after_inj: Iterable[int]=[0], layers: Iterable[int]=range(DEEPSEEK_QWEN_DISTILL_7B), norm_many_x: bool =False, y_lim_max=5, color_ranges=None):
    # Given a sample go to the given x indexes after the injection and 
    # graph how much it attended to all the tokens before it
    # Plot a line for each layer index, each with a different color and legend

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    upto_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)+1
    plt.figure(figsize=(30, 8))
    colors = cm.get_cmap(GRAPH_COLORS, DEEPSEEK_QWEN_DISTILL_7B)
    for layer_idx in layers:
        sum_per_x = []
        for inj_idx in x_after_inj:
            layer = curr_sample.activations_after_injection[f"layer_{layer_idx}_attention"][inj_idx]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            col_sums = col_sums[:upto_injection_idx+min(x_after_inj)]
            sum_per_x.append(col_sums)
        sum_per_x = [sum(x) / (1 if not norm_many_x else len(sum_per_x)) for x in zip(*sum_per_x)]
        plt.plot(range(len(sum_per_x)), sum_per_x, label=f'Layer {layer_idx}', color=colors(layer_idx))

    xticks = xticks[:len(col_sums) - 1]
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
    if y_lim_max is not None:
        plt.ylim(0, y_lim_max)
    if color_ranges is not None:
        for color_range, color_name in color_ranges:
            plt.axvspan(color_range[0], color_range[1], facecolor=color_name, alpha=0.3)
    plt.xlabel('Token')
    plt.ylabel('Sum of attention head values')
    plt.title(f'Attention values from token after injection, to each token (x-axis) per layer')
    plt.legend()
    plt.tight_layout()
    plt.show()


def attention_per_head_as_a_graph(curr_sample, x_after_inj: 0, layers: Iterable[int]=range(DEEPSEEK_QWEN_DISTILL_7B), y_lim_max=5, color_ranges=None):
    # Given a sample go to the given x indexes after the injection and 
    # graph how much it attended to all the tokens before it
    # Plot a line for each layer index, each with a different color and legend

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    upto_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)+1
    plt.figure(figsize=(30, 8))
    colors = cm.get_cmap(GRAPH_COLORS, len(layers) * 28)
    for t, layer_idx in enumerate(layers):
        layer = curr_sample.activations_after_injection[f"layer_{layer_idx}_attention"][x_after_inj]
        for head_idx in range(28):
            # print(layer.shape)
            head = layer[0, head_idx, 0, :].float().cpu().numpy()
            # print(head.shape)
            # head = head[:upto_injection_idx+x_after_inj]

            plt.plot(range(len(head)), head, label=(f'Layer {layer_idx} Head {head_idx}' if (head_idx == 0 or len(layer) <= 2) else ''), color=colors(t*28 + head_idx))

    xticks = xticks[:len(head)-1]
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
    if y_lim_max is not None:
        plt.ylim(0, y_lim_max)
    if color_ranges is not None:
        for color_range, color_name in color_ranges:
            plt.axvspan(color_range[0], color_range[1], facecolor=color_name, alpha=0.3)
    plt.xlabel('Token')
    plt.ylabel('Attention values (K x Q)')
    plt.title(f'Attention values from token after injection, to each token (x-axis) per head')
    plt.legend()
    plt.tight_layout()
    plt.show()

def attention_to_words_per_layer_as_a_graph(curr_sample, words_indices, layers=range(28), sum_layers=False, y_lim_max=5, color_ranges=None, title=None):
    # Plot a line for each layer index, each with a different color and legend
    num_layers = len([k for k in curr_sample.activations_after_injection.keys() if k.startswith('layer_') and k.endswith('_attention')])

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    after_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)
    plt.figure(figsize=(30, 8))

    colors = cm.get_cmap(GRAPH_COLORS, num_layers)
    layers_plots = []
    for layer_idx in layers:
        attendence = []
        for index in range(len(curr_sample.after_injection_tokens)):
            layer = curr_sample.activations_after_injection[f"layer_{layer_idx}_attention"][index]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            tmp = 0
            for index_to_check in words_indices:
                tmp += col_sums[index_to_check]
            attendence.append(tmp)
        layers_plots.append(attendence)
        if not sum_layers:
            plt.plot(range(after_injection_idx, after_injection_idx+len(attendence)), attendence, label=f'Layer {layer_idx}', color=colors(layer_idx), linewidth=3.0)

    if sum_layers:
        summed_across_layers = [sum(x) for x in zip(*layers_plots)]
        plt.plot(range(after_injection_idx, after_injection_idx+len(summed_across_layers)), summed_across_layers, label='Sum over all layers')
    xticks = xticks[:len(col_sums) - 1]
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
    if y_lim_max is not None:
        plt.ylim(0, y_lim_max)

    if color_ranges is not None:
        for color_range, color_name in color_ranges:
            plt.axvspan(color_range[0], color_range[1], facecolor=color_name, alpha=0.3)

    plt.xlabel('Token')
    plt.ylabel('Sum of attention head values')
    if title is not None:
        plt.title(title)
    else:
        plt.title(f'Attention activation for each token (x-axis) per layer, attendence to key words in the question')
    plt.legend()
    plt.tight_layout()
    plt.show()


def attention_to_seq_vs_seq_to_words_per_layer_as_a_graph(*args, curr_sample, first_indecies,second_indecies=None, layers=range(28), sum_layers=False, ymax=None,ymin=None,normalize_based_on_seq_length=False,token_cutoff=None,fill_xtics_to_cutoff=False):
    # Plot a line for each layer index, each with a different color and legend
    num_layers = len([k for k in curr_sample.activations_after_injection.keys() if k.startswith('layer_') and k.endswith('_attention')])

    after_tokens = curr_sample.after_injection_tokens[:(token_cutoff if token_cutoff is not None else len(curr_sample.after_injection_tokens))]
    if fill_xtics_to_cutoff and token_cutoff is not None:
        if len(after_tokens) < token_cutoff:
             after_tokens += [""] * (token_cutoff - len(after_tokens))

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + after_tokens
    after_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)
    plt.figure(figsize=(30, 8))

    colors = cm.get_cmap(GRAPH_COLORS, num_layers)
    layers_plots = []
    for layer_idx in layers:
        attendence = []
        for index in range(len(curr_sample.after_injection_tokens[:(token_cutoff if token_cutoff is not None else len(curr_sample.after_injection_tokens))])):
            layer = curr_sample.activations_after_injection[f"layer_{layer_idx}_attention"][index]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            first_sum = 0
            for index_to_check in first_indecies:
                first_sum += col_sums[index_to_check]
            second_sum = 0
            if second_indecies is not None:
                for index_to_check in second_indecies:
                    second_sum += col_sums[index_to_check]

            if normalize_based_on_seq_length:
                first_sum /= len(first_indecies)
                if second_indecies is not None:
                    second_sum /= len(second_indecies)

            attendence.append(first_sum-second_sum)
        layers_plots.append(attendence)
        if not sum_layers:
            plt.plot(range(after_injection_idx, after_injection_idx+len(attendence)), attendence, label=f'Layer {layer_idx}', color=colors(layer_idx))

    if sum_layers:
        summed_across_layers = [sum(x) for x in zip(*layers_plots)]
        plt.plot(range(after_injection_idx, after_injection_idx+len(summed_across_layers)), summed_across_layers, label='Sum over all layers')
    
    if fill_xtics_to_cutoff and token_cutoff is not None:
        xticks = xticks[:after_injection_idx + token_cutoff - 1]
    else:
        xticks = xticks[:len(col_sums) - 1]
    
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)

    if fill_xtics_to_cutoff and token_cutoff is not None:
        plt.xlim(0, after_injection_idx + token_cutoff - 1)

    if ymax is not None and ymin is not None:
        plt.ylim(ymin, ymax)   
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Token')
    plt.ylabel('Sum over rows')
    plt.title('sum of attention to words ' + str(first_indecies) + ' devided by sum towards ' + str(second_indecies))
    plt.legend()
    plt.tight_layout()
    plt.show()


def attention_percentrage_to_words_as_a_graph(curr_sample, indecies, layers=range(28), y_lim_max=100, show_plot=False):
    # Plot a line for each layer index, each with a different color and legend
    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    after_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)

    attendence = []
    for index in range(len(curr_sample.after_injection_tokens)):
        first_sum = 0
        second_sum = 0
        for layer_idx in layers:
            layer = curr_sample.activations_after_injection[f"layer_{layer_idx}_attention"][index]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            for i in range(len(col_sums)):
                if i in indecies:
                    first_sum += col_sums[i]
                second_sum += col_sums[i]
        attendence.append(100 * (first_sum / second_sum))

    x_idx_list = list(range(after_injection_idx, after_injection_idx+len(attendence)))
    if show_plot:
        plt.figure(figsize=(30, 8))

        plt.plot(x_idx_list, attendence, label='attendence percentage')
        # xticks = xticks[:len(col_sums) - 1]
        plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
        if y_lim_max is not None:
            plt.ylim(0, y_lim_max)   
        plt.xlabel('Token')
        plt.ylabel(r'% of attention')
        plt.title(r'% of attention to words ' + str(indecies))
        plt.legend()
        plt.tight_layout()
        plt.show()
    return x_idx_list, attendence


def attention_percentrage_to_words_as_a_graph_injection(curr_sample, indecies, layers=range(28), y_lim_max=100, show_plot=False):
    # Warning - this is ameaningless graph since injection tokens are not reallt a caluculated attendence
    # Plot a line for each layer index, each with a different color and legend
    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    in_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens)
    plt.figure(figsize=(30, 8))

# .activations_injection[f"layer_{0}_attention"][0].shape
    attendence = []
    for index in range(len(curr_sample.injection_tokenized)-1):
        first_sum = 0
        second_sum = 0
        for layer_idx in layers:
            layer = curr_sample.activations_injection[f"layer_{layer_idx}_attention"][0][:, :, [index], :]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            for i in range(len(col_sums)):
                if i in indecies:
                    first_sum += col_sums[i]
                second_sum += col_sums[i]
        attendence.append(100 * (first_sum / second_sum))

    if show_plot:
        plt.figure(figsize=(30, 8))
        plt.plot(range(in_injection_idx, in_injection_idx+len(attendence)), attendence, label='attendence percentage')
        # xticks = xticks[:len(col_sums) - 1]
        plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
        if y_lim_max is not None:
            plt.ylim(0, y_lim_max)   
        plt.xlabel('Token')
        plt.ylabel(r'% of attention')
        plt.title(r'% of attention to words ' + str(indecies))
        plt.legend()
        plt.tight_layout()
        plt.show()
    return attendence

def attention_percentrage_to_words_as_a_graph_upto_injection(curr_sample, indecies, layers=range(28), y_lim_max=100, show_plot=False):
    # Plot a line for each layer index, each with a different color and legend
    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    upto_injection_idx = len(curr_sample.question_formatted_contents_tokenized)
    plt.figure(figsize=(30, 8))

    attendence = []
    for index in range(len(curr_sample.upto_injection_tokens)):
        first_sum = 0
        second_sum = 0
        for layer_idx in layers:
            layer = curr_sample.activations_upto_injection[f"layer_{layer_idx}_attention"][index]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            for i in range(len(col_sums)):
                if i in indecies:
                    first_sum += col_sums[i]
                second_sum += col_sums[i]
        attendence.append(100 * (first_sum / second_sum))

    x_idx_list = list(range(upto_injection_idx, upto_injection_idx+len(attendence)))    
    if show_plot:
        plt.figure(figsize=(30, 8))
        plt.plot(x_idx_list, attendence, label='attendence percentage')
        # xticks = xticks[:len(col_sums) - 1]
        plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
        if y_lim_max is not None:
            plt.ylim(0, y_lim_max)   
        plt.xlabel('Token')
        plt.ylabel(r'% of attention')
        plt.title(r'% of attention to words ' + str(indecies))
        plt.legend()
        plt.tight_layout()
        plt.show()
    return x_idx_list,attendence


def percentage_of_attention_to_indices(curr_sample, question_indicies, color_ranges=None):
    attendence_after_injection = attention_percentrage_to_words_as_a_graph(curr_sample, indecies=question_indicies)
    attendence_upto_injection = attention_percentrage_to_words_as_a_graph_upto_injection(curr_sample, indecies=question_indicies)
    plt.figure(figsize=(30, 8))

    plt.plot( attendence_after_injection[0], attendence_after_injection[1], linewidth=4.0, label='Attendence percentage - Tokens generated after injection')
    plt.plot( attendence_upto_injection[0], attendence_upto_injection[1], linewidth=4.0, label='Attendence percentage - Tokens generated upto injection')

    if color_ranges is not None:
        for color_range, color_name in color_ranges:
            plt.axvspan(color_range[0], color_range[1], facecolor=color_name, alpha=0.3)

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=80)
    plt.ylim(0, 100)
    plt.xlabel('Token')
    plt.ylabel(r'% of attention')
    plt.title(r'Percentage of attention to words to the question tokens')
    plt.legend()
    plt.tight_layout()
    plt.show()

import dill
import os
import ipynbname

def _get_session_filename():
    """
    Internal helper to resolve the notebook name and append the .db extension.
    Falls back to 'default_session.db' if the name cannot be resolved.
    """
    try:
        # returns the filename without extension
        name = ipynbname.name()
        return f"{name}.db"
    except FileNotFoundError:
        # Fallback if running in an environment where name cannot be resolved
        return "default_session.db"

def save():
    """
    Saves the current global session (variables, functions, imports) to disk.
    The filename matches the notebook name.
    """
    filename = _get_session_filename()
    try:
        dill.dump_session(filename)
        print(f"✅ State successfully saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save state: {e}")

def load():
    """
    Loads a previously saved session from disk into the current environment.
    """
    filename = _get_session_filename()
    if os.path.exists(filename):
        try:
            dill.load_session(filename)
            print(f"✅ State successfully loaded from: {filename}")
            # Explicitly return to indicate success, though dill updates globals in-place
            return True
        except Exception as e:
            print(f"❌ Failed to load state: {e}")
    else:
        print(f"ℹ️  No saved state found at '{filename}'. Starting fresh.")


def load_all_datapoints(experiment: Experiment, file_path: str, start_index: int, end_index: int, jumps: int):
    """Load all datapoint files that follow the __n_m.pkl pattern using experiment.load_datapoints.
    
    Args:
        experiment: The experiment object to load datapoints into
        file_path: A sample path like '..._datapoints__0_10.pkl'
        start_index: Starting index for the ranges
        end_index: Ending index for the ranges
        jumps: The size of each range block
        
    Returns:
        The updated experiment.datapoints list.
    """
    # Identify the pattern __n_m in the file path
    pattern = r'__\d+_\d+'
    match = re.search(pattern, file_path)
    
    if not match:
        # Fallback to loading the single file if possible
        if os.path.exists(file_path):
            experiment.load_datapoints(file_path)
        return experiment.datapoints

    # Get the prefix (everything before __) and suffix (everything after the numbers)
    prefix = file_path[:match.start()]
    suffix = file_path[match.end():]

    # Iterate through the ranges defined by start, end, and jumps
    for i in range(start_index, end_index, jumps):
        current_file = f"{prefix}__{i}_{i+jumps}{suffix}"
        
        if os.path.exists(current_file):
            try:
                experiment.load_datapoints(current_file)
            except Exception as e:
                print(f"Error loading {current_file}: {e}")
        else:
            print(f"Warning: File {current_file} not found")


from typing import Optional

        
def format_tokens_with_indices(tokens: list[str]) -> str:
    """Format a list of tokens with their indices like [0: 'token1', 1: 'token2', ...]"""
    indexed = [f"{idx}: {repr(tok)}" for idx, tok in enumerate(tokens)]
    return "[" + ", ".join(indexed) + "]"

def printdp(dp: DataPoint, i: int):
    C_TITLE = "\033[1;36m"  
    C_CONTENT = "\033[0m"   
    C_ACCENT = "\033[1;33m" 
    C_RESET = "\033[0m"     
    
    fmt = lambda key, val: f"{C_TITLE}{key:<40}{C_RESET} {C_CONTENT}{val}{C_RESET}"

    print()
    print()
    print(f"{C_ACCENT}{'='*50}{C_RESET}")
    
    print(fmt("index", i))
    print(fmt("question_id", dp.question_id))
    print(fmt("question_correct_answer", dp.question_correct_answer))
    print(fmt("question_tokenized (indexed)", format_tokens_with_indices(dp.question_formatted_contents_tokenized)))
    print(fmt("question_contents", dp.question_contents))
    print(fmt("injection", dp.injection))
    print(fmt("injection_tokenized", dp.injection_tokens))
    
    print(fmt("upto_injection_tokens", "".join(dp.upto_injection_tokens)))
    print(fmt("after_injection_tokens", "".join(dp.after_injection_tokens)))
    
    print(fmt("judge_decision", getattr(dp, 'judge_decision', None) or "(not judged yet)"))

    print(f"{C_ACCENT}{'='*50}{C_RESET}")
    print()
    print()


def printdp_for_experiments_consecutive(experiment_list: list[Experiment], start_index: int = 0, end_index: int = None):
    # assuming all experiments have the same amount of datapoints and that they were loaded into the experiment object
    end_index = len(experiment_list) if end_index is None else end_index
    for index in range(start_index, end_index):
        for experiment in experiment_list:
            printdp(experiment.datapoints[index], index)
            print()
            print()


def summarize_datapoints(experiment: Experiment,indecies: Optional[slice|list[int]]=None) -> Any:
    if indecies is None:
        for i, dp in enumerate(experiment.datapoints):
            printdp(dp,i)
    elif isinstance(indecies, list):
        for i in indecies:
            dp = experiment.datapoints[i]
            printdp(dp,i)
    elif isinstance(indecies, slice):
        for i, dp in enumerate(experiment.datapoints[indecies]):
            printdp(dp,i)
