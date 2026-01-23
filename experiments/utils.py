from matplotlib import cm, colors
import matplotlib.pyplot as plt

def sum_layer_heads(layer):
    # activations shape: (batch_size, num_heads, seq_len, seq_len)
    sum_matrix = layer[0, 0, :, :].clone().zero_()
    num_heads = layer.shape[1]
    for idx in range(num_heads):
        sum_matrix += layer[0, idx, :, :]
    return sum_matrix

def attention_per_layer_as_a_graph(curr_sample, x_after_inj=0):
    # Plot a line for each layer index, each with a different color and legend
    num_layers = len([k for k in curr_sample.activations_after_injection.keys() if k.startswith('layer_') and k.endswith('_attention')])

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    plt.figure(figsize=(30, 8))
    colors = cm.get_cmap('viridis', num_layers)
    for index in range(num_layers):
        layer = curr_sample.activations_after_injection[f"layer_{index}_attention"][x_after_inj]
        summed_matrix = sum_layer_heads(layer)
        col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
        plt.plot(range(len(col_sums)), col_sums, marker='o', label=f'Layer {index}', color=colors(index))

    xticks = xticks[:len(col_sums) - 1]
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=90)
    plt.xlabel('Token')
    plt.ylabel('Sum over rows')
    plt.title('Sum over rows for each token (x-axis) per layer')
    plt.legend()
    plt.tight_layout()
    plt.show()


def attention_to_word_per_layer_as_a_graph(curr_sample, index_to_check, sum_layers=False):
    # Plot a line for each layer index, each with a different color and legend
    num_layers = len([k for k in curr_sample.activations_after_injection.keys() if k.startswith('layer_') and k.endswith('_attention')])

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    after_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)
    plt.figure(figsize=(30, 8))

    colors = cm.get_cmap('viridis', num_layers)
    layers_plots = []
    for layer_idx in range(num_layers):
        attendence = []
        for index in range(len(curr_sample.after_injection_tokens)):
            layer = curr_sample.activations_after_injection[f"layer_{layer_idx}_attention"][index]
            summed_matrix = sum_layer_heads(layer)
            col_sums = summed_matrix.float().sum(dim=0).cpu().numpy()
            attendence.append(col_sums[index_to_check])
        layers_plots.append(attendence)
        if not sum_layers:
            plt.plot(range(after_injection_idx, after_injection_idx+len(attendence)), attendence, marker='o', label=f'Layer {layer_idx}', color=colors(layer_idx))

    if sum_layers:
        summed_across_layers = [sum(x) for x in zip(*layers_plots)]
        plt.plot(range(after_injection_idx, after_injection_idx+len(summed_across_layers)), summed_across_layers, marker='o', label='Sum over all layers')
    xticks = xticks[:len(col_sums) - 1]
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=90)
    plt.xlabel('Token')
    plt.ylabel('Sum over rows')
    plt.title('Sum over rows for each token (x-axis) per layer, attendence to word index ' + str(index_to_check))
    plt.legend()
    plt.tight_layout()
    plt.show()


def attention_to_words_per_layer_as_a_graph(curr_sample, words_indices,layers=range(28), sum_layers=False):
    # Plot a line for each layer index, each with a different color and legend
    num_layers = len([k for k in curr_sample.activations_after_injection.keys() if k.startswith('layer_') and k.endswith('_attention')])

    xticks = curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized + curr_sample.after_injection_tokens
    after_injection_idx = len(curr_sample.question_formatted_contents_tokenized + curr_sample.upto_injection_tokens + curr_sample.injection_tokenized)
    plt.figure(figsize=(30, 8))

    colors = cm.get_cmap('viridis', num_layers)
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
            plt.plot(range(after_injection_idx, after_injection_idx+len(attendence)), attendence, marker='o', label=f'Layer {layer_idx}', color=colors(layer_idx))

    if sum_layers:
        summed_across_layers = [sum(x) for x in zip(*layers_plots)]
        plt.plot(range(after_injection_idx, after_injection_idx+len(summed_across_layers)), summed_across_layers, marker='o', label='Sum over all layers')
    xticks = xticks[:len(col_sums) - 1]
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=90)
    plt.xlabel('Token')
    plt.ylabel('Sum over rows')
    plt.title('Sum over rows for each token (x-axis) per layer, attendence to word index ' + str(words_indices))
    plt.legend()
    plt.tight_layout()
    plt.show()


def sample_to_text(sample):
    return ''.join(sample.question_formatted_contents_tokenized + sample.upto_injection_tokens + sample.injection_tokenized + sample.after_injection_tokens)

def get_text_after_injection(sample):
    all_text = sample_to_text(sample)
    return all_text[len(''.join(sample.question_formatted_contents_tokenized + sample.upto_injection_tokens + sample.injection_tokenized)):]
