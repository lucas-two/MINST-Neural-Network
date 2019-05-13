# NEURAL NETWORK v0.1
# Author: Lucas Geurtjens (s5132841)
# Date: 12/05/2019

import random
import math

# def init_weights(weight_list):
#     weight_list = 0.01 * random.random()
#     print(weight_list)
#
# init_weights(3)


def forward_pass(batch_no):
    """
    Performing a forward pass
    """

    forward_pass_result = []
    # FORWARD PASS (Hidden) -> (Output)
    hidden_out = []
    lower = 0
    upper = n_input

    for node in range(n_hidden):  # For each hidden layer node

        # Find its output value from the given inputs values and weights for said output node
        hidden_out.append(out(input_values[batch_no], input_weights[lower:upper], my_bias))
        lower += n_input
        upper += n_input

    forward_pass_result.append(hidden_out)

    # FORWARD PASS (Hidden) -> (Output)
    output_out = []
    lower = 0
    upper = n_hidden

    for node in range(n_output):
        output_out.append(out(hidden_out, hidden_weights[lower:upper], my_bias))
        lower += n_input
        upper += n_input

    forward_pass_result.append(output_out)

    return forward_pass_result


def net(input_lst, weight_lst, bias):
    """
    Calculate the net value of a layer node
    by looking at its connected edges and edge weights.
    """
    net_total = bias

    for node in range(len(input_lst)):
        net_total += input_lst[node] * weight_lst[node]

    return net_total


def out(input_lst, weight_lst, bias):
    """
    Calculate the sigmoid for a given node net value.
    This will be the output of the node.
    """
    return 1 / (1 + math.exp(-1 * net(input_lst, weight_lst, bias)))


def out_calc_e_total(target, fp_lst):
    """
    Calculates the e_total for weights connected from hidden to output.
    This is done using the following formula:

    ETotal = -(TargetOut - OutputOut) * OutputOut * (1 - OutputOut) * OutputOut
    """
    e_total_list = []

    for in_node in range(n_hidden):  # For each hidden layer node
        for out_node in range(n_output):  # Find ETotal for weight connected to each output node

            e_total = -1 * (target[out_node] - fp_lst[1][out_node]) * fp_lst[1][out_node] * (
                        1 - fp_lst[1][out_node]) * fp_lst[0][in_node]

            e_total_list.append(e_total)

    return e_total_list


def hidden_calc_e_total(target, fp_lst, input_vals):
    """
    Calculates the e_total for weights connected from input to hidden layer.
    This is done using the following formula:

    SumEOuts = SUM OF ALL OUTPUT NODES -> -(TargetOut - OutputOut) * OutputOut * (1 - OutputOut) * Weight(Hidden ->Output)
    ETotal = SumEOuts * HiddenOut * (1 - HiddenOut) * InputValue
    """
    e_total_lst = []
    e_sum_lst = []

    # Calculate ESum values
    weight_no = 0  # Keeping track of weight we are on
    for hidden_node in range(n_hidden):  # For each hidden layer node...
        e_sum = 0
        for output_node in range(n_output):  # For each output layer node...
            e_sum += -1 * (target[output_node] - fp_lst[1][output_node]) * fp_lst[1][output_node] * (
                    1 - fp_lst[1][output_node]) * hidden_weights[weight_no]
            weight_no += 1

        e_sum_lst.append(e_sum)

    # Calculate ETotal for each weight
    for input_node in range(n_input):
        for hidden_node in range(n_hidden):
            e_total = e_sum_lst[hidden_node] * fp_lst[1][hidden_node] * (
                    1 - fp_lst[1][hidden_node]) * input_vals[input_node]

            e_total_lst.append(e_total)

    return e_total_lst


def calc_e_final(all_et_lst, size_of_batch):
    """
    Calculate the EFinal of a given set of ETotal sets
    """
    e_final_lst = []

    for i in range(len(all_et_lst[0])):  # For each index of the Etotal list
        et_sum = 0  # Sum of ETotal values with same index
        for lst in all_et_lst:  # For each Etotal list
            et_sum += lst[i]

        e_final = (1/size_of_batch) * et_sum
        e_final_lst.append(e_final)

    return e_final_lst


def calc_new_weight(weights, e_totals, lr):
    """
    Calculate value of new weights
    """
    new_weight_lst = []

    for w in range(len(weights)):
        new_weight = weights[w] - (lr * e_totals[w])
        new_weight_lst.append(new_weight)

    return new_weight_lst


# INPUT
n_input = 2
n_hidden = 2
n_output = 2

epoch = 1
sample_size = 2
batch_size = 2
learning_rate = 0.1
my_bias = 0.1

# First four weights = Input, next four weights = hidden
input_values = [[0.1, 0.1], [0.1, 0.2]]
target_values = [[1, 0], [0, 1]]

input_weights = [0.1, 0.1, 0.2, 0.1]
hidden_weights = [0.1, 0.1, 0.1, 0.2]


for ep in range(epoch):
    for bi in range(round(sample_size / batch_size)):

        collected_e_total = []
        for batch_no in range(batch_size):

            # Forward pass
            forward_pass_list = forward_pass(batch_no)

            # Calculate ETotal
            e_totals_output = out_calc_e_total(target_values[batch_no], forward_pass_list)
            e_total_hidden = hidden_calc_e_total(target_values[batch_no], forward_pass_list, input_values[batch_no])
            combined_e_total = e_total_hidden + e_totals_output
            collected_e_total.append(combined_e_total)

        # Calculate EFinal
        e_final_list = calc_e_final(collected_e_total, batch_size)

        # Calculate new weight total
        combined_weights = input_weights + hidden_weights
        my_new_weights = calc_new_weight(combined_weights, e_final_list, learning_rate)
        for i in my_new_weights:
            print(i)

    # Predictions & Make update plot