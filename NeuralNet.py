# NEURAL NETWORK v0.1
# Author: Lucas Geurtjens (s5132841)
# Date: 12/05/2019

import random
import math

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

    SumEOuts = SUM OF ALL OUTPUT NODES -> -(TargetOut - OutputOut) * OutputOut * (1 - OutputOut) * Weight
    ETotal = SumEOuts * HiddenOut * (1 - HiddenOut) * InputValue
    """

    # NOTE:
    # Brain has lost ability to think.
    # Will come back to solving this tomorrow.

    # e_total_list = []
    # e_sum_values = []

    # for in_node in range(n_input):
    #     e_out_sum = 0
    #     for out_node in range(n_output):
    #         e_out_sum += -1 * (target[out_node] - fp_lst[1][out_node]) * fp_lst[1][out_node] * (
    #                 1 - fp_lst[1][out_node]) * input_weights[in_node + out_node]
    #
    #     for hidden_node in range(n_hidden):
    #
    # for weight in range(len(input_weights)):
    #     for in_node in range(n_input):
    #         e_out_sum = 0
    #         for out_node in range(n_output):
    #             e_out_sum += -1 * (target[out_node] - fp_lst[1][out_node]) * fp_lst[1][out_node] * (
    #                     1 - fp_lst[1][out_node]) * input_weights[num]
    #             print("Weight Used:", input_weights[num])


    #
    #
    # for num in range(input_weights):  # For all weights
    #
    #     e_out_sum = 0
    #
    #     for out_node in range(n_output):  # For all output nodes
    #         # Find the sum of outs
    #         e_out_sum += -1 * (target[out_node] - fp_lst[1][out_node]) * fp_lst[1][out_node] * (
    #                 1 - fp_lst[1][out_node]) * input_weights[num]
    #         print("Weight Used:", input_weights[num])
    #
    #     for in_node in range(n_input):
    #         e_total = e_out_sum * fp_lst[0][in_node] * (1 - fp_lst[0][in_node]) * input_vals[in_node]
    #         e_total_list.append(e_total)
    #
    # return e_total_list


def calc_e_final():
    return


def calc_new_weight():
    return



batch_no = 0

forward_pass_list = forward_pass(batch_no)
e_totals_output = out_calc_e_total(target_values[batch_no], forward_pass_list)
e_total_hidden = hidden_calc_e_total(target_values[batch_no], forward_pass_list, input_values[batch_no])
print(e_total_hidden)

# For all output nodes...
# sum E_Outs += -(Target[out_node] - Out[out_node) * Out[out_node] * (1 - Out[out_node]) * weight
# Etotal = sum E_Outs * Out[hidden] * (1 - Out[hidden]) * Out[input value]





    # e_out_sum += -1 * (target_values[batch_no] - )
    # e_out_sum += -1 * Target[]

# for _ in range(epoch):
#     for _ in range(sample_size / batch_size):
#
#         # Forward pass ->
#
#         for _ in range(batch_size):
#             # Calculate Etotal for Weight
#
#         # Calculate Efinal for Weight
#
#         # Calculate new weight total
#
#     # Predictions & Make update plot