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


def out_calc_e_total():
    return
    # for output -> hidden
    # Etotal = -(Target[out_node] - Out[out_node) * Out[out_node] * (1 - Out[out_node]) * Out[hidden_node] (For hidden layers)



    return

def hidden_calc_e_total():
    return
# for hidden -> input
# For all output nodes...
# sum E_Outs += -(Target[out_node] - Out[out_node) * Out[out_node] * (1 - Out[out_node]) * weight
# Etotal = sum E_Outs * Out[hidden] * (1 - Out[hidden]) * Out[input value]

def calc_e_final():
    return

def calc_new_weight():
    return


print(forward_pass(0))

#
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