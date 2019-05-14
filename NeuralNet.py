# NEURAL NETWORK v0.1
# Author: Lucas Geurtjens (s5132841)
# Date: 12/05/2019

import random
import math
import numpy as np


def init_weights(no_input, no_hidden, no_output):
    """
    Create a set of random weights
    """
    weight_lst = []
    no_of_weights = (no_input * no_hidden) + (no_hidden * no_output)

    for weight in range(no_of_weights):
        weight = 0.01 * random.random()
        weight_lst.append(weight)

    return weight_lst


def forward_pass(batch_num, input_vals, input_bias):
    """
    Performing a forward pass
    """
    forward_pass_result = []

    # FORWARD PASS (Input) -> (Hidden)
    hidden_out = []
    lower = 0
    upper = n_input

    for node in range(n_hidden):  # For each hidden layer node

        # Find its output value from the given inputs values and weights for said output node
        hidden_out.append(out(input_vals[batch_num], input_weights[lower:upper], input_bias))
        lower += n_input
        upper += n_input

    forward_pass_result.append(hidden_out)

    # FORWARD PASS (Hidden) -> (Output)
    output_out = []
    lower = 0
    upper = n_hidden

    loop_counter = 0
    for node in range(n_output):
        output_out.append(out(hidden_out, hidden_weights[lower:upper], input_bias))
        lower += n_hidden
        upper += n_hidden

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
    for input_node in range(n_input):  # 784
        for hidden_node in range(n_hidden):
            e_total = e_sum_lst[hidden_node] * fp_lst[0][hidden_node] * (
                    1 - fp_lst[0][hidden_node]) * input_vals[input_node]

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


def quadratic_cost(output_out, target_out):
    """
    Use the quadratic cost function to calculate test accuracy.

    C(W,B) = FOR EACH OUTPUT NODE -> 1/2(TargetOut - OutputOut)^2
    """
    total = 0
    for node in range(len(output_out)):
        total += 0.5 * (target_out[node] - output_out[node]) ** 2

    return total


def cross_entropy_cost(output_out, target_out):
    """
    Use the cross entropy_cost function to calculate test accuracy

    C (W, B) = 1/n -> FOR EACH OUTPUT NODE (Y1 - Y1 ln(OutputOut) - (1 - Y1) ln(1 - OutputOut)
    """
    total = 0
    for node in range(len(output_out)):
        total += target_out[node] - target_out[node] * np.log(output_out[node]) - (1 - target_out[node]) * np.log(1 - output_out[node])

    total = 1 / total

    return total




# INPUT
n_input = 784
n_hidden = 30
n_output = 10
training_set_f = "TrainDigitX.csv.gz"
training_target_f = "TrainDigitY.csv.gz"

# Hyper Parameters
epoch = 30
sample_size = 50000
batch_size = 20  # Mini batch
learning_rate = 3
my_bias = 0.01

# Grab the training target values
training_target = np.loadtxt(training_target_f, dtype=int, delimiter=',')

target_values = []
# Convert the training target values into a matrix
for i in training_target:
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    data[i] = 1
    target_values.append(data)


# Grab the training input values
input_values = np.loadtxt(training_set_f, dtype=float, delimiter=',')

# Shuffle the training data
shuffled_set = list(zip(target_values, input_values))
random.shuffle(shuffled_set)
target_values, input_values = zip(*shuffled_set)
target_values = list(target_values)
input_values = list(input_values)

# testset  # Get file
# testset_predictions = 0  # Save file

# First four weights = Input, next four weights = hidden
# input_values = [[0.1, 0.1], [0.1, 0.2]]
# target_values = [[1, 0], [0, 1]]
# input_weights = [0.1, 0.1, 0.2, 0.1]
# hidden_weights = [0.1, 0.1, 0.1, 0.2]

# Initialise the weights as random small float values, then split them into groups
initial_weights = init_weights(n_input, n_hidden, n_output)
input_weights = initial_weights[0:n_input * n_hidden]
hidden_weights = initial_weights[n_input * n_hidden: n_input * n_hidden + n_hidden * n_output]

for ep in range(epoch):
    for bi in range(round(sample_size / batch_size)):

        collected_e_total = []
        for batch_no in range(batch_size):

            # Forward pass
            forward_pass_list = forward_pass(batch_no, input_values, my_bias)

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

        # Split weights up again
        input_weights = my_new_weights[0:len(input_weights)]
        hidden_weights = my_new_weights[len(input_weights): len(input_weights) + len(hidden_weights)]
        print("Successfully completed batch %s of %s" % (bi + 1, round((sample_size/batch_size))))

    print("Successfully completed epoch %s of %s" % (ep + 1, epoch))

    # Predictions & Make update plot
    # fp = forward_pass(0, test_set, my_bias)  # Forward pass
    # f_out = fp[0]  # Grab the output results of the forward pass
    # accuracy = quadratic_cost(fp[0],  target_test_set)
    # accuracy_2 = cross_entropy_cost(fp[0],  target_test_set)
    # print(accuracy, accuracy_2)

