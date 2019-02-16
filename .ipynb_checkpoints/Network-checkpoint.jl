using Base, Distributions, LinearAlgebra

# A struct which holds a layer of a neural network.
# input: Defines the number of input neurons (n).
# output: Defines the number of output neurons (m).
# weights: A nxm matrix which is used to transform a n-dimensional input vector to a m-dimensional output vector.
# nonlinearity: A function which is applied to the transformed vector. Default is the rectified linear unit (max(0, x)).
#               Any function f: n -> n can be passed as an argument.
mutable struct Layer
    input::Int
    output::Int
    weights::Array{Float64, 2}
    nonlinearity::Function
    bias::Array
    function Layer(input::Int, output::Int, nonlinearity::Function=sigm)
        bias = 0.2 * randn(output)'
        weights = xavier_initialization(input, output)
        new(input, output, weights, nonlinearity, bias)
    end
end

# Builds a Feed-Forward Network given a list of Weights.
# input: Dimension of the input layer ('zeroth' layer).
# output: Dimension of the output layer ('final' layer).
mutable struct NeuralNetwork
    layers::Array{Layer, 1}
end

# Holds a vector which serves as the bias-update and a matrix which is used to update the weights.
# input_size: Number of neurons of the current layer.
# output_size: Number of neurons of the next layer.
mutable struct LayerUpdate
    bias_update::Array{Float64, 1}
    weight_update::Array{Float64, 2}
    function LayerUpdate(input_size::Int, output_size::Int)
        bias_update = zeros(input_size)
        weight_update = zeros(input_size, output_size)
        new(bias_update, weight_update)
    end
end

# Holds the accumulated weight changes for each minibatch.
# network: The network which is being trained.
mutable struct MiniBatchUpdate
    num_layers::Int
    updates::Array{LayerUpdate, 1}
    function MiniBatchUpdate(network::NeuralNetwork)
        updates = []
        for layer in network.layers
            layer_update = LayerUpdate(layer.input, layer.output)
            push!(updates, layer_update)
        end
        new(size(network.layers)[1], updates)
    end
end

# Initializes the weights of a layer with the Xavier method such that the activity neither explodes nor fades.
# input_size: Dimension of the input layer.
# output_size: Dimension of the output layer.
function xavier_initialization(input_size::Int, output_size::Int, μ::Float64=0.0)
    σ = √(1 / input_size)
    gaussian = Normal(μ, σ)
    rand(gaussian, input_size, output_size)
end

# Activation functions
relu(input) = map(x -> max(0, x), input)
sigm(input) = map(x -> 1 / (1 + exp(-x)), input)
soft(input) = map(x -> exp(x) / sum(exp.(input)), input)
unit(input) = map(x -> x, input)
# Derivatives of activation functions
d_relu(input) = map(x -> x > 0 ? 1 : 0, input)
d_sigm(input) = map(x -> sigm(x)*(1 - sigm(x)), input)
# Loss functions
square(input) = map(x -> x*x, input)
# Returns the loss for a given output, i.e. the difference between the target-output and the obtained output.
loss(output::Array{Float64, 1}, target::Array{Float64, 1}, loss_function::Function=square) = 0.5sum(loss_function(output - target))

# Feeds an input vector through a network in order to obtain an output vector.
# input: The input vector which is fed through the network.
# network: The network which is used to obtain the output.
function feed_forward(input::Array{Float64, 1}, network::NeuralNetwork)
    input = input'
    for layer in network.layers
        nonlin = layer.nonlinearity
        input = nonlin(*(input, layer.weights) + layer.bias)
    end

    vec(input)
end

# Feeds an input vector through a network in order to obtain an output vector. As the input is fed, the activity
# of each layer is saved. This saved history is then used to perform the error-backpropagation needed to train the network.
# input: The input vector which is fed through the network.
# network: The network which is used to obtain the output.
function feed_forward_full_activation(input::Array{Float64, 1}, network::NeuralNetwork)
    input = input'
    activations = []
    push!(activations, input)
    for layer in network.layers
        nonlin = layer.nonlinearity
        input = nonlin(*(input, layer.weights) + layer.bias)
        push!(activations, input)
    end

    return activations
end

# Back propagates the error for each layer. This error is used to adjust the weights of the network later on ('training').
# input: An input vector
# label: A target vector
# activations: The activation of each layer for the current input. Backprop checks how much each 'activation' contributed
#              to the total loss.
# network: The network which is being trained.
function propagate_errors(input::Array{Float64, 1}, label::Array{<:Float64, 1}, activations, network::NeuralNetwork)
    loss = vec(activations[end]) - label
    errors = [loss]
    for (id, layer) in enumerate(reverse(network.layers))
        activation = activations[end - id]
        error = (layer.weights * errors[id]) .* vec(d_sigm(activation))
        push!(errors, error)
    end

    return reverse(errors)
end

# Calculates the total loss for a set of inputs and labels. For stochastic gradient descent, these sets should be random.
# Based on the loss of a batch a vector (matrix respectively) is calculated which is then used to update the bias (weights).
# input_batch: An array of input vectors
# input_labels: An array of the respective labels
# network: The network which is being trained.
function execute_minibatch(input_batch, input_labels, network::NeuralNetwork)
    batch_update = MiniBatchUpdate(network)
    for i in 1:size(input_batch)[1]
        activations = feed_forward_full_activation(input_batch[i], network)
        error = propagate_errors(input_batch[i], input_labels[i], activations, network)
        for (id_layer, layer) in enumerate(batch_update.updates)
            layer.weight_update += .*(activations[id_layer], error[id_layer + 1])'
            layer.bias_update += error[id_layer]
        end
    end

    return batch_update
end

# Updates the weights and biases of a network with a mini-batch-update.
# alpha: The learning rate which scales the adjustment of the parameters.
# batch_update: A struct holding the updates of the weights and biases for each layer.
# network: The network which is being trained.
function update_network(α::Float64, batch_update::MiniBatchUpdate, network::NeuralNetwork)
    for (id, layer) in enumerate(network.layers)
        layer.weights -= α .* batch_update.updates[id].weight_update
        if id < size(batch_update.updates)[1]
            layer.bias -= α .* batch_update.updates[id + 1].bias_update'
        end
    end

    return network
end

# Updates the weights of a network in order to reduce the loss of its output.
# cycles: The number of steps which are used to train the network. At each step one batch is fed.
# batch_size: The number of samples in each batch.
# network: The network which is being trained.
# alpha: The learning rate which scales the adjustment of the parameters.
function train_network(cycles::Int, batch_size::Int, network::NeuralNetwork, training_data::Array{Float64,3}, training_data_labels::Array{Int64,1}, 
        α::Float64=0.001, training_set_size::Int=60000, num_categories::Int=10)    
    for i in 1:cycles
        current_batch, current_batch_labels = build_batch(batch_size, training_data, training_data_labels)        
        batch_update = execute_minibatch(current_batch, current_batch_labels, network)
        network = update_network(α, batch_update, network)
    end

    return network
end

# Returns an array with input vectors and and array with labels in order to perform stochastic gradient descent.
# batch_size: Number of input vectors which are fed before the network is updated.
# training_data: The training data from which it is randomly sampled.
# training_data_labels: The respective label for each input.
# training_set_size: The number of samples in the training_data set.
# num_categories: The number of different labels.
function build_batch(batch_size::Int, training_data::Array{Float64,3}, training_data_labels::Array{Int64,1}, 
        training_set_size::Int=60000, num_categories::Int=10)
    current_batch = []
    current_batch_labels = []
    for j in 1:batch_size
        rnd = ceil(Int,  rand() * training_set_size)
        push!(current_batch, vcat(training_data[:, :, rnd]...))
        push!(current_batch_labels, target_vector(training_data_labels[rnd], num_categories))
    end
    
    return current_batch, current_batch_labels    
end
