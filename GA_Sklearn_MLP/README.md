# GA for Sklearn MLP Model

This genetic algorithm class was created to wrap onto the Sklearn Multi-Layer Perceptrion ([MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)) neural network model.

The model calls for an argument of layers, which is a tuple of the layer size of the model. For example, the call

```
sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ))
```

Would generate a network of architecture:

```
inputs -> (n_inputs, 100) -> (100, n_ouputs) -> outputs
```

Then the tuple can be extended to have more and more layers, such as

```
sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 10, 10))
```

Would generate a network of architecture:

```
inputs -> (n_inputs, 100) -> (100, 10) -> (10, 10) -> (10, n_ouputs) -> outputs
```

The purpose of the package in this repo is to generate this tuple through a genetic algorithm search. An interesting aspect is thus, if at any point, a gene is mutated to zero, the network architecture must shift; this is because it does not make sense to have a layer of size 0 in a neural network

```
(100, 0, 10) --> (100, 10)
```

