/**
 * CNN Visualizer - Constants and Configuration
 */

export const LAYER_TYPES = {
    INPUT: 'input',
    CONV2D: 'conv2d',
    MAXPOOL: 'maxpool',
    FLATTEN: 'flatten',
    DENSE: 'dense',
    DROPOUT: 'dropout',
    OUTPUT: 'output'
};

export const LAYER_CONFIGS = {
    [LAYER_TYPES.CONV2D]: {
        name: 'Conv2D',
        defaults: {
            filters: 32,
            kernelSize: 3,
            strides: 1,
            activation: 'relu',
            padding: 'same'
        },
        color: '#667eea',
        description: 'Convolutional layer that extracts features using learnable filters. Detects edges, textures, and patterns in the input.',
        tooltip: 'Extracts features using a sliding filter. The filter learns to detect patterns like edges and shapes.'
    },
    [LAYER_TYPES.MAXPOOL]: {
        name: 'MaxPooling',
        defaults: {
            poolSize: 2,
            strides: 2
        },
        color: '#3fb950',
        description: 'Reduces spatial dimensions by keeping only the maximum value in each window.',
        tooltip: 'Reduces image size by keeping the strongest activations. Helps the network focus on important features.'
    },
    [LAYER_TYPES.FLATTEN]: {
        name: 'Flatten',
        defaults: {},
        color: '#d29922',
        description: 'Converts multi-dimensional data into a 1D vector for dense layers.',
        tooltip: 'Reshapes the 2D feature maps into a 1D array so it can connect to Dense layers.'
    },
    [LAYER_TYPES.DENSE]: {
        name: 'Dense',
        defaults: {
            units: 128,
            activation: 'relu'
        },
        color: '#f85149',
        description: 'Fully connected layer where each neuron connects to all inputs.',
        tooltip: 'Every neuron connects to all neurons in the previous layer. Used for learning complex patterns.'
    },
    [LAYER_TYPES.DROPOUT]: {
        name: 'Dropout',
        defaults: {
            rate: 0.25
        },
        color: '#8b949e',
        description: 'Randomly disables neurons during training to prevent overfitting.',
        tooltip: 'Randomly turns off some neurons during training. Helps the network generalize better.'
    }
};

export const DATASETS = {
    mnist: {
        name: 'MNIST',
        description: 'Handwritten digits (0-9)',
        inputShape: [28, 28, 1],
        numClasses: 10,
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    },
    cifar10: {
        name: 'CIFAR-10',
        description: 'Small color images',
        inputShape: [32, 32, 3],
        numClasses: 10,
        labels: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }
};

export const PRESETS = {
    simple: {
        name: 'Simple CNN',
        layers: [
            { type: LAYER_TYPES.CONV2D, config: { filters: 16, kernelSize: 3 } },
            { type: LAYER_TYPES.MAXPOOL, config: { poolSize: 2 } },
            { type: LAYER_TYPES.FLATTEN, config: {} },
            { type: LAYER_TYPES.DENSE, config: { units: 64 } }
        ]
    },
    lenet: {
        name: 'LeNet-5',
        layers: [
            { type: LAYER_TYPES.CONV2D, config: { filters: 6, kernelSize: 5, padding: 'valid' } },
            { type: LAYER_TYPES.MAXPOOL, config: { poolSize: 2 } },
            { type: LAYER_TYPES.CONV2D, config: { filters: 16, kernelSize: 5, padding: 'valid' } },
            { type: LAYER_TYPES.MAXPOOL, config: { poolSize: 2 } },
            { type: LAYER_TYPES.FLATTEN, config: {} },
            { type: LAYER_TYPES.DENSE, config: { units: 120 } },
            { type: LAYER_TYPES.DENSE, config: { units: 84 } }
        ]
    }
};

export const ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear'];

export const CHART_COLORS = {
    loss: '#f85149',
    accuracy: '#3fb950',
    grid: '#21262d',
    text: '#8b949e'
};
