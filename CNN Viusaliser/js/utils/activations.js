/**
 * CNN Visualizer - Activation Functions
 * Implementations and explanations for common activation functions
 */

export const ACTIVATIONS = {
    none: {
        name: 'None (Linear)',
        description: 'No activation applied. Output equals input directly. Rarely used in hidden layers as it cannot learn non-linear patterns.',
        formula: 'f(x) = x',
        apply: (x) => x,
        color: '#8b949e',
        graph: {
            xRange: [-3, 3],
            yRange: [-3, 3]
        }
    },

    relu: {
        name: 'ReLU',
        description: 'Rectified Linear Unit - the most popular activation. Sets all negative values to zero while keeping positive values unchanged. Computationally efficient and helps avoid vanishing gradients.',
        formula: 'f(x) = max(0, x)',
        apply: (x) => Math.max(0, x),
        color: '#3fb950',
        graph: {
            xRange: [-3, 3],
            yRange: [-0.5, 3]
        }
    },

    leakyRelu: {
        name: 'Leaky ReLU',
        description: 'Allows a small gradient for negative values instead of zero. This prevents "dying ReLU" problem where neurons get stuck outputting zero.',
        formula: 'f(x) = max(0.01x, x)',
        alpha: 0.01,
        apply: (x, alpha = 0.01) => x >= 0 ? x : alpha * x,
        color: '#56d364',
        graph: {
            xRange: [-3, 3],
            yRange: [-0.5, 3]
        }
    },

    sigmoid: {
        name: 'Sigmoid',
        description: 'Squashes values to a 0-1 range. Historically popular but can cause vanishing gradients. Often used in output layer for binary classification.',
        formula: 'f(x) = 1 / (1 + e^(-x))',
        apply: (x) => 1 / (1 + Math.exp(-x)),
        color: '#667eea',
        graph: {
            xRange: [-5, 5],
            yRange: [-0.1, 1.1]
        }
    },

    tanh: {
        name: 'Tanh',
        description: 'Hyperbolic tangent - squashes values to -1 to 1 range. Zero-centered unlike sigmoid, which can help with training. Common in RNNs.',
        formula: 'f(x) = (e^x - e^(-x)) / (e^x + e^(-x))',
        apply: (x) => Math.tanh(x),
        color: '#764ba2',
        graph: {
            xRange: [-3, 3],
            yRange: [-1.2, 1.2]
        }
    },

    elu: {
        name: 'ELU',
        description: 'Exponential Linear Unit - like ReLU for positive values but smoothly approaches -α for negative values. Can produce negative outputs, helping with zero-centering.',
        formula: 'f(x) = x if x > 0, α(e^x - 1) if x ≤ 0',
        alpha: 1.0,
        apply: (x, alpha = 1.0) => x >= 0 ? x : alpha * (Math.exp(x) - 1),
        color: '#f85149',
        graph: {
            xRange: [-3, 3],
            yRange: [-1.5, 3]
        }
    },

    softplus: {
        name: 'Softplus',
        description: 'Smooth approximation of ReLU. Always positive and differentiable everywhere. Useful when smooth gradients are important.',
        formula: 'f(x) = ln(1 + e^x)',
        apply: (x) => Math.log(1 + Math.exp(x)),
        color: '#d29922',
        graph: {
            xRange: [-3, 3],
            yRange: [-0.1, 3.5]
        }
    },

    swish: {
        name: 'Swish',
        description: 'Self-gated activation discovered by Google. Often outperforms ReLU in deep networks. Smooth and non-monotonic.',
        formula: 'f(x) = x · sigmoid(x)',
        apply: (x) => x / (1 + Math.exp(-x)),
        color: '#a371f7',
        graph: {
            xRange: [-5, 5],
            yRange: [-1, 5]
        }
    }
};

/**
 * Apply activation function element-wise to an array
 */
export function applyActivation(data, activationType) {
    const activation = ACTIVATIONS[activationType];
    if (!activation) return data;

    if (data instanceof Float32Array) {
        const result = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
            result[i] = activation.apply(data[i]);
        }
        return result;
    }

    return data.map(x => activation.apply(x));
}

/**
 * Generate graph points for activation function visualization
 */
export function getActivationGraphPoints(activationType, numPoints = 100) {
    const activation = ACTIVATIONS[activationType];
    if (!activation) return [];

    const { xRange } = activation.graph;
    const step = (xRange[1] - xRange[0]) / numPoints;
    const points = [];

    for (let x = xRange[0]; x <= xRange[1]; x += step) {
        points.push({
            x: x,
            y: activation.apply(x)
        });
    }

    return points;
}

/**
 * Get all activations as an array for UI
 */
export function getActivationList() {
    return Object.entries(ACTIVATIONS).map(([key, value]) => ({
        id: key,
        ...value
    }));
}
