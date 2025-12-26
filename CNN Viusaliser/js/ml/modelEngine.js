/**
 * CNN Visualizer - Model Engine
 * Handles building, training, and inference with TensorFlow.js models
 */

import { LAYER_TYPES, LAYER_CONFIGS, DATASETS } from '../utils/constants.js';

export class ModelEngine {
    constructor() {
        this.model = null;
        this.isTraining = false;
        this.shouldStop = false;
        this.activationModels = [];
        this.trainingHistory = {
            loss: [],
            accuracy: [],
            valLoss: [],
            valAccuracy: []
        };
    }

    buildModel(architecture, inputShape, numClasses) {
        // Dispose previous model
        this.dispose();

        this.model = tf.sequential();
        let currentShape = [...inputShape];

        // Input layer
        let firstLayer = true;

        for (const layer of architecture) {
            const config = { ...LAYER_CONFIGS[layer.type].defaults, ...layer.config };

            try {
                switch (layer.type) {
                    case LAYER_TYPES.CONV2D:
                        const convConfig = {
                            filters: config.filters,
                            kernelSize: config.kernelSize,
                            strides: config.strides || 1,
                            padding: config.padding || 'same',
                            activation: config.activation || 'relu'
                        };
                        if (firstLayer) {
                            convConfig.inputShape = inputShape;
                            firstLayer = false;
                        }
                        this.model.add(tf.layers.conv2d(convConfig));

                        // Update shape
                        if (config.padding === 'valid') {
                            currentShape[0] = currentShape[0] - config.kernelSize + 1;
                            currentShape[1] = currentShape[1] - config.kernelSize + 1;
                        }
                        currentShape[2] = config.filters;
                        break;

                    case LAYER_TYPES.MAXPOOL:
                        const poolConfig = {
                            poolSize: config.poolSize || 2,
                            strides: config.strides || config.poolSize || 2
                        };
                        if (firstLayer) {
                            poolConfig.inputShape = inputShape;
                            firstLayer = false;
                        }
                        this.model.add(tf.layers.maxPooling2d(poolConfig));

                        // Update shape
                        currentShape[0] = Math.floor(currentShape[0] / poolConfig.poolSize);
                        currentShape[1] = Math.floor(currentShape[1] / poolConfig.poolSize);
                        break;

                    case LAYER_TYPES.FLATTEN:
                        const flattenConfig = {};
                        if (firstLayer) {
                            flattenConfig.inputShape = inputShape;
                            firstLayer = false;
                        }
                        this.model.add(tf.layers.flatten(flattenConfig));
                        currentShape = [currentShape[0] * currentShape[1] * currentShape[2]];
                        break;

                    case LAYER_TYPES.DENSE:
                        const denseConfig = {
                            units: config.units,
                            activation: config.activation || 'relu'
                        };
                        if (firstLayer) {
                            denseConfig.inputShape = inputShape;
                            firstLayer = false;
                        }
                        this.model.add(tf.layers.dense(denseConfig));
                        currentShape = [config.units];
                        break;

                    case LAYER_TYPES.DROPOUT:
                        const dropoutConfig = {
                            rate: config.rate || 0.25
                        };
                        this.model.add(tf.layers.dropout(dropoutConfig));
                        break;
                }
            } catch (error) {
                console.error(`Error adding layer ${layer.type}:`, error);
                throw new Error(`Failed to add ${layer.type} layer: ${error.message}`);
            }
        }

        // Add output layer
        this.model.add(tf.layers.dense({
            units: numClasses,
            activation: 'softmax'
        }));

        // Build activation models for visualization
        this.buildActivationModels();

        return this.model;
    }

    buildActivationModels() {
        // Dispose previous activation models
        this.activationModels.forEach(m => m.model.dispose());
        this.activationModels = [];

        if (!this.model) return;

        // Create models that output intermediate activations for ALL layer types
        for (let i = 0; i < this.model.layers.length; i++) {
            const layer = this.model.layers[i];
            const className = layer.getClassName();

            // Include Conv2D, MaxPooling2D, Flatten, and Dense layers
            if (className === 'Conv2D' || className === 'MaxPooling2D' ||
                className === 'Flatten' || className === 'Dense') {
                try {
                    const activationModel = tf.model({
                        inputs: this.model.input,
                        outputs: layer.output
                    });
                    this.activationModels.push({
                        model: activationModel,
                        layerIndex: i,
                        layerName: layer.name,
                        layerType: className
                    });
                } catch (e) {
                    console.warn(`Could not create activation model for layer ${i}:`, e);
                }
            }
        }
    }

    compile(learningRate = 0.01) {
        if (!this.model) throw new Error('No model to compile');

        this.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
    }

    async train(trainData, options, callbacks) {
        if (!this.model) throw new Error('No model to train');
        if (this.isTraining) throw new Error('Training already in progress');

        this.isTraining = true;
        this.shouldStop = false;
        this.trainingHistory = { loss: [], accuracy: [], valLoss: [], valAccuracy: [] };

        const {
            epochs = 5,
            batchSize = 32,
            validationSplit = 0.1
        } = options;

        try {
            await this.model.fit(trainData.images, trainData.labels, {
                epochs,
                batchSize,
                validationSplit,
                shuffle: true,
                callbacks: {
                    onEpochBegin: async (epoch) => {
                        if (callbacks.onEpochBegin) {
                            callbacks.onEpochBegin(epoch);
                        }
                    },
                    onEpochEnd: async (epoch, logs) => {
                        this.trainingHistory.loss.push(logs.loss);
                        this.trainingHistory.accuracy.push(logs.acc);
                        if (logs.val_loss !== undefined) {
                            this.trainingHistory.valLoss.push(logs.val_loss);
                            this.trainingHistory.valAccuracy.push(logs.val_acc);
                        }

                        if (callbacks.onEpochEnd) {
                            await callbacks.onEpochEnd(epoch, logs, this.trainingHistory);
                        }

                        if (this.shouldStop) {
                            this.model.stopTraining = true;
                        }
                    },
                    onBatchEnd: async (batch, logs) => {
                        if (callbacks.onBatchEnd) {
                            callbacks.onBatchEnd(batch, logs);
                        }

                        if (this.shouldStop) {
                            this.model.stopTraining = true;
                        }
                    }
                }
            });
        } finally {
            this.isTraining = false;
        }

        return this.trainingHistory;
    }

    stopTraining() {
        this.shouldStop = true;
    }

    predict(input) {
        if (!this.model) return null;

        return tf.tidy(() => {
            let inputTensor = input;
            if (!(input instanceof tf.Tensor)) {
                inputTensor = tf.tensor(input);
            }

            // Ensure correct shape
            if (inputTensor.rank === 3) {
                inputTensor = inputTensor.expandDims(0);
            }

            return this.model.predict(inputTensor);
        });
    }

    getActivations(input) {
        if (!this.model || this.activationModels.length === 0) return [];

        // Don't use tf.tidy() here since we return tensors that need to survive
        // The caller is responsible for disposing these tensors
        let inputTensor = input;
        let createdInput = false;

        if (!(input instanceof tf.Tensor)) {
            inputTensor = tf.tensor(input);
            createdInput = true;
        }

        let expandedInput = inputTensor;
        if (inputTensor.rank === 3) {
            expandedInput = inputTensor.expandDims(0);
        }

        const activations = [];

        for (const { model, layerIndex, layerName, layerType } of this.activationModels) {
            try {
                const activation = model.predict(expandedInput);
                activations.push({
                    layerIndex,
                    layerName,
                    layerType,
                    tensor: activation,
                    shape: activation.shape
                });
            } catch (e) {
                console.warn(`Could not get activation for layer ${layerName}:`, e);
            }
        }

        // Dispose intermediate tensors we created
        if (expandedInput !== inputTensor) {
            expandedInput.dispose();
        }
        if (createdInput) {
            inputTensor.dispose();
        }

        return activations;
    }

    getModelSummary() {
        if (!this.model) return null;

        const summary = [];
        let totalParams = 0;

        for (const layer of this.model.layers) {
            const config = layer.getConfig();
            const outputShape = layer.outputShape;
            const params = layer.countParams();
            totalParams += params;

            summary.push({
                name: layer.name,
                type: layer.getClassName(),
                outputShape: outputShape,
                params: params,
                config: config
            });
        }

        return { layers: summary, totalParams };
    }

    getLayerWeights(layerIndex) {
        if (!this.model || layerIndex >= this.model.layers.length) return null;

        const layer = this.model.layers[layerIndex];
        const weights = layer.getWeights();

        if (weights.length === 0) return null;

        return weights.map(w => ({
            name: w.name,
            shape: w.shape,
            data: w.dataSync()
        }));
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.activationModels.forEach(m => m.model.dispose());
        this.activationModels = [];
    }

    isReady() {
        return this.model !== null;
    }
}
