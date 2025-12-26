/**
 * CNN Visualizer - Main Application
 * Interactive browser-based CNN visualization tool
 */

import { LAYER_TYPES, DATASETS, PRESETS } from './utils/constants.js';
import { TooltipManager } from './utils/tooltips.js';
import { ArchitectureBuilder } from './ui/architectureBuilder.js';
import { ModelEngine } from './ml/modelEngine.js';
import { DataLoader } from './ml/dataLoader.js';
import { FeatureMapRenderer } from './visualization/featureMapRenderer.js';
import { TrainingCharts } from './visualization/trainingCharts.js';
import { ImageRenderer } from './visualization/imageRenderer.js';

class CNNVisualizer {
    constructor() {
        this.modelEngine = new ModelEngine();
        this.dataLoader = new DataLoader();
        this.tooltipManager = null;
        this.architectureBuilder = null;
        this.featureMapRenderer = null;
        this.trainingCharts = null;

        this.currentDataset = 'mnist';
        this.isTraining = false;
        this.selectedImageIndex = 0;

        this.init();
    }

    async init() {
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        this.updateTFStatus(true);

        // Initialize components
        this.initTooltips();
        this.initArchitectureBuilder();
        this.initCharts();
        this.initFeatureMaps();
        this.initEventListeners();

        // Load initial dataset
        await this.loadDataset('mnist');

        console.log('CNN Visualizer initialized');
    }

    updateTFStatus(ready) {
        const statusEl = document.getElementById('tf-status');
        const statusText = statusEl.querySelector('.status-text');

        if (ready) {
            statusEl.classList.add('ready');
            statusText.textContent = 'TensorFlow.js Ready';
        } else {
            statusEl.classList.remove('ready');
            statusText.textContent = 'Loading...';
        }
    }

    initTooltips() {
        this.tooltipManager = new TooltipManager();
    }

    initArchitectureBuilder() {
        const container = document.getElementById('architecture-layers');

        this.architectureBuilder = new ArchitectureBuilder(container, (layers) => {
            this.onArchitectureChange(layers);
        });
    }

    initCharts() {
        const lossCanvas = document.getElementById('loss-chart');
        const accuracyCanvas = document.getElementById('accuracy-chart');

        this.trainingCharts = new TrainingCharts(lossCanvas, accuracyCanvas);
    }

    initFeatureMaps() {
        const container = document.getElementById('feature-maps-container');
        this.featureMapRenderer = new FeatureMapRenderer(container);
    }

    initEventListeners() {
        // Layer buttons
        document.querySelectorAll('.layer-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const layerType = btn.dataset.layer;
                this.architectureBuilder.addLayer(layerType);
            });
        });

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const presetName = btn.dataset.preset;
                this.architectureBuilder.loadPreset(presetName);
            });
        });

        // Clear network button
        document.getElementById('clear-network').addEventListener('click', () => {
            this.architectureBuilder.clear();
            this.featureMapRenderer.clear();
        });

        // Dataset selector
        document.getElementById('dataset').addEventListener('change', async (e) => {
            await this.loadDataset(e.target.value);
        });

        // Learning rate slider
        const lrSlider = document.getElementById('learning-rate');
        const lrValue = document.getElementById('lr-value');

        lrSlider.addEventListener('input', () => {
            const lr = Math.pow(10, parseFloat(lrSlider.value));
            lrValue.textContent = lr.toFixed(lr < 0.01 ? 4 : 3);
        });

        // Train button
        document.getElementById('train-btn').addEventListener('click', () => {
            this.startTraining();
        });

        // Stop button
        document.getElementById('stop-btn').addEventListener('click', () => {
            this.stopTraining();
        });

        // Load samples button
        document.getElementById('load-samples').addEventListener('click', () => {
            this.loadSampleImages();
        });
    }

    async loadDataset(datasetName) {
        this.currentDataset = datasetName;

        const progressText = document.getElementById('progress-text');
        progressText.textContent = `Loading ${datasetName.toUpperCase()} dataset...`;

        try {
            await this.dataLoader.loadDataset(datasetName);

            // Update input shape display
            this.architectureBuilder.setDataset(datasetName);

            // Update output shape
            const outputShape = document.getElementById('output-shape');
            const datasetInfo = DATASETS[datasetName];
            outputShape.textContent = `${datasetInfo.numClasses} classes`;

            progressText.textContent = `${datasetName.toUpperCase()} loaded successfully`;

            // Load sample images
            await this.loadSampleImages();

            // Show first image in input preview
            this.updateInputPreview(0);

        } catch (error) {
            console.error('Failed to load dataset:', error);
            progressText.textContent = `Failed to load ${datasetName}`;
        }
    }

    async loadSampleImages() {
        const grid = document.getElementById('sample-grid');
        grid.innerHTML = '';

        const samples = this.dataLoader.getSampleImages(10);
        if (!samples) return;

        const datasetInfo = this.dataLoader.getDatasetInfo();
        const isColor = datasetInfo.inputShape[2] === 3;

        samples.forEach((sample, index) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'sample-image';
            wrapper.title = `Label: ${datasetInfo.labels[sample.label]}`;

            const canvas = document.createElement('canvas');
            ImageRenderer.renderToCanvas(canvas, sample.image, isColor);

            wrapper.appendChild(canvas);
            wrapper.addEventListener('click', () => {
                this.updateInputPreview(sample.index);
            });

            grid.appendChild(wrapper);

            // Dispose the tensor after rendering
            sample.image.dispose();
        });
    }

    updateInputPreview(imageIndex) {
        this.selectedImageIndex = imageIndex;

        const sample = this.dataLoader.getImage(imageIndex);
        if (!sample) return;

        const canvas = document.getElementById('input-preview');
        const datasetInfo = this.dataLoader.getDatasetInfo();

        ImageRenderer.renderScaled(canvas, sample.image, 56, 56);

        sample.image.dispose();

        // If model is ready, show feature maps
        this.updateFeatureMaps();
    }

    onArchitectureChange(layers) {
        // Could validate architecture here
        console.log('Architecture updated:', layers.length, 'layers');
    }

    async startTraining() {
        const architecture = this.architectureBuilder.getArchitecture();

        if (architecture.length === 0) {
            this.showStatus('Please add at least one layer to the network');
            return;
        }

        // Check if architecture ends with Dense layer (for classification)
        const hasFlatten = architecture.some(l => l.type === LAYER_TYPES.FLATTEN);
        const hasDense = architecture.some(l => l.type === LAYER_TYPES.DENSE);

        if (!hasFlatten) {
            this.showStatus('Add a Flatten layer before Dense layers');
            return;
        }

        // Get training parameters
        const learningRate = Math.pow(10, parseFloat(document.getElementById('learning-rate').value));
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);

        // Get dataset info
        const datasetInfo = DATASETS[this.currentDataset];

        try {
            // Build and compile model
            this.showStatus('Building model...');
            this.modelEngine.buildModel(architecture, datasetInfo.inputShape, datasetInfo.numClasses);
            this.modelEngine.compile(learningRate);

            // Update UI
            this.setTrainingState(true);
            this.trainingCharts.reset();

            // Start training
            const trainData = this.dataLoader.trainData;

            await this.modelEngine.train(trainData, { epochs, batchSize }, {
                onEpochBegin: (epoch) => {
                    this.showStatus(`Training epoch ${epoch + 1}/${epochs}...`);
                },
                onEpochEnd: async (epoch, logs, history) => {
                    // Update stats
                    document.getElementById('current-epoch').textContent = `${epoch + 1} / ${epochs}`;
                    document.getElementById('current-loss').textContent = logs.loss.toFixed(4);
                    document.getElementById('current-accuracy').textContent = `${(logs.acc * 100).toFixed(1)}%`;

                    // Update progress bar
                    const progress = ((epoch + 1) / epochs) * 100;
                    document.getElementById('progress-fill').style.width = `${progress}%`;

                    // Update charts
                    this.trainingCharts.update(history);

                    // Update feature maps periodically
                    this.updateFeatureMaps();

                    // Animate connectors
                    this.animateDataFlow();
                },
                onBatchEnd: (batch, logs) => {
                    // Could update more frequently here
                }
            });

            this.showStatus('Training complete!');
            this.updateOutputPrediction();

        } catch (error) {
            console.error('Training error:', error);
            this.showStatus(`Error: ${error.message}`);
        } finally {
            this.setTrainingState(false);
        }
    }

    stopTraining() {
        this.modelEngine.stopTraining();
        this.showStatus('Training stopped');
    }

    setTrainingState(isTraining) {
        this.isTraining = isTraining;

        const trainBtn = document.getElementById('train-btn');
        const stopBtn = document.getElementById('stop-btn');

        trainBtn.style.display = isTraining ? 'none' : 'flex';
        trainBtn.disabled = isTraining;
        stopBtn.style.display = isTraining ? 'flex' : 'none';

        // Disable layer buttons during training
        document.querySelectorAll('.layer-btn').forEach(btn => {
            btn.disabled = isTraining;
            btn.style.opacity = isTraining ? 0.5 : 1;
        });
    }

    updateFeatureMaps() {
        if (!this.modelEngine.isReady()) return;

        try {
            const sample = this.dataLoader.getImage(this.selectedImageIndex);
            if (!sample) return;

            const activations = this.modelEngine.getActivations(sample.image);
            this.featureMapRenderer.render(activations);

            // Dispose tensors
            sample.image.dispose();
            activations.forEach(a => {
                if (a.tensor && !a.tensor.isDisposed) {
                    a.tensor.dispose();
                }
            });
        } catch (error) {
            console.warn('Could not update feature maps:', error);
        }
    }

    updateOutputPrediction() {
        if (!this.modelEngine.isReady()) return;

        try {
            const sample = this.dataLoader.getImage(this.selectedImageIndex);
            if (!sample) return;

            const prediction = this.modelEngine.predict(sample.image);
            const probs = prediction.dataSync();

            // Update output bars
            const outputBars = document.getElementById('output-bars');
            outputBars.innerHTML = '';

            const maxProb = Math.max(...probs);

            for (let i = 0; i < probs.length; i++) {
                const bar = document.createElement('div');
                bar.className = 'output-bar';
                bar.style.width = `${(probs[i] / maxProb) * 100}%`;
                bar.style.opacity = probs[i] / maxProb;
                outputBars.appendChild(bar);
            }

            sample.image.dispose();
            prediction.dispose();
        } catch (error) {
            console.warn('Could not update prediction:', error);
        }
    }

    animateDataFlow() {
        const connectors = document.querySelectorAll('.layer-connector');
        connectors.forEach((connector, index) => {
            setTimeout(() => {
                connector.classList.add('animating');
                setTimeout(() => {
                    connector.classList.remove('animating');
                }, 500);
            }, index * 100);
        });
    }

    showStatus(text) {
        document.getElementById('progress-text').textContent = text;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.cnnVisualizer = new CNNVisualizer();
});
