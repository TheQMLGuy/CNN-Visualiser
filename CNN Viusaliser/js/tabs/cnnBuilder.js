/**
 * CNN Visualizer - Full CNN Builder Tab
 * Complete network building and training functionality
 */

import { LAYER_TYPES, LAYER_CONFIGS, PRESETS, DATASETS } from '../utils/constants.js';
import { ArchitectureBuilder } from '../ui/architectureBuilder.js';
import { ModelEngine } from '../ml/modelEngine.js';
import { FeatureMapRenderer } from '../visualization/featureMapRenderer.js';
import { TrainingCharts } from '../visualization/trainingCharts.js';
import { ImageRenderer } from '../visualization/imageRenderer.js';

export class CNNBuilder {
    constructor(container, dataLoader) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.modelEngine = new ModelEngine();
        this.architectureBuilder = null;
        this.trainingCharts = null;
        this.featureMapRenderer = null;

        this.isTraining = false;
        this.selectedImageIndex = 0;

        this.init();
    }

    init() {
        this.render();
        this.initComponents();
        this.attachEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="cnn-builder">
                <div class="builder-left">
                    <div class="layer-palette">
                        <h3>🧱 Layer Palette</h3>
                        <p class="hint">Click to add layers</p>
                        <div class="layer-buttons">
                            <button class="layer-btn" data-layer="conv2d">
                                <span class="layer-icon conv-icon">🔲</span>
                                <span class="layer-info">
                                    <span class="layer-name">Conv2D</span>
                                    <span class="layer-desc">Feature extraction</span>
                                </span>
                            </button>
                            <button class="layer-btn" data-layer="maxpool">
                                <span class="layer-icon pool-icon">📊</span>
                                <span class="layer-info">
                                    <span class="layer-name">MaxPooling</span>
                                    <span class="layer-desc">Downsampling</span>
                                </span>
                            </button>
                            <button class="layer-btn" data-layer="flatten">
                                <span class="layer-icon flatten-icon">═══</span>
                                <span class="layer-info">
                                    <span class="layer-name">Flatten</span>
                                    <span class="layer-desc">2D → 1D</span>
                                </span>
                            </button>
                            <button class="layer-btn" data-layer="dense">
                                <span class="layer-icon dense-icon">●●●</span>
                                <span class="layer-info">
                                    <span class="layer-name">Dense</span>
                                    <span class="layer-desc">Fully connected</span>
                                </span>
                            </button>
                            <button class="layer-btn" data-layer="dropout">
                                <span class="layer-icon dropout-icon">⚫⚪</span>
                                <span class="layer-info">
                                    <span class="layer-name">Dropout</span>
                                    <span class="layer-desc">Regularization</span>
                                </span>
                            </button>
                        </div>
                        
                        <div class="preset-section">
                            <h4>Quick Presets</h4>
                            <button class="preset-btn" data-preset="simple">Simple CNN</button>
                            <button class="preset-btn" data-preset="lenet">LeNet-5</button>
                        </div>
                    </div>
                </div>

                <div class="builder-center">
                    <div class="network-header">
                        <h3>Network Architecture</h3>
                        <button class="action-btn" id="clear-network-btn">🗑️ Clear</button>
                    </div>
                    
                    <div class="network-canvas">
                        <div class="input-layer">
                            <canvas id="builder-input-preview" width="56" height="56"></canvas>
                            <div class="layer-label">
                                <span class="layer-title">Input</span>
                                <span class="layer-shape" id="builder-input-shape">28×28×1</span>
                            </div>
                        </div>
                        
                        <div class="layer-connector">→</div>
                        
                        <div class="architecture-layers" id="builder-architecture-layers">
                            <div class="empty-state" id="builder-empty-state">
                                <p>➕ Click layers to build your network</p>
                            </div>
                        </div>
                        
                        <div class="layer-connector output-connector" style="display: none;">→</div>
                        
                        <div class="output-layer" id="builder-output-layer" style="display: none;">
                            <div class="output-bars" id="builder-output-bars"></div>
                            <div class="layer-label">
                                <span class="layer-title">Output</span>
                                <span class="layer-shape" id="builder-output-shape">10 classes</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="feature-maps-section" id="builder-feature-maps" style="display: none;">
                        <h4>Feature Maps</h4>
                        <div class="feature-maps-container" id="builder-feature-maps-container"></div>
                    </div>
                </div>

                <div class="builder-right">
                    <div class="training-controls">
                        <h3>⚙️ Training</h3>
                        
                        <div class="control-group">
                            <label>Learning Rate</label>
                            <div class="range-input">
                                <input type="range" id="builder-lr" min="-4" max="-1" step="0.1" value="-2">
                                <span id="builder-lr-value">0.01</span>
                            </div>
                        </div>
                        
                        <div class="control-group">
                            <label>Epochs</label>
                            <input type="number" id="builder-epochs" value="5" min="1" max="50">
                        </div>
                        
                        <div class="control-group">
                            <label>Batch Size</label>
                            <select id="builder-batch-size">
                                <option value="16">16</option>
                                <option value="32" selected>32</option>
                                <option value="64">64</option>
                                <option value="128">128</option>
                            </select>
                        </div>
                        
                        <button class="train-btn" id="builder-train-btn">
                            ▶ Train Network
                        </button>
                        <button class="stop-btn" id="builder-stop-btn" style="display: none;">
                            ⏹ Stop Training
                        </button>
                    </div>
                    
                    <div class="training-stats">
                        <div class="stat-card">
                            <span class="stat-label">Epoch</span>
                            <span class="stat-value" id="builder-epoch">0 / 0</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-label">Loss</span>
                            <span class="stat-value" id="builder-loss">—</span>
                        </div>
                        <div class="stat-card">
                            <span class="stat-label">Accuracy</span>
                            <span class="stat-value" id="builder-accuracy">—</span>
                        </div>
                    </div>
                    
                    <div class="chart-section">
                        <h4>Training Progress</h4>
                        <div class="chart-container">
                            <canvas id="builder-loss-chart" width="240" height="100"></canvas>
                        </div>
                        <div class="chart-container">
                            <canvas id="builder-accuracy-chart" width="240" height="100"></canvas>
                        </div>
                    </div>
                    
                    <div class="sample-images-section">
                        <h4>Sample Images</h4>
                        <div class="sample-grid" id="builder-sample-grid"></div>
                        <button class="small-btn" id="builder-load-samples">Load Samples</button>
                    </div>
                </div>
            </div>
        `;
    }

    initComponents() {
        // Initialize architecture builder
        const archContainer = this.container.querySelector('#builder-architecture-layers');
        this.architectureBuilder = new ArchitectureBuilder(archContainer, (layers) => {
            this.onArchitectureChange(layers);
        });

        // Set empty state and output layer references
        this.architectureBuilder.emptyState = this.container.querySelector('#builder-empty-state');
        this.architectureBuilder.outputConnector = this.container.querySelector('.output-connector');
        this.architectureBuilder.outputLayer = this.container.querySelector('#builder-output-layer');

        // Initialize charts
        const lossCanvas = this.container.querySelector('#builder-loss-chart');
        const accCanvas = this.container.querySelector('#builder-accuracy-chart');
        this.trainingCharts = new TrainingCharts(lossCanvas, accCanvas);

        // Initialize feature map renderer
        const fmContainer = this.container.querySelector('#builder-feature-maps-container');
        this.featureMapRenderer = new FeatureMapRenderer(fmContainer);

        // Update dataset info
        if (this.dataLoader) {
            const info = this.dataLoader.getDatasetInfo();
            if (info) {
                this.container.querySelector('#builder-input-shape').textContent =
                    `${info.inputShape[0]}×${info.inputShape[1]}×${info.inputShape[2]}`;
                this.container.querySelector('#builder-output-shape').textContent =
                    `${info.numClasses} classes`;
            }
        }
    }

    attachEventListeners() {
        // Layer buttons
        this.container.querySelectorAll('.layer-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const layerType = btn.dataset.layer;
                this.architectureBuilder.addLayer(layerType);
            });
        });

        // Preset buttons
        this.container.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const presetName = btn.dataset.preset;
                this.architectureBuilder.loadPreset(presetName);
            });
        });

        // Clear button
        this.container.querySelector('#clear-network-btn').addEventListener('click', () => {
            this.architectureBuilder.clear();
            this.featureMapRenderer.clear();
        });

        // Learning rate slider
        const lrSlider = this.container.querySelector('#builder-lr');
        lrSlider.addEventListener('input', () => {
            const lr = Math.pow(10, parseFloat(lrSlider.value));
            this.container.querySelector('#builder-lr-value').textContent = lr.toFixed(lr < 0.01 ? 4 : 3);
        });

        // Train button
        this.container.querySelector('#builder-train-btn').addEventListener('click', () => {
            this.startTraining();
        });

        // Stop button
        this.container.querySelector('#builder-stop-btn').addEventListener('click', () => {
            this.stopTraining();
        });

        // Load samples button
        this.container.querySelector('#builder-load-samples').addEventListener('click', () => {
            this.loadSampleImages();
        });
    }

    onArchitectureChange(layers) {
        console.log('Architecture updated:', layers.length, 'layers');
    }

    async startTraining() {
        const architecture = this.architectureBuilder.getArchitecture();

        if (architecture.length === 0) {
            alert('Please add at least one layer to the network');
            return;
        }

        const hasFlatten = architecture.some(l => l.type === LAYER_TYPES.FLATTEN);
        if (!hasFlatten) {
            alert('Add a Flatten layer before Dense layers');
            return;
        }

        const lr = Math.pow(10, parseFloat(this.container.querySelector('#builder-lr').value));
        const epochs = parseInt(this.container.querySelector('#builder-epochs').value);
        const batchSize = parseInt(this.container.querySelector('#builder-batch-size').value);

        const datasetInfo = this.dataLoader.getDatasetInfo();

        try {
            // Build and compile model
            this.modelEngine.buildModel(architecture, datasetInfo.inputShape, datasetInfo.numClasses);
            this.modelEngine.compile(lr);

            // Update UI
            this.setTrainingState(true);
            this.trainingCharts.reset();

            // Start training
            const trainData = this.dataLoader.trainData;

            await this.modelEngine.train(trainData, { epochs, batchSize }, {
                onEpochBegin: (epoch) => {
                    this.container.querySelector('#builder-epoch').textContent =
                        `${epoch + 1} / ${epochs}`;
                },
                onEpochEnd: async (epoch, logs, history) => {
                    this.container.querySelector('#builder-epoch').textContent =
                        `${epoch + 1} / ${epochs}`;
                    this.container.querySelector('#builder-loss').textContent =
                        logs.loss.toFixed(4);
                    this.container.querySelector('#builder-accuracy').textContent =
                        `${(logs.acc * 100).toFixed(1)}%`;

                    this.trainingCharts.update(history);
                    this.updateFeatureMaps();
                },
                onBatchEnd: (batch, logs) => { }
            });

            this.updateOutputPrediction();

        } catch (error) {
            console.error('Training error:', error);
            alert(`Training error: ${error.message}`);
        } finally {
            this.setTrainingState(false);
        }
    }

    stopTraining() {
        this.modelEngine.stopTraining();
    }

    setTrainingState(isTraining) {
        this.isTraining = isTraining;

        const trainBtn = this.container.querySelector('#builder-train-btn');
        const stopBtn = this.container.querySelector('#builder-stop-btn');

        trainBtn.style.display = isTraining ? 'none' : 'block';
        stopBtn.style.display = isTraining ? 'block' : 'none';

        this.container.querySelectorAll('.layer-btn').forEach(btn => {
            btn.disabled = isTraining;
            btn.style.opacity = isTraining ? '0.5' : '1';
        });
    }

    loadSampleImages() {
        const grid = this.container.querySelector('#builder-sample-grid');
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
                this.selectedImageIndex = sample.index;
                this.updateInputPreview(sample.index);
            });

            grid.appendChild(wrapper);
            sample.image.dispose();
        });
    }

    updateInputPreview(imageIndex) {
        this.selectedImageIndex = imageIndex;

        const sample = this.dataLoader.getImage(imageIndex);
        if (!sample) return;

        const canvas = this.container.querySelector('#builder-input-preview');
        ImageRenderer.renderScaled(canvas, sample.image, 56, 56);
        sample.image.dispose();

        this.updateFeatureMaps();
    }

    updateFeatureMaps() {
        if (!this.modelEngine.isReady()) return;

        try {
            const sample = this.dataLoader.getImage(this.selectedImageIndex);
            if (!sample) return;

            const activations = this.modelEngine.getActivations(sample.image);
            this.featureMapRenderer.render(activations);

            // Show feature maps section
            this.container.querySelector('#builder-feature-maps').style.display = 'block';

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

            const outputBars = this.container.querySelector('#builder-output-bars');
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

    onActivate() {
        // Load samples if not already loaded
        if (this.container.querySelector('#builder-sample-grid').children.length === 0) {
            this.loadSampleImages();
        }
    }

    dispose() {
        this.modelEngine.dispose();
    }
}
