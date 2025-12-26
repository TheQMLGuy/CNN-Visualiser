/**
 * CNN Platform - Create Mode Controller
 * Build, train, and test custom CNNs
 */

import { LAYER_TYPES, PRESETS } from '../utils/constants.js';
import { ModelEngine } from '../ml/modelEngine.js';
import { TrainingCharts } from '../visualization/trainingCharts.js';
import { ImageRenderer } from '../visualization/imageRenderer.js';

export class CreateMode {
    constructor(container, dataLoader, onBack) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.onBack = onBack;
        this.modelEngine = new ModelEngine();
        this.trainingCharts = null;

        this.currentTab = 'training';
        this.architecture = [];
        this.isTraining = false;
        this.isTrained = false;

        this.render();
        this.attachEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="create-mode">
                <header class="mode-topbar">
                    <button class="back-btn" id="create-back-btn">← Back to Menu</button>
                    <h2>🔧 Create Your Own CNN</h2>
                    <div class="tab-switcher">
                        <button class="tab-switch-btn active" data-tab="training">Training</button>
                        <button class="tab-switch-btn" data-tab="testing">Testing</button>
                    </div>
                </header>

                <div class="create-content">
                    <!-- Training Tab -->
                    <div class="create-tab active" id="create-training-tab">
                        <div class="training-layout">
                            <aside class="layer-sidebar">
                                <h3>Add Layers</h3>
                                <div class="layer-buttons">
                                    <button class="add-layer-btn" data-type="conv2d">
                                        <span class="layer-icon">🔲</span> Conv2D
                                    </button>
                                    <button class="add-layer-btn" data-type="maxpool">
                                        <span class="layer-icon">📊</span> MaxPool
                                    </button>
                                    <button class="add-layer-btn" data-type="flatten">
                                        <span class="layer-icon">═══</span> Flatten
                                    </button>
                                    <button class="add-layer-btn" data-type="dense">
                                        <span class="layer-icon">●●●</span> Dense
                                    </button>
                                    <button class="add-layer-btn" data-type="dropout">
                                        <span class="layer-icon">⚫⚪</span> Dropout
                                    </button>
                                </div>
                                
                                <div class="presets-section">
                                    <h4>Quick Presets</h4>
                                    <button class="preset-btn" data-preset="simple">Simple CNN</button>
                                    <button class="preset-btn" data-preset="lenet">LeNet-5</button>
                                </div>
                                
                                <button class="clear-btn" id="clear-layers-btn">🗑️ Clear All</button>
                            </aside>

                            <main class="architecture-area">
                                <div class="architecture-header">
                                    <h3>Your Network Architecture</h3>
                                    <span class="layer-count" id="layer-count">0 layers</span>
                                </div>
                                <div class="architecture-canvas" id="architecture-canvas">
                                    <div class="input-block">
                                        <div class="block-icon">📥</div>
                                        <div class="block-label">Input</div>
                                        <div class="block-shape" id="input-shape">28×28×1</div>
                                    </div>
                                    <div class="layers-container" id="layers-container">
                                        <div class="empty-state">Click layers to add them here</div>
                                    </div>
                                    <div class="output-block" id="output-block" style="display: none;">
                                        <div class="block-icon">📤</div>
                                        <div class="block-label">Output</div>
                                        <div class="block-shape">10 classes</div>
                                    </div>
                                </div>
                            </main>

                            <aside class="training-sidebar">
                                <div class="training-config">
                                    <h3>Training Settings</h3>
                                    <div class="config-group">
                                        <label>Dataset</label>
                                        <select id="create-dataset">
                                            <option value="mnist">MNIST (Digits)</option>
                                            <option value="cifar10">CIFAR-10</option>
                                        </select>
                                    </div>
                                    <div class="config-group">
                                        <label>Learning Rate: <span id="lr-display">0.01</span></label>
                                        <input type="range" id="create-lr" min="-4" max="-1" step="0.1" value="-2">
                                    </div>
                                    <div class="config-group">
                                        <label>Epochs</label>
                                        <input type="number" id="create-epochs" value="5" min="1" max="50">
                                    </div>
                                    <div class="config-group">
                                        <label>Batch Size</label>
                                        <select id="create-batch">
                                            <option value="32">32</option>
                                            <option value="64">64</option>
                                            <option value="128">128</option>
                                        </select>
                                    </div>
                                </div>

                                <button class="train-btn" id="start-training-btn">▶ Start Training</button>
                                <button class="stop-btn" id="stop-training-btn" style="display: none;">⏹ Stop</button>

                                <div class="training-stats">
                                    <div class="stat-item">
                                        <span class="stat-label">Epoch</span>
                                        <span class="stat-value" id="stat-epoch">—</span>
                                    </div>
                                    <div class="stat-item">
                                        <span class="stat-label">Loss</span>
                                        <span class="stat-value" id="stat-loss">—</span>
                                    </div>
                                    <div class="stat-item">
                                        <span class="stat-label">Accuracy</span>
                                        <span class="stat-value" id="stat-accuracy">—</span>
                                    </div>
                                </div>

                                <div class="chart-container">
                                    <canvas id="create-loss-chart" width="220" height="100"></canvas>
                                </div>
                                <div class="chart-container">
                                    <canvas id="create-acc-chart" width="220" height="100"></canvas>
                                </div>
                            </aside>
                        </div>
                    </div>

                    <!-- Testing Tab -->
                    <div class="create-tab" id="create-testing-tab">
                        <div class="testing-layout">
                            <div class="drawing-section">
                                <h3>✏️ Draw a Digit</h3>
                                <div class="drawing-canvas-container">
                                    <canvas id="drawing-canvas" width="280" height="280"></canvas>
                                </div>
                                <div class="drawing-controls">
                                    <button id="clear-drawing-btn">🗑️ Clear</button>
                                    <button id="predict-btn" class="primary">🔮 Predict</button>
                                </div>
                            </div>

                            <div class="prediction-section">
                                <h3>📊 Prediction Results</h3>
                                <div class="prediction-display" id="prediction-display">
                                    <div class="prediction-main">
                                        <span class="pred-label">Predicted:</span>
                                        <span class="pred-value" id="pred-value">—</span>
                                    </div>
                                    <div class="confidence-bars" id="confidence-bars">
                                        <!-- Confidence bars generated dynamically -->
                                    </div>
                                </div>
                                
                                <div class="model-status" id="model-status">
                                    <span class="status-icon">⚠️</span>
                                    <span>Train your model first to make predictions</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.initDrawingCanvas();
        this.initCharts();
        this.updateDatasetInfo();
    }

    attachEventListeners() {
        // Back button
        this.container.querySelector('#create-back-btn').addEventListener('click', () => {
            this.dispose();
            this.onBack();
        });

        // Tab switching
        this.container.querySelectorAll('.tab-switch-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.switchTab(btn.dataset.tab);
            });
        });

        // Add layer buttons
        this.container.querySelectorAll('.add-layer-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.addLayer(btn.dataset.type);
            });
        });

        // Preset buttons
        this.container.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.loadPreset(btn.dataset.preset);
            });
        });

        // Clear layers
        this.container.querySelector('#clear-layers-btn').addEventListener('click', () => {
            this.clearArchitecture();
        });

        // Learning rate
        const lrSlider = this.container.querySelector('#create-lr');
        lrSlider.addEventListener('input', () => {
            const lr = Math.pow(10, parseFloat(lrSlider.value));
            this.container.querySelector('#lr-display').textContent = lr.toFixed(lr < 0.01 ? 4 : 3);
        });

        // Dataset change
        this.container.querySelector('#create-dataset').addEventListener('change', async (e) => {
            await this.dataLoader.loadDataset(e.target.value);
            this.updateDatasetInfo();
        });

        // Training buttons
        this.container.querySelector('#start-training-btn').addEventListener('click', () => {
            this.startTraining();
        });
        this.container.querySelector('#stop-training-btn').addEventListener('click', () => {
            this.stopTraining();
        });

        // Drawing controls
        this.container.querySelector('#clear-drawing-btn').addEventListener('click', () => {
            this.clearDrawing();
        });
        this.container.querySelector('#predict-btn').addEventListener('click', () => {
            this.predict();
        });
    }

    switchTab(tabId) {
        this.container.querySelectorAll('.tab-switch-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        this.container.querySelectorAll('.create-tab').forEach(tab => {
            tab.classList.toggle('active', tab.id === `create-${tabId}-tab`);
        });
        this.currentTab = tabId;
    }

    addLayer(type) {
        const defaultConfigs = {
            conv2d: { type: LAYER_TYPES.CONV2D, filters: 32, kernelSize: 3, activation: 'relu' },
            maxpool: { type: LAYER_TYPES.MAXPOOL, poolSize: 2 },
            flatten: { type: LAYER_TYPES.FLATTEN },
            dense: { type: LAYER_TYPES.DENSE, units: 64, activation: 'relu' },
            dropout: { type: LAYER_TYPES.DROPOUT, rate: 0.25 }
        };

        this.architecture.push({ ...defaultConfigs[type], id: Date.now() });
        this.renderArchitecture();
    }

    loadPreset(name) {
        if (name === 'simple') {
            this.architecture = [
                { type: LAYER_TYPES.CONV2D, filters: 32, kernelSize: 3, activation: 'relu', id: 1 },
                { type: LAYER_TYPES.MAXPOOL, poolSize: 2, id: 2 },
                { type: LAYER_TYPES.FLATTEN, id: 3 },
                { type: LAYER_TYPES.DENSE, units: 64, activation: 'relu', id: 4 },
                { type: LAYER_TYPES.DROPOUT, rate: 0.5, id: 5 }
            ];
        } else if (name === 'lenet') {
            this.architecture = [
                { type: LAYER_TYPES.CONV2D, filters: 6, kernelSize: 5, activation: 'relu', id: 1 },
                { type: LAYER_TYPES.MAXPOOL, poolSize: 2, id: 2 },
                { type: LAYER_TYPES.CONV2D, filters: 16, kernelSize: 5, activation: 'relu', id: 3 },
                { type: LAYER_TYPES.MAXPOOL, poolSize: 2, id: 4 },
                { type: LAYER_TYPES.FLATTEN, id: 5 },
                { type: LAYER_TYPES.DENSE, units: 120, activation: 'relu', id: 6 },
                { type: LAYER_TYPES.DENSE, units: 84, activation: 'relu', id: 7 }
            ];
        }
        this.renderArchitecture();
    }

    clearArchitecture() {
        this.architecture = [];
        this.renderArchitecture();
    }

    renderArchitecture() {
        const container = this.container.querySelector('#layers-container');
        const outputBlock = this.container.querySelector('#output-block');
        const layerCount = this.container.querySelector('#layer-count');

        if (this.architecture.length === 0) {
            container.innerHTML = '<div class="empty-state">Click layers to add them here</div>';
            outputBlock.style.display = 'none';
        } else {
            container.innerHTML = this.architecture.map((layer, idx) => `
                <div class="layer-block ${layer.type}" data-index="${idx}">
                    <div class="layer-connector">→</div>
                    <div class="layer-content">
                        <div class="layer-icon">${this.getLayerIcon(layer.type)}</div>
                        <div class="layer-info">
                            <div class="layer-name">${this.getLayerName(layer)}</div>
                            <div class="layer-params">${this.getLayerParams(layer)}</div>
                        </div>
                        <button class="remove-layer-btn" data-index="${idx}">×</button>
                    </div>
                </div>
            `).join('');
            outputBlock.style.display = 'flex';

            // Attach remove handlers
            container.querySelectorAll('.remove-layer-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.removeLayer(parseInt(btn.dataset.index));
                });
            });
        }

        layerCount.textContent = `${this.architecture.length} layers`;
    }

    getLayerIcon(type) {
        const icons = {
            [LAYER_TYPES.CONV2D]: '🔲',
            [LAYER_TYPES.MAXPOOL]: '📊',
            [LAYER_TYPES.FLATTEN]: '═══',
            [LAYER_TYPES.DENSE]: '●●●',
            [LAYER_TYPES.DROPOUT]: '⚫⚪'
        };
        return icons[type] || '?';
    }

    getLayerName(layer) {
        const names = {
            [LAYER_TYPES.CONV2D]: 'Conv2D',
            [LAYER_TYPES.MAXPOOL]: 'MaxPool',
            [LAYER_TYPES.FLATTEN]: 'Flatten',
            [LAYER_TYPES.DENSE]: 'Dense',
            [LAYER_TYPES.DROPOUT]: 'Dropout'
        };
        return names[layer.type] || layer.type;
    }

    getLayerParams(layer) {
        switch (layer.type) {
            case LAYER_TYPES.CONV2D:
                return `${layer.filters} filters, ${layer.kernelSize}×${layer.kernelSize}`;
            case LAYER_TYPES.MAXPOOL:
                return `${layer.poolSize}×${layer.poolSize}`;
            case LAYER_TYPES.DENSE:
                return `${layer.units} units`;
            case LAYER_TYPES.DROPOUT:
                return `${(layer.rate * 100).toFixed(0)}%`;
            default:
                return '';
        }
    }

    removeLayer(index) {
        this.architecture.splice(index, 1);
        this.renderArchitecture();
    }

    updateDatasetInfo() {
        const info = this.dataLoader.getDatasetInfo();
        if (info) {
            this.container.querySelector('#input-shape').textContent =
                `${info.inputShape[0]}×${info.inputShape[1]}×${info.inputShape[2]}`;
        }
    }

    initCharts() {
        const lossCanvas = this.container.querySelector('#create-loss-chart');
        const accCanvas = this.container.querySelector('#create-acc-chart');
        this.trainingCharts = new TrainingCharts(lossCanvas, accCanvas);
    }

    async startTraining() {
        if (this.architecture.length === 0) {
            alert('Add at least one layer!');
            return;
        }

        const hasFlatten = this.architecture.some(l => l.type === LAYER_TYPES.FLATTEN);
        if (!hasFlatten) {
            alert('Add a Flatten layer before Dense layers!');
            return;
        }

        const lr = Math.pow(10, parseFloat(this.container.querySelector('#create-lr').value));
        const epochs = parseInt(this.container.querySelector('#create-epochs').value);
        const batchSize = parseInt(this.container.querySelector('#create-batch').value);
        const datasetInfo = this.dataLoader.getDatasetInfo();

        try {
            this.setTrainingState(true);
            this.trainingCharts.reset();

            this.modelEngine.buildModel(this.architecture, datasetInfo.inputShape, datasetInfo.numClasses);
            this.modelEngine.compile(lr);

            const trainData = this.dataLoader.trainData;

            await this.modelEngine.train(trainData, { epochs, batchSize }, {
                onEpochEnd: async (epoch, logs, history) => {
                    this.container.querySelector('#stat-epoch').textContent = `${epoch + 1}/${epochs}`;
                    this.container.querySelector('#stat-loss').textContent = logs.loss.toFixed(4);
                    this.container.querySelector('#stat-accuracy').textContent = `${(logs.acc * 100).toFixed(1)}%`;
                    this.trainingCharts.update(history);
                }
            });

            this.isTrained = true;
            this.updateModelStatus();

        } catch (error) {
            console.error('Training error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            this.setTrainingState(false);
        }
    }

    stopTraining() {
        this.modelEngine.stopTraining();
    }

    setTrainingState(isTraining) {
        this.isTraining = isTraining;
        this.container.querySelector('#start-training-btn').style.display = isTraining ? 'none' : 'block';
        this.container.querySelector('#stop-training-btn').style.display = isTraining ? 'block' : 'none';
    }

    initDrawingCanvas() {
        const canvas = this.container.querySelector('#drawing-canvas');
        const ctx = canvas.getContext('2d');

        // Set up drawing
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });

        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);

        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            isDrawing = true;
            lastX = touch.clientX - rect.left;
            lastY = touch.clientY - rect.top;
        });

        canvas.addEventListener('touchmove', (e) => {
            if (!isDrawing) return;
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            lastX = x;
            lastY = y;
        });

        canvas.addEventListener('touchend', () => isDrawing = false);
    }

    clearDrawing() {
        const canvas = this.container.querySelector('#drawing-canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.container.querySelector('#pred-value').textContent = '—';
        this.container.querySelector('#confidence-bars').innerHTML = '';
    }

    predict() {
        if (!this.isTrained || !this.modelEngine.isReady()) {
            alert('Train your model first!');
            return;
        }

        const canvas = this.container.querySelector('#drawing-canvas');
        const datasetInfo = this.dataLoader.getDatasetInfo();
        const size = datasetInfo.inputShape[0];

        // Resize canvas to model input size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = size;
        tempCanvas.height = size;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, size, size);

        // Get image data
        const imageData = tempCtx.getImageData(0, 0, size, size);
        const pixels = imageData.data;

        // Convert to tensor
        const data = [];
        for (let i = 0; i < pixels.length; i += 4) {
            data.push(pixels[i] / 255); // Use red channel (grayscale)
        }

        const tensor = tf.tensor(data).reshape([1, size, size, 1]);
        const prediction = this.modelEngine.predict(tensor);
        const probs = prediction.dataSync();

        // Display results
        const maxIdx = probs.indexOf(Math.max(...probs));
        this.container.querySelector('#pred-value').textContent = maxIdx;

        // Render confidence bars
        const barsContainer = this.container.querySelector('#confidence-bars');
        barsContainer.innerHTML = Array.from(probs).map((prob, idx) => `
            <div class="confidence-bar ${idx === maxIdx ? 'predicted' : ''}">
                <span class="conf-label">${idx}</span>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width: ${(prob * 100).toFixed(1)}%"></div>
                </div>
                <span class="conf-value">${(prob * 100).toFixed(1)}%</span>
            </div>
        `).join('');

        tensor.dispose();
        prediction.dispose();
    }

    updateModelStatus() {
        const status = this.container.querySelector('#model-status');
        if (this.isTrained) {
            status.innerHTML = `<span class="status-icon">✅</span><span>Model trained! Draw a digit and click Predict</span>`;
            status.classList.add('ready');
        }
    }

    show() {
        this.container.style.display = 'block';
    }

    hide() {
        this.container.style.display = 'none';
    }

    dispose() {
        this.modelEngine.dispose();
    }
}
