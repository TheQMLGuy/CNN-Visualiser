/**
 * CNN Platform - Explore Mode Controller
 * Deep exploration of pre-trained CNN models
 */

import { ModelEngine } from '../ml/modelEngine.js';
import { LAYER_TYPES } from '../utils/constants.js';

export class ExploreMode {
    constructor(container, dataLoader, onBack) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.onBack = onBack;
        this.modelEngine = new ModelEngine();

        this.isModelLoaded = false;
        this.isTraining = false;
        this.currentLayerIndex = 0;
        this.layerActivations = [];
        this.modelLayers = [];

        // LeNet-5 inspired architecture
        this.architecture = [
            { type: LAYER_TYPES.CONV2D, filters: 6, kernelSize: 5, activation: 'relu', name: 'Conv1' },
            { type: LAYER_TYPES.MAXPOOL, poolSize: 2, name: 'Pool1' },
            { type: LAYER_TYPES.CONV2D, filters: 16, kernelSize: 5, activation: 'relu', name: 'Conv2' },
            { type: LAYER_TYPES.MAXPOOL, poolSize: 2, name: 'Pool2' },
            { type: LAYER_TYPES.FLATTEN, name: 'Flatten' },
            { type: LAYER_TYPES.DENSE, units: 120, activation: 'relu', name: 'Dense1' },
            { type: LAYER_TYPES.DENSE, units: 84, activation: 'relu', name: 'Dense2' }
        ];

        this.render();
        this.attachEventListeners();
        this.initDrawingCanvas();
    }

    render() {
        this.container.innerHTML = `
            <div class="explore-mode">
                <header class="mode-topbar">
                    <button class="back-btn" id="explore-back-btn">← Back to Menu</button>
                    <h2>🔍 Explore Pre-trained LeNet-5</h2>
                    <div class="model-badge" id="model-badge">
                        <span class="badge-icon">⏳</span>
                        <span class="badge-text">Model not loaded</span>
                    </div>
                </header>

                <div class="explore-content">
                    <aside class="explore-input-panel">
                        <h3>Input Image</h3>
                        <div class="input-drawing-section">
                            <p class="input-hint">✏️ Draw a digit (0-9)</p>
                            <div class="drawing-wrapper">
                                <canvas id="explore-drawing-canvas" width="196" height="196"></canvas>
                            </div>
                            <div class="drawing-actions">
                                <button id="explore-clear-btn">🗑️ Clear</button>
                                <button id="explore-run-btn" class="primary">▶ Run Network</button>
                            </div>
                        </div>
                        
                        <div class="sample-images-section">
                            <h4>Or use sample images</h4>
                            <div class="sample-grid" id="explore-samples"></div>
                            <button id="load-samples-btn">Load Samples</button>
                        </div>

                        <div class="prediction-result" id="explore-prediction">
                            <div class="pred-header">Network Output</div>
                            <div class="pred-digit" id="explore-pred-digit">—</div>
                            <div class="pred-confidence" id="explore-pred-conf">Draw or select an image</div>
                        </div>
                    </aside>

                    <main class="explore-layers-panel">
                        <div class="layer-tabs" id="layer-tabs">
                            <button class="layer-tab" data-layer="input">
                                <span class="tab-icon">📥</span>
                                <span class="tab-name">Input</span>
                            </button>
                            <button class="layer-tab" data-layer="0">
                                <span class="tab-icon">🔲</span>
                                <span class="tab-name">Conv1</span>
                            </button>
                            <button class="layer-tab" data-layer="1">
                                <span class="tab-icon">📊</span>
                                <span class="tab-name">Pool1</span>
                            </button>
                            <button class="layer-tab" data-layer="2">
                                <span class="tab-icon">🔲</span>
                                <span class="tab-name">Conv2</span>
                            </button>
                            <button class="layer-tab" data-layer="3">
                                <span class="tab-icon">📊</span>
                                <span class="tab-name">Pool2</span>
                            </button>
                            <button class="layer-tab" data-layer="4">
                                <span class="tab-icon">═══</span>
                                <span class="tab-name">Flatten</span>
                            </button>
                            <button class="layer-tab" data-layer="5">
                                <span class="tab-icon">●●●</span>
                                <span class="tab-name">Dense1</span>
                            </button>
                            <button class="layer-tab" data-layer="6">
                                <span class="tab-icon">●●●</span>
                                <span class="tab-name">Dense2</span>
                            </button>
                            <button class="layer-tab" data-layer="output">
                                <span class="tab-icon">📤</span>
                                <span class="tab-name">Output</span>
                            </button>
                        </div>

                        <div class="layer-view" id="layer-view">
                            <div class="layer-info" id="layer-info">
                                <h3>Layer Details</h3>
                                <p>Select a layer tab to explore its activations</p>
                            </div>
                            <div class="layer-visualization" id="layer-viz">
                                <div class="empty-viz-state">
                                    <p>🎨 Draw an image and click "Run Network" to see activations</p>
                                </div>
                            </div>
                        </div>
                    </main>

                    <aside class="explore-info-panel">
                        <div class="neuron-details" id="neuron-details">
                            <h3>🔬 Neuron Inspector</h3>
                            <p class="help-text">Click on any neuron/feature map to inspect it</p>
                            
                            <div class="selected-neuron-info" id="selected-neuron-info" style="display: none;">
                                <div class="neuron-header">
                                    <span class="neuron-badge">Neuron <span id="neuron-id">-</span></span>
                                </div>
                                <div class="neuron-activation">
                                    <span class="label">Activation:</span>
                                    <span class="value" id="neuron-activation-value">-</span>
                                </div>
                                <div class="neuron-canvas-container">
                                    <canvas id="neuron-detail-canvas" width="100" height="100"></canvas>
                                </div>
                            </div>
                        </div>

                        <div class="model-controls">
                            <h3>⚙️ Model Status</h3>
                            <div class="status-item" id="status-training">
                                <span class="status-dot"></span>
                                <span class="status-text">Not trained</span>
                            </div>
                            <button id="train-model-btn" class="control-btn">
                                🏋️ Train Model (5 epochs)
                            </button>
                            <div class="training-progress" id="explore-training-progress" style="display: none;">
                                <div class="progress-bar">
                                    <div class="progress-fill" id="explore-progress-fill"></div>
                                </div>
                                <span class="progress-text" id="explore-progress-text">0%</span>
                            </div>
                        </div>

                        <div class="layer-legend">
                            <h3>📋 Layer Legend</h3>
                            <div class="legend-item">
                                <span class="legend-icon">🔲</span>
                                <span>Convolution - Feature extraction</span>
                            </div>
                            <div class="legend-item">
                                <span class="legend-icon">📊</span>
                                <span>Pooling - Downsampling</span>
                            </div>
                            <div class="legend-item">
                                <span class="legend-icon">═══</span>
                                <span>Flatten - 2D → 1D</span>
                            </div>
                            <div class="legend-item">
                                <span class="legend-icon">●●●</span>
                                <span>Dense - Classification</span>
                            </div>
                        </div>
                    </aside>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Back button
        this.container.querySelector('#explore-back-btn').addEventListener('click', () => {
            this.dispose();
            this.onBack();
        });

        // Drawing controls
        this.container.querySelector('#explore-clear-btn').addEventListener('click', () => {
            this.clearDrawing();
        });

        this.container.querySelector('#explore-run-btn').addEventListener('click', () => {
            this.runNetwork();
        });

        // Train button
        this.container.querySelector('#train-model-btn').addEventListener('click', () => {
            this.trainModel();
        });

        // Load samples
        this.container.querySelector('#load-samples-btn').addEventListener('click', () => {
            this.loadSampleImages();
        });

        // Layer tabs
        this.container.querySelectorAll('.layer-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                this.selectLayer(tab.dataset.layer);
            });
        });
    }

    initDrawingCanvas() {
        const canvas = this.container.querySelector('#explore-drawing-canvas');
        const ctx = canvas.getContext('2d');

        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 12;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        const startDraw = (x, y) => {
            isDrawing = true;
            lastX = x;
            lastY = y;
        };

        const draw = (x, y) => {
            if (!isDrawing) return;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            lastX = x;
            lastY = y;
        };

        const endDraw = () => {
            isDrawing = false;
        };

        canvas.addEventListener('mousedown', (e) => startDraw(e.offsetX, e.offsetY));
        canvas.addEventListener('mousemove', (e) => draw(e.offsetX, e.offsetY));
        canvas.addEventListener('mouseup', endDraw);
        canvas.addEventListener('mouseout', endDraw);

        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            startDraw(touch.clientX - rect.left, touch.clientY - rect.top);
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            draw(touch.clientX - rect.left, touch.clientY - rect.top);
        });
        canvas.addEventListener('touchend', endDraw);
    }

    clearDrawing() {
        const canvas = this.container.querySelector('#explore-drawing-canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        this.container.querySelector('#explore-pred-digit').textContent = '—';
        this.container.querySelector('#explore-pred-conf').textContent = 'Draw or select an image';
        this.clearLayerViz();
    }

    clearLayerViz() {
        this.container.querySelector('#layer-viz').innerHTML = `
            <div class="empty-viz-state">
                <p>🎨 Draw an image and click "Run Network" to see activations</p>
            </div>
        `;
    }

    async loadSampleImages() {
        if (!this.dataLoader.isLoaded) {
            await this.dataLoader.loadDataset('mnist');
        }

        const samples = this.dataLoader.getSampleImages(8);
        const grid = this.container.querySelector('#explore-samples');
        grid.innerHTML = '';

        // Clear old sample data (no dispose needed - they're just data objects now)
        this.sampleTensors = [];

        samples.forEach((sample, idx) => {
            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;
            canvas.className = 'sample-thumb';
            canvas.title = `Label: ${sample.label}`;

            // Keep a copy of the tensor data for later use
            const tensorData = sample.image.dataSync();
            this.sampleTensors.push({ data: Array.from(tensorData), label: sample.label });

            // Render image to small canvas
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(28, 28);

            for (let i = 0; i < tensorData.length; i++) {
                const val = Math.floor(tensorData[i] * 255);
                imageData.data[i * 4] = val;
                imageData.data[i * 4 + 1] = val;
                imageData.data[i * 4 + 2] = val;
                imageData.data[i * 4 + 3] = 255;
            }
            ctx.putImageData(imageData, 0, 0);

            const sampleIndex = idx;
            canvas.addEventListener('click', () => {
                this.loadSampleFromData(this.sampleTensors[sampleIndex].data);
            });

            grid.appendChild(canvas);
            sample.image.dispose();
        });
    }

    loadSampleFromData(data) {
        const drawCanvas = this.container.querySelector('#explore-drawing-canvas');
        const ctx = drawCanvas.getContext('2d');

        // Create temp canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(28, 28);

        for (let i = 0; i < data.length; i++) {
            const val = Math.floor(data[i] * 255);
            imageData.data[i * 4] = val;
            imageData.data[i * 4 + 1] = val;
            imageData.data[i * 4 + 2] = val;
            imageData.data[i * 4 + 3] = 255;
        }
        tempCtx.putImageData(imageData, 0, 0);

        // Scale up to drawing canvas
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, drawCanvas.width, drawCanvas.height);
    }

    loadSampleToDrawing(tensor) {
        const drawCanvas = this.container.querySelector('#explore-drawing-canvas');
        const ctx = drawCanvas.getContext('2d');

        // Clear and scale
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

        const data = tensor.dataSync();
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(28, 28);

        for (let i = 0; i < data.length; i++) {
            const val = Math.floor(data[i] * 255);
            imageData.data[i * 4] = val;
            imageData.data[i * 4 + 1] = val;
            imageData.data[i * 4 + 2] = val;
            imageData.data[i * 4 + 3] = 255;
        }
        tempCtx.putImageData(imageData, 0, 0);

        // Scale up to drawing canvas
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, drawCanvas.width, drawCanvas.height);
    }

    async trainModel() {
        if (this.isTraining) return;

        try {
            this.isTraining = true;
            const trainBtn = this.container.querySelector('#train-model-btn');
            trainBtn.disabled = true;
            trainBtn.textContent = '⏳ Training...';

            // Show progress
            const progressContainer = this.container.querySelector('#explore-training-progress');
            progressContainer.style.display = 'block';

            // Ensure dataset is loaded
            if (!this.dataLoader.isLoaded) {
                await this.dataLoader.loadDataset('mnist');
            }

            const datasetInfo = this.dataLoader.getDatasetInfo();

            // Build model
            this.modelEngine.buildModel(this.architecture, datasetInfo.inputShape, datasetInfo.numClasses);
            this.modelEngine.compile(0.01);

            const trainData = this.dataLoader.trainData;

            await this.modelEngine.train(trainData, { epochs: 5, batchSize: 64 }, {
                onEpochEnd: async (epoch, logs) => {
                    const progress = ((epoch + 1) / 5) * 100;
                    this.container.querySelector('#explore-progress-fill').style.width = `${progress}%`;
                    this.container.querySelector('#explore-progress-text').textContent =
                        `Epoch ${epoch + 1}/5 - Acc: ${(logs.acc * 100).toFixed(1)}%`;
                }
            });

            this.isModelLoaded = true;
            this.updateModelBadge(true);
            trainBtn.textContent = '✅ Model Trained';

            const statusEl = this.container.querySelector('#status-training');
            statusEl.querySelector('.status-dot').classList.add('ready');
            statusEl.querySelector('.status-text').textContent = 'Trained & Ready';

        } catch (error) {
            console.error('Training error:', error);
            alert(`Error: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    updateModelBadge(ready) {
        const badge = this.container.querySelector('#model-badge');
        if (ready) {
            badge.innerHTML = `<span class="badge-icon">✅</span><span class="badge-text">Model Ready</span>`;
            badge.classList.add('ready');
        }
    }

    async runNetwork() {
        if (!this.isModelLoaded || !this.modelEngine.isReady()) {
            alert('Please train the model first!');
            return;
        }

        const canvas = this.container.querySelector('#explore-drawing-canvas');

        // Prepare input tensor
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);

        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const data = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            data.push(imageData.data[i] / 255);
        }

        const inputTensor = tf.tensor(data).reshape([1, 28, 28, 1]);

        // Get prediction
        const prediction = this.modelEngine.predict(inputTensor);
        const probs = prediction.dataSync();
        const maxIdx = probs.indexOf(Math.max(...probs));
        const confidence = (probs[maxIdx] * 100).toFixed(1);

        this.container.querySelector('#explore-pred-digit').textContent = maxIdx;
        this.container.querySelector('#explore-pred-conf').textContent = `${confidence}% confidence`;

        // Get activations for all layers
        this.layerActivations = await this.modelEngine.getActivations(inputTensor);

        // Store input for visualization
        this.inputTensor = inputTensor;

        // Update current layer view
        this.selectLayer('input');

        prediction.dispose();
    }

    selectLayer(layerId) {
        // Update tab styling
        this.container.querySelectorAll('.layer-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.layer === layerId);
        });

        const layerInfo = this.container.querySelector('#layer-info');
        const layerViz = this.container.querySelector('#layer-viz');

        if (layerId === 'input') {
            layerInfo.innerHTML = `
                <h3>📥 Input Layer</h3>
                <p>Original image fed to the network (28×28 grayscale)</p>
            `;
            this.visualizeInput(layerViz);
        } else if (layerId === 'output') {
            layerInfo.innerHTML = `
                <h3>📤 Output Layer</h3>
                <p>Final classification (10 classes for digits 0-9)</p>
            `;
            this.visualizeOutput(layerViz);
        } else {
            const idx = parseInt(layerId);
            const layer = this.architecture[idx];
            layerInfo.innerHTML = `
                <h3>${this.getLayerIcon(layer.type)} ${layer.name}</h3>
                <p>${this.getLayerDescription(layer)}</p>
            `;
            this.visualizeLayer(layerViz, idx);
        }
    }

    getLayerIcon(type) {
        const icons = {
            [LAYER_TYPES.CONV2D]: '🔲',
            [LAYER_TYPES.MAXPOOL]: '📊',
            [LAYER_TYPES.FLATTEN]: '═══',
            [LAYER_TYPES.DENSE]: '●●●'
        };
        return icons[type] || '?';
    }

    getLayerDescription(layer) {
        switch (layer.type) {
            case LAYER_TYPES.CONV2D:
                return `${layer.filters} filters, ${layer.kernelSize}×${layer.kernelSize} kernels. Detects features like edges and patterns.`;
            case LAYER_TYPES.MAXPOOL:
                return `${layer.poolSize}×${layer.poolSize} pooling. Reduces spatial dimensions while keeping important features.`;
            case LAYER_TYPES.FLATTEN:
                return 'Converts 2D feature maps to 1D vector for dense layers.';
            case LAYER_TYPES.DENSE:
                return `${layer.units} neurons. Each neuron looks at all inputs from previous layer.`;
            default:
                return '';
        }
    }

    visualizeInput(container) {
        if (!this.inputTensor) {
            container.innerHTML = '<p class="viz-hint">Run the network to see input visualization</p>';
            return;
        }

        container.innerHTML = `
            <div class="input-viz">
                <canvas id="input-viz-canvas" width="196" height="196"></canvas>
                <p class="viz-caption">28×28 grayscale image</p>
            </div>
        `;

        const canvas = container.querySelector('#input-viz-canvas');
        const ctx = canvas.getContext('2d');
        const data = this.inputTensor.dataSync();

        const imageData = ctx.createImageData(28, 28);
        for (let i = 0; i < data.length; i++) {
            const val = Math.floor(data[i] * 255);
            imageData.data[i * 4] = val;
            imageData.data[i * 4 + 1] = val;
            imageData.data[i * 4 + 2] = val;
            imageData.data[i * 4 + 3] = 255;
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        tempCanvas.getContext('2d').putImageData(imageData, 0, 0);

        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, 196, 196);
    }

    visualizeOutput(container) {
        if (!this.layerActivations.length) {
            container.innerHTML = '<p class="viz-hint">Run the network to see output</p>';
            return;
        }

        const lastActivationObj = this.layerActivations[this.layerActivations.length - 1];
        if (!lastActivationObj) return;

        // Final dense layer before softmax - show as bar chart
        const prediction = this.modelEngine.predict(this.inputTensor);
        const probs = prediction.dataSync();
        const maxIdx = probs.indexOf(Math.max(...probs));

        container.innerHTML = `
            <div class="output-viz">
                <h4>Class Probabilities</h4>
                <div class="prob-bars">
                    ${Array.from(probs).map((p, i) => `
                        <div class="prob-bar ${i === maxIdx ? 'max' : ''}">
                            <span class="prob-label">${i}</span>
                            <div class="prob-fill" style="width: ${p * 100}%"></div>
                            <span class="prob-value">${(p * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        prediction.dispose();
    }

    visualizeLayer(container, layerIdx) {
        if (!this.layerActivations.length || !this.layerActivations[layerIdx]) {
            container.innerHTML = '<p class="viz-hint">Run the network to see layer activations</p>';
            return;
        }

        const activationObj = this.layerActivations[layerIdx];
        const tensor = activationObj.tensor;
        const layer = this.architecture[layerIdx];

        if (layer.type === LAYER_TYPES.CONV2D || layer.type === LAYER_TYPES.MAXPOOL) {
            // Show feature maps
            this.visualizeFeatureMaps(container, tensor, layer);
        } else if (layer.type === LAYER_TYPES.FLATTEN || layer.type === LAYER_TYPES.DENSE) {
            // Show neuron grid
            this.visualizeNeuronGrid(container, tensor, layer);
        }
    }

    visualizeFeatureMaps(container, activation, layer) {
        const shape = activation.shape;
        const numMaps = shape[3];
        const height = shape[1];
        const width = shape[2];

        container.innerHTML = `
            <div class="feature-maps-viz">
                <p class="viz-info">${numMaps} feature maps, each ${height}×${width} pixels</p>
                <div class="maps-grid" id="maps-grid"></div>
            </div>
        `;

        const grid = container.querySelector('#maps-grid');
        const data = activation.dataSync();

        for (let f = 0; f < numMaps; f++) {
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            canvas.className = 'feature-map-canvas';
            canvas.title = `Feature Map ${f + 1}`;

            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(width, height);

            // Find min/max for this feature map
            let min = Infinity, max = -Infinity;
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width * numMaps + x * numMaps + f;
                    min = Math.min(min, data[idx]);
                    max = Math.max(max, data[idx]);
                }
            }

            // Normalize and draw
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width * numMaps + x * numMaps + f;
                    const val = max > min ? (data[idx] - min) / (max - min) : 0;
                    const pixelIdx = (y * width + x) * 4;

                    // Use colorful mapping
                    const r = Math.floor(val * 255);
                    const g = Math.floor((1 - Math.abs(val - 0.5) * 2) * 255);
                    const b = Math.floor((1 - val) * 255);

                    imageData.data[pixelIdx] = r;
                    imageData.data[pixelIdx + 1] = g;
                    imageData.data[pixelIdx + 2] = b;
                    imageData.data[pixelIdx + 3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);

            // Click handler to inspect
            canvas.addEventListener('click', () => {
                this.inspectFeatureMap(f, activation, layer);
            });

            grid.appendChild(canvas);
        }
    }

    visualizeNeuronGrid(container, activation, layer) {
        const data = activation.dataSync();
        const numNeurons = data.length;

        // Calculate grid dimensions
        const cols = Math.ceil(Math.sqrt(numNeurons));
        const rows = Math.ceil(numNeurons / cols);

        container.innerHTML = `
            <div class="neuron-grid-viz">
                <p class="viz-info">${numNeurons} neurons visualized as ${rows}×${cols} grid</p>
                <p class="viz-hint">Click any neuron to inspect its activation</p>
                <div class="neuron-grid" id="neuron-grid" style="
                    display: grid; 
                    grid-template-columns: repeat(${cols}, 1fr);
                    gap: 2px;
                    max-width: ${cols * 20}px;
                "></div>
            </div>
        `;

        const grid = container.querySelector('#neuron-grid');

        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        for (let v of data) {
            min = Math.min(min, v);
            max = Math.max(max, v);
        }

        for (let i = 0; i < numNeurons; i++) {
            const cell = document.createElement('div');
            cell.className = 'neuron-cell';

            const normalized = max > min ? (data[i] - min) / (max - min) : 0;
            const hue = 240 - normalized * 240; // Blue (cold) to Red (hot)
            cell.style.backgroundColor = `hsl(${hue}, 70%, ${40 + normalized * 30}%)`;
            cell.style.width = '16px';
            cell.style.height = '16px';
            cell.title = `Neuron ${i}: ${data[i].toFixed(4)}`;

            cell.addEventListener('click', () => {
                this.inspectNeuron(i, data[i]);
            });

            grid.appendChild(cell);
        }
    }

    inspectFeatureMap(mapIdx, activation, layer) {
        const detailsPanel = this.container.querySelector('#selected-neuron-info');
        detailsPanel.style.display = 'block';

        this.container.querySelector('#neuron-id').textContent = `Feature Map ${mapIdx + 1}`;
        this.container.querySelector('#neuron-activation-value').textContent = 'See visualization →';

        // Draw larger version
        const canvas = this.container.querySelector('#neuron-detail-canvas');
        const ctx = canvas.getContext('2d');

        const shape = activation.shape;
        const data = activation.dataSync();
        const height = shape[1];
        const width = shape[2];
        const numMaps = shape[3];

        // Find min/max
        let min = Infinity, max = -Infinity;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width * numMaps + x * numMaps + mapIdx;
                min = Math.min(min, data[idx]);
                max = Math.max(max, data[idx]);
            }
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(width, height);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width * numMaps + x * numMaps + mapIdx;
                const val = max > min ? (data[idx] - min) / (max - min) : 0;
                const pixelIdx = (y * width + x) * 4;

                const intensity = Math.floor(val * 255);
                imageData.data[pixelIdx] = intensity;
                imageData.data[pixelIdx + 1] = intensity;
                imageData.data[pixelIdx + 2] = intensity;
                imageData.data[pixelIdx + 3] = 255;
            }
        }

        tempCtx.putImageData(imageData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, 100, 100);
    }

    inspectNeuron(idx, value) {
        const detailsPanel = this.container.querySelector('#selected-neuron-info');
        detailsPanel.style.display = 'block';

        this.container.querySelector('#neuron-id').textContent = idx;
        this.container.querySelector('#neuron-activation-value').textContent = value.toFixed(6);

        // Draw activation bar
        const canvas = this.container.querySelector('#neuron-detail-canvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, 100, 100);

        const barHeight = Math.min(value * 50, 90);
        ctx.fillStyle = value > 0 ? '#3fb950' : '#f85149';
        ctx.fillRect(25, 100 - barHeight, 50, barHeight);

        ctx.strokeStyle = '#8b949e';
        ctx.strokeRect(25, 10, 50, 80);
    }

    show() {
        this.container.style.display = 'block';
    }

    hide() {
        this.container.style.display = 'none';
    }

    dispose() {
        if (this.inputTensor) {
            this.inputTensor.dispose();
        }
        this.layerActivations.forEach(a => {
            if (a && a.dispose) a.dispose();
        });
        this.modelEngine.dispose();
    }
}
