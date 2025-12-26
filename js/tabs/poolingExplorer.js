/**
 * CNN Visualizer - Pooling Explorer Tab
 * Interactive visualization of pooling operations
 */

export class PoolingExplorer {
    constructor(container, dataLoader, sharedState) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.sharedState = sharedState;
        this.poolSize = 2;
        this.poolType = 'max';
        this.currentImage = null;

        this.init();
    }

    init() {
        this.render();
        this.attachEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="pool-explorer">
                <div class="pool-explorer-left">
                    <div class="pool-controls">
                        <h3>🎛️ Pooling Settings</h3>
                        
                        <div class="control-group">
                            <label>Pool Type</label>
                            <div class="button-group">
                                <button class="pool-type-btn active" data-type="max">Max Pool</button>
                                <button class="pool-type-btn" data-type="avg">Avg Pool</button>
                            </div>
                        </div>
                        
                        <div class="control-group">
                            <label>Pool Size</label>
                            <div class="button-group">
                                <button class="pool-size-btn active" data-size="2">2×2</button>
                                <button class="pool-size-btn" data-size="3">3×3</button>
                                <button class="pool-size-btn" data-size="4">4×4</button>
                            </div>
                        </div>
                        
                        <div class="pool-info-box">
                            <h4>📐 Dimension Change</h4>
                            <div class="dimension-display">
                                <span id="dim-before">28×28</span>
                                <span class="dim-arrow">→</span>
                                <span id="dim-after">14×14</span>
                            </div>
                            <p class="dim-reduction" id="dim-reduction">75% reduction</p>
                        </div>
                    </div>
                    
                    <div class="pool-explanation">
                        <h4 id="pool-title">Max Pooling</h4>
                        <p id="pool-description">
                            Max pooling takes the maximum value from each window. 
                            This helps the network become invariant to small translations 
                            and reduces the spatial size while keeping the most prominent features.
                        </p>
                    </div>
                </div>

                <div class="pool-explorer-center">
                    <div class="image-source-panel mini">
                        <button class="source-btn" id="pool-random-btn">
                            <span class="btn-icon">🎲</span> Random Image
                        </button>
                        <button class="source-btn" id="pool-from-conv-btn">
                            <span class="btn-icon">🔗</span> From Convolution Tab
                        </button>
                    </div>
                    
                    <div class="pool-visualization">
                        <div class="pool-image-container">
                            <h4>Input Image</h4>
                            <canvas id="pool-input" width="224" height="224"></canvas>
                            <div class="pool-overlay" id="pool-overlay"></div>
                        </div>
                        
                        <div class="pool-arrow">→</div>
                        
                        <div class="pool-image-container">
                            <h4>After Pooling</h4>
                            <canvas id="pool-output" width="224" height="224"></canvas>
                        </div>
                    </div>
                    
                    <div class="pool-detail-view">
                        <h4>🔍 Detailed View (Hover over input)</h4>
                        <div class="detail-grid" id="detail-grid">
                            <div class="detail-window">
                                <div class="detail-cells" id="detail-cells">
                                    <!-- Will be populated dynamically -->
                                </div>
                                <div class="detail-result">
                                    <span class="result-label">Result:</span>
                                    <span class="result-value" id="detail-result">—</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="pool-explorer-right">
                    <div class="pool-stats">
                        <h3>📊 Statistics</h3>
                        <div class="stat-row">
                            <span class="stat-name">Input Size:</span>
                            <span class="stat-value" id="pool-stat-input">0 × 0</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">Output Size:</span>
                            <span class="stat-value" id="pool-stat-output">0 × 0</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">Parameters:</span>
                            <span class="stat-value">0 (no learnable params)</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">Operations:</span>
                            <span class="stat-value" id="pool-stat-ops">0</span>
                        </div>
                    </div>
                    
                    <div class="pool-tips">
                        <h4>💡 Why Pooling?</h4>
                        <ul>
                            <li>Reduces spatial dimensions</li>
                            <li>Decreases computation needed</li>
                            <li>Provides translation invariance</li>
                            <li>Helps prevent overfitting</li>
                            <li>Extracts dominant features</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Pool type buttons
        this.container.querySelectorAll('.pool-type-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.container.querySelectorAll('.pool-type-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.poolType = btn.dataset.type;
                this.updatePoolExplanation();
                this.applyPooling();
            });
        });

        // Pool size buttons
        this.container.querySelectorAll('.pool-size-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.container.querySelectorAll('.pool-size-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.poolSize = parseInt(btn.dataset.size);
                this.updateDimensionDisplay();
                this.applyPooling();
            });
        });

        // Random image button
        this.container.querySelector('#pool-random-btn').addEventListener('click', () => {
            this.loadRandomImage();
        });

        // From convolution tab button
        this.container.querySelector('#pool-from-conv-btn').addEventListener('click', () => {
            if (this.sharedState && this.sharedState.convolutionImage) {
                this.currentImage = this.sharedState.convolutionImage;
                this.renderInputImage();
                this.applyPooling();
            }
        });
    }

    updatePoolExplanation() {
        const title = this.container.querySelector('#pool-title');
        const description = this.container.querySelector('#pool-description');

        if (this.poolType === 'max') {
            title.textContent = 'Max Pooling';
            description.textContent = 'Max pooling takes the maximum value from each window. This helps the network become invariant to small translations and reduces the spatial size while keeping the most prominent features.';
        } else {
            title.textContent = 'Average Pooling';
            description.textContent = 'Average pooling computes the mean of all values in each window. This creates a smoother downsampled image and can be useful when all pixels in a region contribute equally to the feature.';
        }
    }

    updateDimensionDisplay() {
        if (!this.currentImage) {
            this.container.querySelector('#dim-before').textContent = '28×28';
            this.container.querySelector('#dim-after').textContent = Math.floor(28 / this.poolSize) + '×' + Math.floor(28 / this.poolSize);
            return;
        }

        const inputSize = this.currentImage.length;
        const outputSize = Math.floor(inputSize / this.poolSize);
        const reduction = Math.round((1 - (outputSize * outputSize) / (inputSize * inputSize)) * 100);

        this.container.querySelector('#dim-before').textContent = `${inputSize}×${inputSize}`;
        this.container.querySelector('#dim-after').textContent = `${outputSize}×${outputSize}`;
        this.container.querySelector('#dim-reduction').textContent = `${reduction}% reduction`;

        this.container.querySelector('#pool-stat-input').textContent = `${inputSize} × ${inputSize}`;
        this.container.querySelector('#pool-stat-output').textContent = `${outputSize} × ${outputSize}`;
        this.container.querySelector('#pool-stat-ops').textContent = `${outputSize * outputSize * this.poolSize * this.poolSize}`;
    }

    loadRandomImage() {
        if (!this.dataLoader || !this.dataLoader.trainData) {
            console.warn('Data not loaded yet');
            return;
        }

        const totalImages = this.dataLoader.trainData.images.shape[0];
        const randomIndex = Math.floor(Math.random() * totalImages);

        const sample = this.dataLoader.getImage(randomIndex);
        if (!sample) return;

        // Store the image data
        const imageArray = sample.image.arraySync();

        // Handle different shapes
        if (imageArray[0][0].length) {
            // Has channels
            this.currentImage = imageArray.map(row => row.map(pixel => pixel[0]));
        } else {
            this.currentImage = imageArray;
        }

        sample.image.dispose();

        this.renderInputImage();
        this.updateDimensionDisplay();
        this.applyPooling();
    }

    renderInputImage() {
        if (!this.currentImage) return;

        const canvas = this.container.querySelector('#pool-input');
        this.renderGrayscaleToCanvas(canvas, this.currentImage, 224, 224);
    }

    applyPooling() {
        if (!this.currentImage) return;

        const inputSize = this.currentImage.length;
        const outputSize = Math.floor(inputSize / this.poolSize);
        const pooled = [];

        for (let y = 0; y < outputSize; y++) {
            const row = [];
            for (let x = 0; x < outputSize; x++) {
                // Get pool window
                const values = [];
                for (let py = 0; py < this.poolSize; py++) {
                    for (let px = 0; px < this.poolSize; px++) {
                        const iy = y * this.poolSize + py;
                        const ix = x * this.poolSize + px;
                        if (iy < inputSize && ix < inputSize) {
                            values.push(this.currentImage[iy][ix]);
                        }
                    }
                }

                // Apply pooling
                if (this.poolType === 'max') {
                    row.push(Math.max(...values));
                } else {
                    row.push(values.reduce((a, b) => a + b, 0) / values.length);
                }
            }
            pooled.push(row);
        }

        // Render output
        const canvas = this.container.querySelector('#pool-output');
        this.renderGrayscaleToCanvas(canvas, pooled, 224, 224);
    }

    renderGrayscaleToCanvas(canvas, data, displayWidth, displayHeight) {
        const height = data.length;
        const width = data[0].length;

        // Create temporary canvas at original size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(width, height);

        // Find min/max for normalization
        const flat = data.flat();
        const min = Math.min(...flat);
        const max = Math.max(...flat);
        const range = max - min || 1;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const value = Math.floor(((data[y][x] - min) / range) * 255);
                imageData.data[idx] = value;
                imageData.data[idx + 1] = value;
                imageData.data[idx + 2] = value;
                imageData.data[idx + 3] = 255;
            }
        }

        tempCtx.putImageData(imageData, 0, 0);

        // Scale to display canvas
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas, 0, 0, displayWidth, displayHeight);
    }

    onActivate() {
        // Called when tab is activated
        if (!this.currentImage) {
            this.loadRandomImage();
        }
    }

    dispose() {
        // Cleanup if needed
    }
}
