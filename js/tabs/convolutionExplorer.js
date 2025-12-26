/**
 * CNN Visualizer - Convolution Explorer Tab
 * Interactive visualization of convolution operations
 */

import { KERNELS, KERNEL_CATEGORIES, getKernelList, getKernelsByCategory } from '../utils/kernels.js';
import { ACTIVATIONS, applyActivation, getActivationList, getActivationGraphPoints } from '../utils/activations.js';
import { ImageRenderer } from '../visualization/imageRenderer.js';

export class ConvolutionExplorer {
    constructor(container, dataLoader) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.currentImage = null;
        this.currentKernel = 'sobelX';
        this.currentActivation = 'relu';
        this.customKernel = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ];
        this.isCustomKernel = false;

        this.init();
    }

    init() {
        this.render();
        this.attachEventListeners();
        this.loadRandomImage();
    }

    render() {
        this.container.innerHTML = `
            <div class="conv-explorer">
                <div class="conv-explorer-left">
                    <div class="image-source-panel">
                        <h3>📷 Image Source</h3>
                        <div class="image-source-buttons">
                            <button class="source-btn active" id="random-image-btn">
                                <span class="btn-icon">🎲</span>
                                Random from Dataset
                            </button>
                            <label class="source-btn upload-btn">
                                <span class="btn-icon">📁</span>
                                Upload Image
                                <input type="file" id="image-upload" accept="image/*" hidden>
                            </label>
                        </div>
                        <div class="original-image-container">
                            <span class="image-label">Original Image</span>
                            <canvas id="original-image" width="128" height="128"></canvas>
                        </div>
                    </div>

                    <div class="kernel-panel">
                        <h3>🔲 Select Kernel</h3>
                        <div class="kernel-categories">
                            ${Object.entries(KERNEL_CATEGORIES).map(([key, cat]) => `
                                <button class="category-btn ${key === 'edge' ? 'active' : ''}" data-category="${key}">
                                    ${cat.icon} ${cat.name}
                                </button>
                            `).join('')}
                        </div>
                        <div class="kernel-grid" id="kernel-grid">
                            ${this.renderKernelButtons('edge')}
                        </div>
                    </div>
                </div>

                <div class="conv-explorer-center">
                    <div class="visualization-flow">
                        <div class="viz-step">
                            <h4>Original</h4>
                            <canvas id="viz-original" width="128" height="128"></canvas>
                        </div>
                        <div class="viz-arrow">→</div>
                        <div class="viz-step">
                            <h4>After Convolution</h4>
                            <canvas id="viz-convolved" width="128" height="128"></canvas>
                        </div>
                        <div class="viz-arrow">→</div>
                        <div class="viz-step">
                            <h4>After Activation</h4>
                            <canvas id="viz-activated" width="128" height="128"></canvas>
                        </div>
                    </div>

                    <div class="kernel-visualization">
                        <div class="kernel-editor">
                            <h4>Kernel Values</h4>
                            <div class="kernel-matrix" id="kernel-matrix">
                                ${this.renderKernelMatrix()}
                            </div>
                            <div class="kernel-actions">
                                <button id="reset-kernel" class="small-btn">Reset</button>
                                <button id="normalize-kernel" class="small-btn">Normalize</button>
                            </div>
                        </div>
                        <div class="kernel-info" id="kernel-info">
                            <h4 id="kernel-name">${KERNELS[this.currentKernel].name}</h4>
                            <p id="kernel-description">${KERNELS[this.currentKernel].description}</p>
                        </div>
                    </div>
                </div>

                <div class="conv-explorer-right">
                    <div class="activation-panel">
                        <h3>⚡ Activation Function</h3>
                        <div class="activation-selector">
                            ${getActivationList().map(act => `
                                <button class="activation-btn ${act.id === this.currentActivation ? 'active' : ''}" 
                                        data-activation="${act.id}"
                                        style="--act-color: ${act.color}">
                                    ${act.name}
                                </button>
                            `).join('')}
                        </div>
                        <div class="activation-info" id="activation-info">
                            <div class="activation-graph">
                                <canvas id="activation-graph" width="200" height="120"></canvas>
                            </div>
                            <div class="activation-formula">
                                <code id="activation-formula">${ACTIVATIONS[this.currentActivation].formula}</code>
                            </div>
                            <p id="activation-description">${ACTIVATIONS[this.currentActivation].description}</p>
                        </div>
                    </div>

                    <div class="statistics-panel">
                        <h3>📊 Statistics</h3>
                        <div class="stat-row">
                            <span class="stat-name">Original Range:</span>
                            <span class="stat-value" id="stat-original-range">—</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">After Conv:</span>
                            <span class="stat-value" id="stat-conv-range">—</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-name">After Activation:</span>
                            <span class="stat-value" id="stat-act-range">—</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Draw initial activation graph
        this.drawActivationGraph();
    }

    renderKernelButtons(category) {
        const kernels = getKernelsByCategory(category);
        return kernels.map(kernel => `
            <button class="kernel-btn ${kernel.id === this.currentKernel ? 'active' : ''}" 
                    data-kernel="${kernel.id}"
                    style="--kernel-color: ${kernel.color}">
                <span class="kernel-preview">${this.getKernelPreviewSVG(kernel.kernel)}</span>
                <span class="kernel-name">${kernel.name}</span>
            </button>
        `).join('');
    }

    getKernelPreviewSVG(kernel) {
        // Create a mini visualization of the kernel
        const size = kernel.length;
        const cellSize = 12;
        const svgSize = size * cellSize;

        let svg = `<svg width="${svgSize}" height="${svgSize}" viewBox="0 0 ${svgSize} ${svgSize}">`;

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const value = kernel[y][x];
                const normalizedValue = Math.min(1, Math.max(-1, value));
                const color = normalizedValue >= 0
                    ? `rgba(102, 126, 234, ${Math.abs(normalizedValue)})`
                    : `rgba(248, 81, 73, ${Math.abs(normalizedValue)})`;
                svg += `<rect x="${x * cellSize}" y="${y * cellSize}" width="${cellSize}" height="${cellSize}" fill="${color}" stroke="rgba(255,255,255,0.2)" stroke-width="0.5"/>`;
            }
        }
        svg += '</svg>';
        return svg;
    }

    renderKernelMatrix() {
        const kernel = this.isCustomKernel ? this.customKernel : KERNELS[this.currentKernel].kernel;
        let html = '';

        for (let y = 0; y < kernel.length; y++) {
            html += '<div class="kernel-row">';
            for (let x = 0; x < kernel[y].length; x++) {
                const value = kernel[y][x];
                const displayValue = Number.isInteger(value) ? value : value.toFixed(3);
                html += `
                    <input type="number" 
                           class="kernel-cell" 
                           data-y="${y}" 
                           data-x="${x}" 
                           value="${displayValue}" 
                           step="0.1">
                `;
            }
            html += '</div>';
        }
        return html;
    }

    attachEventListeners() {
        // Random image button
        this.container.querySelector('#random-image-btn').addEventListener('click', () => {
            this.loadRandomImage();
        });

        // Image upload
        this.container.querySelector('#image-upload').addEventListener('change', (e) => {
            this.handleImageUpload(e);
        });

        // Category buttons
        this.container.querySelectorAll('.category-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.container.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const category = btn.dataset.category;
                this.container.querySelector('#kernel-grid').innerHTML = this.renderKernelButtons(category);
                this.attachKernelButtonListeners();
            });
        });

        // Kernel buttons
        this.attachKernelButtonListeners();

        // Activation buttons
        this.container.querySelectorAll('.activation-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.container.querySelectorAll('.activation-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentActivation = btn.dataset.activation;
                this.updateActivationInfo();
                this.applyConvolution();
            });
        });

        // Kernel cell editing
        this.attachKernelCellListeners();

        // Reset kernel
        this.container.querySelector('#reset-kernel').addEventListener('click', () => {
            this.isCustomKernel = false;
            this.updateKernelMatrix();
            this.applyConvolution();
        });

        // Normalize kernel
        this.container.querySelector('#normalize-kernel').addEventListener('click', () => {
            this.normalizeKernel();
        });
    }

    attachKernelButtonListeners() {
        this.container.querySelectorAll('.kernel-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.container.querySelectorAll('.kernel-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentKernel = btn.dataset.kernel;
                this.isCustomKernel = false;
                this.updateKernelInfo();
                this.updateKernelMatrix();
                this.applyConvolution();
            });
        });
    }

    attachKernelCellListeners() {
        this.container.querySelectorAll('.kernel-cell').forEach(cell => {
            cell.addEventListener('input', (e) => {
                const y = parseInt(e.target.dataset.y);
                const x = parseInt(e.target.dataset.x);
                const value = parseFloat(e.target.value) || 0;

                if (!this.isCustomKernel) {
                    // Copy current kernel to custom
                    this.customKernel = KERNELS[this.currentKernel].kernel.map(row => [...row]);
                    this.isCustomKernel = true;
                }

                this.customKernel[y][x] = value;
                this.applyConvolution();
            });
        });
    }

    updateKernelInfo() {
        const kernel = KERNELS[this.currentKernel];
        this.container.querySelector('#kernel-name').textContent = kernel.name;
        this.container.querySelector('#kernel-description').textContent = kernel.description;
    }

    updateKernelMatrix() {
        this.container.querySelector('#kernel-matrix').innerHTML = this.renderKernelMatrix();
        this.attachKernelCellListeners();
    }

    updateActivationInfo() {
        const activation = ACTIVATIONS[this.currentActivation];
        this.container.querySelector('#activation-formula').textContent = activation.formula;
        this.container.querySelector('#activation-description').textContent = activation.description;
        this.drawActivationGraph();
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
        this.currentImage = sample.image.arraySync();

        // Render to original canvas
        const canvas = this.container.querySelector('#original-image');
        const vizCanvas = this.container.querySelector('#viz-original');

        ImageRenderer.renderScaled(canvas, sample.image, 128, 128);
        ImageRenderer.renderScaled(vizCanvas, sample.image, 128, 128);

        // Dispose tensor
        sample.image.dispose();

        // Apply convolution
        this.applyConvolution();
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Get dataset info for target size
                const datasetInfo = this.dataLoader.getDatasetInfo();
                const targetSize = datasetInfo.inputShape[0];

                // Create canvas to resize and extract pixel data
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = targetSize;
                tempCanvas.height = targetSize;
                const ctx = tempCanvas.getContext('2d');

                // Draw resized image
                ctx.drawImage(img, 0, 0, targetSize, targetSize);

                // Get pixel data
                const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
                const pixels = imageData.data;

                // Convert to grayscale array matching dataset format
                const channels = datasetInfo.inputShape[2];
                if (channels === 1) {
                    // Grayscale
                    this.currentImage = [];
                    for (let y = 0; y < targetSize; y++) {
                        const row = [];
                        for (let x = 0; x < targetSize; x++) {
                            const idx = (y * targetSize + x) * 4;
                            const gray = (pixels[idx] + pixels[idx + 1] + pixels[idx + 2]) / 3 / 255;
                            row.push([gray]);
                        }
                        this.currentImage.push(row);
                    }
                } else {
                    // Color
                    this.currentImage = [];
                    for (let y = 0; y < targetSize; y++) {
                        const row = [];
                        for (let x = 0; x < targetSize; x++) {
                            const idx = (y * targetSize + x) * 4;
                            row.push([
                                pixels[idx] / 255,
                                pixels[idx + 1] / 255,
                                pixels[idx + 2] / 255
                            ]);
                        }
                        this.currentImage.push(row);
                    }
                }

                // Render to canvases
                const tensor = tf.tensor(this.currentImage);
                const canvas = this.container.querySelector('#original-image');
                const vizCanvas = this.container.querySelector('#viz-original');

                ImageRenderer.renderScaled(canvas, tensor, 128, 128);
                ImageRenderer.renderScaled(vizCanvas, tensor, 128, 128);

                tensor.dispose();

                // Apply convolution
                this.applyConvolution();
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    applyConvolution() {
        if (!this.currentImage) return;

        const kernel = this.isCustomKernel ? this.customKernel : KERNELS[this.currentKernel].kernel;

        // Convert image to tensor
        const imageTensor = tf.tensor(this.currentImage);

        // Get dimensions
        const [height, width, channels] = imageTensor.shape;

        // For visualization, work with first channel if multiple
        let grayTensor = channels > 1
            ? imageTensor.slice([0, 0, 0], [-1, -1, 1]).squeeze()
            : imageTensor.squeeze();

        // Apply convolution
        const convolvedData = this.convolve2D(grayTensor.arraySync(), kernel);

        // Get statistics before activation
        const convMin = Math.min(...convolvedData.flat());
        const convMax = Math.max(...convolvedData.flat());

        // Apply activation
        const activatedData = convolvedData.map(row =>
            row.map(val => ACTIVATIONS[this.currentActivation].apply(val))
        );

        // Get statistics after activation
        const actMin = Math.min(...activatedData.flat());
        const actMax = Math.max(...activatedData.flat());

        // Normalize for display
        const normalizedConv = this.normalizeForDisplay(convolvedData);
        const normalizedAct = this.normalizeForDisplay(activatedData);

        // Render convolved image
        const convCanvas = this.container.querySelector('#viz-convolved');
        this.renderGrayscaleImage(convCanvas, normalizedConv, 128, 128);

        // Render activated image
        const actCanvas = this.container.querySelector('#viz-activated');
        this.renderGrayscaleImage(actCanvas, normalizedAct, 128, 128);

        // Update statistics
        const originalData = grayTensor.arraySync();
        const origMin = Math.min(...originalData.flat());
        const origMax = Math.max(...originalData.flat());

        this.container.querySelector('#stat-original-range').textContent =
            `[${origMin.toFixed(2)}, ${origMax.toFixed(2)}]`;
        this.container.querySelector('#stat-conv-range').textContent =
            `[${convMin.toFixed(2)}, ${convMax.toFixed(2)}]`;
        this.container.querySelector('#stat-act-range').textContent =
            `[${actMin.toFixed(2)}, ${actMax.toFixed(2)}]`;

        // Cleanup
        imageTensor.dispose();
        grayTensor.dispose();
    }

    convolve2D(image, kernel) {
        const imageHeight = image.length;
        const imageWidth = image[0].length;
        const kernelSize = kernel.length;
        const pad = Math.floor(kernelSize / 2);

        const result = [];

        for (let y = 0; y < imageHeight; y++) {
            const row = [];
            for (let x = 0; x < imageWidth; x++) {
                let sum = 0;

                for (let ky = 0; ky < kernelSize; ky++) {
                    for (let kx = 0; kx < kernelSize; kx++) {
                        const iy = y + ky - pad;
                        const ix = x + kx - pad;

                        // Handle boundary (zero padding)
                        if (iy >= 0 && iy < imageHeight && ix >= 0 && ix < imageWidth) {
                            sum += image[iy][ix] * kernel[ky][kx];
                        }
                    }
                }

                row.push(sum);
            }
            result.push(row);
        }

        return result;
    }

    normalizeForDisplay(data) {
        const flat = data.flat();
        const min = Math.min(...flat);
        const max = Math.max(...flat);
        const range = max - min || 1;

        return data.map(row => row.map(val => (val - min) / range));
    }

    renderGrayscaleImage(canvas, data, displayWidth, displayHeight) {
        const height = data.length;
        const width = data[0].length;

        // Create temporary canvas at original size
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(width, height);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const value = Math.floor(data[y][x] * 255);
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

    normalizeKernel() {
        const kernel = this.isCustomKernel ? this.customKernel : KERNELS[this.currentKernel].kernel;
        const sum = kernel.flat().reduce((a, b) => a + Math.abs(b), 0) || 1;

        if (!this.isCustomKernel) {
            this.customKernel = kernel.map(row => [...row]);
            this.isCustomKernel = true;
        }

        for (let y = 0; y < this.customKernel.length; y++) {
            for (let x = 0; x < this.customKernel[y].length; x++) {
                this.customKernel[y][x] /= sum;
            }
        }

        this.updateKernelMatrix();
        this.applyConvolution();
    }

    drawActivationGraph() {
        const canvas = this.container.querySelector('#activation-graph');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const activation = ACTIVATIONS[this.currentActivation];

        // Clear canvas
        ctx.fillStyle = '#21262d';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = 'rgba(139, 148, 158, 0.2)';
        ctx.lineWidth = 1;

        // Vertical center line (y-axis)
        ctx.beginPath();
        ctx.moveTo(width / 2, 0);
        ctx.lineTo(width / 2, height);
        ctx.stroke();

        // Horizontal center line (x-axis)
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Draw function
        const points = getActivationGraphPoints(this.currentActivation, 100);
        const { xRange, yRange } = activation.graph;

        ctx.strokeStyle = activation.color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        let started = false;
        for (const point of points) {
            const screenX = ((point.x - xRange[0]) / (xRange[1] - xRange[0])) * width;
            const screenY = height - ((point.y - yRange[0]) / (yRange[1] - yRange[0])) * height;

            if (screenY < 0 || screenY > height) continue;

            if (!started) {
                ctx.moveTo(screenX, screenY);
                started = true;
            } else {
                ctx.lineTo(screenX, screenY);
            }
        }
        ctx.stroke();

        // Add labels
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px Inter, sans-serif';
        ctx.fillText('x', width - 12, height / 2 + 12);
        ctx.fillText('y', width / 2 + 5, 12);
    }

    dispose() {
        // Cleanup if needed
    }
}
