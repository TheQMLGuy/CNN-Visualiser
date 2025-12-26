/**
 * CNN Visualizer - Flatten & Dense Explorer Tab
 * Interactive visualization of flatten and dense layer operations
 */

export class FlattenDenseExplorer {
    constructor(container, dataLoader) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.currentImage = null;
        this.denseUnits = 10;
        this.weights = null;
        this.animationId = null;

        this.init();
    }

    init() {
        this.render();
        this.attachEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="flatten-dense-explorer">
                <div class="flatten-section">
                    <h3>═══ Flatten Layer</h3>
                    <p class="section-description">
                        Flatten converts a 2D feature map into a 1D vector, preparing it for dense layers.
                    </p>
                    
                    <div class="flatten-visualization">
                        <div class="flatten-input">
                            <h4>2D Feature Map</h4>
                            <div class="feature-map-grid" id="feature-map-grid">
                                <!-- 8x8 grid will be rendered here -->
                            </div>
                            <span class="shape-label" id="flatten-input-shape">8 × 8 = 64 values</span>
                        </div>
                        
                        <div class="flatten-animation">
                            <button id="animate-flatten-btn" class="animate-btn">
                                ▶ Animate Flatten
                            </button>
                        </div>
                        
                        <div class="flatten-output">
                            <h4>1D Vector</h4>
                            <div class="vector-display" id="vector-display">
                                <!-- Vector will be rendered here -->
                            </div>
                            <span class="shape-label" id="flatten-output-shape">64 values</span>
                        </div>
                    </div>
                </div>

                <div class="dense-section">
                    <h3>● Dense Layer (Fully Connected)</h3>
                    <p class="section-description">
                        Each neuron in a dense layer connects to every neuron in the previous layer,
                        learning weighted combinations of features.
                    </p>
                    
                    <div class="dense-controls">
                        <div class="control-group">
                            <label>Output Units</label>
                            <input type="range" id="dense-units" min="2" max="16" value="10">
                            <span id="dense-units-value">10</span>
                        </div>
                        <button id="randomize-weights" class="small-btn">🎲 Randomize Weights</button>
                    </div>
                    
                    <div class="dense-visualization">
                        <div class="dense-layer-view">
                            <canvas id="dense-canvas" width="600" height="300"></canvas>
                        </div>
                        
                        <div class="dense-info">
                            <div class="info-card">
                                <h5>Parameters</h5>
                                <p id="dense-params">640 weights + 10 biases = 650</p>
                            </div>
                            <div class="info-card">
                                <h5>Formula</h5>
                                <code>output = activation(input · W + b)</code>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="connection-detail">
                    <h4>🔍 Weight Inspection</h4>
                    <p class="hint">Hover over connections to see weight values</p>
                    <div class="weight-info" id="weight-info">
                        <span class="weight-label">Weight:</span>
                        <span class="weight-value" id="weight-value">—</span>
                    </div>
                </div>
            </div>
        `;

        this.initializeFeatureMap();
        this.initializeWeights();
        this.renderDenseLayer();
    }

    attachEventListeners() {
        // Dense units slider
        const unitsSlider = this.container.querySelector('#dense-units');
        unitsSlider.addEventListener('input', (e) => {
            this.denseUnits = parseInt(e.target.value);
            this.container.querySelector('#dense-units-value').textContent = this.denseUnits;
            this.initializeWeights();
            this.updateParamsDisplay();
            this.renderDenseLayer();
        });

        // Randomize weights
        this.container.querySelector('#randomize-weights').addEventListener('click', () => {
            this.initializeWeights();
            this.renderDenseLayer();
        });

        // Animate flatten
        this.container.querySelector('#animate-flatten-btn').addEventListener('click', () => {
            this.animateFlatten();
        });
    }

    initializeFeatureMap() {
        const grid = this.container.querySelector('#feature-map-grid');
        const vector = this.container.querySelector('#vector-display');
        const size = 8;

        // Create random feature map values
        this.featureMap = [];
        let html = '';

        for (let y = 0; y < size; y++) {
            const row = [];
            for (let x = 0; x < size; x++) {
                const value = Math.random();
                row.push(value);
                const color = Math.floor(value * 255);
                html += `<div class="grid-cell" data-idx="${y * size + x}" style="background: rgb(${color}, ${color}, ${color})"></div>`;
            }
            this.featureMap.push(row);
        }
        grid.innerHTML = html;

        // Create vector display
        const flat = this.featureMap.flat();
        let vectorHtml = '';
        for (let i = 0; i < flat.length; i++) {
            const color = Math.floor(flat[i] * 255);
            vectorHtml += `<div class="vector-cell" data-idx="${i}" style="background: rgb(${color}, ${color}, ${color})"></div>`;
        }
        vector.innerHTML = vectorHtml;
    }

    initializeWeights() {
        const inputSize = 64; // 8x8 flattened
        this.weights = [];

        for (let o = 0; o < this.denseUnits; o++) {
            const neuronWeights = [];
            for (let i = 0; i < inputSize; i++) {
                neuronWeights.push((Math.random() - 0.5) * 2);
            }
            this.weights.push(neuronWeights);
        }

        // Initialize biases
        this.biases = [];
        for (let o = 0; o < this.denseUnits; o++) {
            this.biases.push((Math.random() - 0.5) * 0.1);
        }
    }

    updateParamsDisplay() {
        const inputSize = 64;
        const weights = inputSize * this.denseUnits;
        const biases = this.denseUnits;
        this.container.querySelector('#dense-params').textContent =
            `${weights} weights + ${biases} biases = ${weights + biases}`;
    }

    animateFlatten() {
        const gridCells = this.container.querySelectorAll('.grid-cell');
        const vectorCells = this.container.querySelectorAll('.vector-cell');

        // Reset all
        gridCells.forEach(cell => cell.classList.remove('highlight', 'done'));
        vectorCells.forEach(cell => cell.classList.remove('highlight', 'done'));

        let idx = 0;
        const animate = () => {
            if (idx >= gridCells.length) {
                // Animation complete
                gridCells.forEach(cell => cell.classList.add('done'));
                vectorCells.forEach(cell => cell.classList.add('done'));
                return;
            }

            // Remove previous highlight
            if (idx > 0) {
                gridCells[idx - 1].classList.remove('highlight');
                gridCells[idx - 1].classList.add('done');
                vectorCells[idx - 1].classList.remove('highlight');
                vectorCells[idx - 1].classList.add('done');
            }

            // Highlight current
            gridCells[idx].classList.add('highlight');
            vectorCells[idx].classList.add('highlight');

            idx++;
            this.animationId = setTimeout(animate, 50);
        };

        animate();
    }

    renderDenseLayer() {
        const canvas = this.container.querySelector('#dense-canvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear
        ctx.fillStyle = '#161b22';
        ctx.fillRect(0, 0, width, height);

        const inputSize = 16; // Show simplified version
        const inputSpacing = height / (inputSize + 1);
        const outputSpacing = height / (this.denseUnits + 1);
        const inputX = 80;
        const outputX = width - 80;

        // Draw connections
        for (let i = 0; i < inputSize; i++) {
            const y1 = inputSpacing * (i + 1);

            for (let o = 0; o < this.denseUnits; o++) {
                const y2 = outputSpacing * (o + 1);
                const weightIdx = Math.floor(i * 64 / inputSize);
                const weight = this.weights[o][weightIdx];

                // Color based on weight sign and magnitude
                const absWeight = Math.abs(weight);
                const alpha = Math.min(1, absWeight * 0.5 + 0.1);

                if (weight >= 0) {
                    ctx.strokeStyle = `rgba(102, 126, 234, ${alpha})`;
                } else {
                    ctx.strokeStyle = `rgba(248, 81, 73, ${alpha})`;
                }

                ctx.lineWidth = absWeight * 2 + 0.5;
                ctx.beginPath();
                ctx.moveTo(inputX + 15, y1);
                ctx.lineTo(outputX - 15, y2);
                ctx.stroke();
            }
        }

        // Draw input neurons
        ctx.fillStyle = '#3fb950';
        for (let i = 0; i < inputSize; i++) {
            const y = inputSpacing * (i + 1);
            ctx.beginPath();
            ctx.arc(inputX, y, 8, 0, Math.PI * 2);
            ctx.fill();
        }

        // Input label
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Input', inputX, height - 10);
        ctx.fillText('(64 values)', inputX, height - 25);

        // Draw output neurons
        ctx.fillStyle = '#667eea';
        for (let o = 0; o < this.denseUnits; o++) {
            const y = outputSpacing * (o + 1);
            ctx.beginPath();
            ctx.arc(outputX, y, 10, 0, Math.PI * 2);
            ctx.fill();

            // Label
            ctx.fillStyle = '#f0f6fc';
            ctx.font = '10px Inter, sans-serif';
            ctx.fillText(o.toString(), outputX, y + 4);
            ctx.fillStyle = '#667eea';
        }

        // Output label
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText('Output', outputX, height - 10);
        ctx.fillText(`(${this.denseUnits} units)`, outputX, height - 25);

        // Legend
        ctx.fillStyle = '#8b949e';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('🔵 Positive weights', 10, 20);
        ctx.fillText('🔴 Negative weights', 10, 38);
        ctx.fillText('Line thickness = weight magnitude', 10, 56);
    }

    onActivate() {
        this.updateParamsDisplay();
    }

    dispose() {
        if (this.animationId) {
            clearTimeout(this.animationId);
        }
    }
}
