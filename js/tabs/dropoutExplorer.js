/**
 * CNN Visualizer - Dropout Explorer Tab
 * Interactive visualization of dropout regularization
 */

export class DropoutExplorer {
    constructor(container) {
        this.container = container;
        this.dropoutRate = 0.5;
        this.neurons = [];
        this.animationId = null;
        this.isAnimating = false;

        this.init();
    }

    init() {
        this.render();
        this.initializeNeurons();
        this.attachEventListeners();
        this.renderNetwork();
    }

    render() {
        this.container.innerHTML = `
            <div class="dropout-explorer">
                <div class="dropout-left">
                    <div class="dropout-controls">
                        <h3>⚫ Dropout Settings</h3>
                        
                        <div class="control-group">
                            <label>Dropout Rate: <span id="rate-value">50%</span></label>
                            <input type="range" id="dropout-rate" min="0" max="90" value="50" step="5">
                        </div>
                        
                        <div class="dropout-buttons">
                            <button id="apply-dropout" class="action-btn">
                                🎲 Apply Dropout
                            </button>
                            <button id="toggle-animation" class="action-btn secondary">
                                ▶ Auto Animate
                            </button>
                        </div>
                    </div>
                    
                    <div class="dropout-explanation">
                        <h4>What is Dropout?</h4>
                        <p>
                            Dropout randomly "drops" (deactivates) neurons during training
                            with probability <strong id="drop-prob">0.5</strong>.
                        </p>
                        <p class="highlight">
                            This prevents neurons from co-adapting too much,
                            forcing the network to learn more robust features.
                        </p>
                    </div>
                    
                    <div class="dropout-stats">
                        <h4>📊 Current State</h4>
                        <div class="stat-row">
                            <span>Total Neurons:</span>
                            <span id="total-neurons">24</span>
                        </div>
                        <div class="stat-row">
                            <span>Active Neurons:</span>
                            <span id="active-neurons" class="active-count">24</span>
                        </div>
                        <div class="stat-row">
                            <span>Dropped Neurons:</span>
                            <span id="dropped-neurons" class="dropped-count">0</span>
                        </div>
                    </div>
                </div>

                <div class="dropout-center">
                    <div class="network-visualization">
                        <h4>Neural Network Layer</h4>
                        <div class="network-canvas-wrapper">
                            <canvas id="dropout-canvas" width="600" height="400"></canvas>
                        </div>
                    </div>
                    
                    <div class="phase-indicator">
                        <div class="phase training">
                            <span class="phase-icon">🎓</span>
                            <span class="phase-label">Training Phase</span>
                            <span class="phase-desc">Dropout is ACTIVE - neurons randomly dropped</span>
                        </div>
                    </div>
                </div>

                <div class="dropout-right">
                    <div class="dropout-tips">
                        <h4>💡 Key Insights</h4>
                        <ul>
                            <li>
                                <strong>Training vs Inference:</strong>
                                Dropout only applies during training, not when making predictions.
                            </li>
                            <li>
                                <strong>Common Rates:</strong>
                                0.2-0.5 for hidden layers, 0.5 is a popular default.
                            </li>
                            <li>
                                <strong>Effect:</strong>
                                Creates an ensemble effect - like training many different networks.
                            </li>
                            <li>
                                <strong>Scaling:</strong>
                                During inference, outputs are scaled by (1 - dropout_rate) to compensate.
                            </li>
                        </ul>
                    </div>
                    
                    <div class="dropout-comparison">
                        <h4>🔄 With vs Without</h4>
                        <div class="comparison-item">
                            <span class="comp-label">Without Dropout:</span>
                            <span class="comp-desc">Risk of overfitting, neurons may overspecialize</span>
                        </div>
                        <div class="comparison-item">
                            <span class="comp-label">With Dropout:</span>
                            <span class="comp-desc">Better generalization, more robust features</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    initializeNeurons() {
        this.neurons = [];
        const rows = 4;
        const cols = 6;

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                this.neurons.push({
                    row,
                    col,
                    active: true,
                    x: 0,
                    y: 0
                });
            }
        }
    }

    attachEventListeners() {
        // Dropout rate slider
        const rateSlider = this.container.querySelector('#dropout-rate');
        rateSlider.addEventListener('input', (e) => {
            this.dropoutRate = parseInt(e.target.value) / 100;
            this.container.querySelector('#rate-value').textContent = `${e.target.value}%`;
            this.container.querySelector('#drop-prob').textContent = this.dropoutRate.toFixed(2);
        });

        // Apply dropout button
        this.container.querySelector('#apply-dropout').addEventListener('click', () => {
            this.applyDropout();
        });

        // Toggle animation
        this.container.querySelector('#toggle-animation').addEventListener('click', () => {
            this.toggleAnimation();
        });
    }

    applyDropout() {
        let droppedCount = 0;

        this.neurons.forEach(neuron => {
            neuron.active = Math.random() > this.dropoutRate;
            if (!neuron.active) droppedCount++;
        });

        // Update stats
        const total = this.neurons.length;
        const active = total - droppedCount;

        this.container.querySelector('#total-neurons').textContent = total;
        this.container.querySelector('#active-neurons').textContent = active;
        this.container.querySelector('#dropped-neurons').textContent = droppedCount;

        this.renderNetwork();
    }

    toggleAnimation() {
        const btn = this.container.querySelector('#toggle-animation');

        if (this.isAnimating) {
            this.isAnimating = false;
            if (this.animationId) {
                clearInterval(this.animationId);
                this.animationId = null;
            }
            btn.textContent = '▶ Auto Animate';
            btn.classList.remove('active');
        } else {
            this.isAnimating = true;
            btn.textContent = '⏹ Stop';
            btn.classList.add('active');

            this.animationId = setInterval(() => {
                this.applyDropout();
            }, 800);
        }
    }

    renderNetwork() {
        const canvas = this.container.querySelector('#dropout-canvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);

        const rows = 4;
        const cols = 6;
        const neuronRadius = 20;
        const horizontalSpacing = width / (cols + 1);
        const verticalSpacing = height / (rows + 1);

        // Calculate positions
        this.neurons.forEach((neuron, idx) => {
            neuron.x = horizontalSpacing * (neuron.col + 1);
            neuron.y = verticalSpacing * (neuron.row + 1);
        });

        // Draw connections (to next layer simulation)
        const nextLayerX = width - 50;
        const nextLayerY = height / 2;

        this.neurons.forEach(neuron => {
            if (neuron.active) {
                ctx.strokeStyle = 'rgba(102, 126, 234, 0.3)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(neuron.x + neuronRadius, neuron.y);
                ctx.lineTo(nextLayerX, nextLayerY);
                ctx.stroke();
            }
        });

        // Draw neurons
        this.neurons.forEach(neuron => {
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, neuronRadius, 0, Math.PI * 2);

            if (neuron.active) {
                // Active neuron
                ctx.fillStyle = '#667eea';
                ctx.fill();
                ctx.strokeStyle = '#8b9aed';
                ctx.lineWidth = 2;
                ctx.stroke();
            } else {
                // Dropped neuron
                ctx.fillStyle = 'rgba(248, 81, 73, 0.3)';
                ctx.fill();
                ctx.strokeStyle = '#f85149';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.setLineDash([]);

                // X mark
                ctx.strokeStyle = '#f85149';
                ctx.lineWidth = 2;
                const x = neuron.x;
                const y = neuron.y;
                const s = neuronRadius * 0.5;
                ctx.beginPath();
                ctx.moveTo(x - s, y - s);
                ctx.lineTo(x + s, y + s);
                ctx.moveTo(x + s, y - s);
                ctx.lineTo(x - s, y + s);
                ctx.stroke();
            }
        });

        // Draw output neuron indicator
        ctx.beginPath();
        ctx.arc(nextLayerX, nextLayerY, 25, 0, Math.PI * 2);
        ctx.fillStyle = '#3fb950';
        ctx.fill();
        ctx.strokeStyle = '#56d364';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Hidden Layer', width / 2, height - 15);
        ctx.fillText('(with Dropout)', width / 2, height - 30);

        ctx.fillStyle = '#f0f6fc';
        ctx.fillText('Next', nextLayerX, nextLayerY + 45);
        ctx.fillText('Layer', nextLayerX, nextLayerY + 60);
    }

    onActivate() {
        // Reset to all active when tab is activated
        this.neurons.forEach(neuron => neuron.active = true);
        this.container.querySelector('#active-neurons').textContent = this.neurons.length;
        this.container.querySelector('#dropped-neurons').textContent = '0';
        this.renderNetwork();
    }

    dispose() {
        if (this.animationId) {
            clearInterval(this.animationId);
        }
    }
}
