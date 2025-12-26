/**
 * CNN Platform - Mode Selector
 * Landing page with 3 mode selection cards
 */

export class ModeSelector {
    constructor(container, onModeSelect) {
        this.container = container;
        this.onModeSelect = onModeSelect;
        this.render();
        this.attachEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="mode-selector">
                <div class="mode-header">
                    <div class="mode-logo">
                        <svg width="64" height="64" viewBox="0 0 32 32" fill="none">
                            <rect x="2" y="2" width="28" height="28" rx="6" fill="url(#logo-gradient-lg)"/>
                            <path d="M8 12h4v8H8zM14 10h4v12h-4zM20 8h4v16h-4z" fill="white" opacity="0.9"/>
                            <defs>
                                <linearGradient id="logo-gradient-lg" x1="0" y1="0" x2="32" y2="32">
                                    <stop offset="0%" stop-color="#667eea"/>
                                    <stop offset="100%" stop-color="#764ba2"/>
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>
                    <h1>CNN Visualizer</h1>
                    <p class="tagline">Interactive Convolutional Neural Network Learning Platform</p>
                </div>

                <div class="mode-cards">
                    <div class="mode-card create-mode" data-mode="create">
                        <div class="mode-icon">🔧</div>
                        <h2>Create Your Own CNN</h2>
                        <p class="mode-description">
                            Build a custom neural network from scratch. Stack layers, 
                            train on datasets, and test with your own drawings.
                        </p>
                        <ul class="mode-features">
                            <li>Drag-and-drop layer builder</li>
                            <li>Real-time training visualization</li>
                            <li>Draw digits to test your model</li>
                        </ul>
                        <button class="mode-btn">Start Building →</button>
                    </div>

                    <div class="mode-card learn-mode" data-mode="learn">
                        <div class="mode-icon">📚</div>
                        <h2>Learn CNN Components</h2>
                        <p class="mode-description">
                            Understand how each part of a CNN works through 
                            interactive visualizations and experiments.
                        </p>
                        <ul class="mode-features">
                            <li>Convolution & kernels explained</li>
                            <li>Pooling operations visualized</li>
                            <li>Activation functions in action</li>
                        </ul>
                        <button class="mode-btn">Start Learning →</button>
                    </div>

                    <div class="mode-card explore-mode" data-mode="explore">
                        <div class="mode-icon">🔍</div>
                        <h2>Explore Pre-trained CNN</h2>
                        <p class="mode-description">
                            Dive deep into a trained LeNet-5 model. See how neurons 
                            activate and what features each layer detects.
                        </p>
                        <ul class="mode-features">
                            <li>Layer-by-layer exploration</li>
                            <li>Neuron activation grids</li>
                            <li>Draw inputs & watch the network</li>
                        </ul>
                        <button class="mode-btn">Start Exploring →</button>
                    </div>
                </div>

                <div class="mode-footer">
                    <p>Built with TensorFlow.js | Runs entirely in your browser</p>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        this.container.querySelectorAll('.mode-card').forEach(card => {
            card.addEventListener('click', () => {
                const mode = card.dataset.mode;
                this.animateSelection(card, mode);
            });
        });
    }

    animateSelection(card, mode) {
        // Add selection animation
        card.classList.add('selected');

        // Fade out other cards
        this.container.querySelectorAll('.mode-card').forEach(c => {
            if (c !== card) {
                c.classList.add('fading');
            }
        });

        // After animation, switch to selected mode
        setTimeout(() => {
            this.onModeSelect(mode);
        }, 400);
    }

    show() {
        this.container.style.display = 'block';
        this.container.classList.add('fade-in');
    }

    hide() {
        this.container.style.display = 'none';
    }
}
