/**
 * CNN Platform - Learn Mode Controller
 * Handles the component learning tabs
 */

import { ConvolutionExplorer } from '../tabs/convolutionExplorer.js';
import { PoolingExplorer } from '../tabs/poolingExplorer.js';
import { FlattenDenseExplorer } from '../tabs/flattenDenseExplorer.js';
import { DropoutExplorer } from '../tabs/dropoutExplorer.js';

export class LearnMode {
    constructor(container, dataLoader, onBack) {
        this.container = container;
        this.dataLoader = dataLoader;
        this.onBack = onBack;
        this.currentTab = 'convolution';
        this.tabs = {
            convolution: null,
            pooling: null,
            'flatten-dense': null,
            dropout: null
        };

        this.render();
        this.attachEventListeners();
        this.initTab('convolution');
    }

    render() {
        this.container.innerHTML = `
            <div class="learn-mode">
                <header class="mode-topbar">
                    <button class="back-btn" id="learn-back-btn">← Back to Menu</button>
                    <h2>📚 Learn CNN Components</h2>
                    <div class="topbar-spacer"></div>
                </header>

                <div class="learn-layout">
                    <nav class="learn-nav">
                        <div class="nav-header">
                            <h3>Components</h3>
                        </div>
                        <div class="nav-buttons">
                            <button class="nav-btn active" data-tab="convolution">
                                <span class="nav-icon">🔲</span>
                                <span class="nav-label">Convolution</span>
                                <span class="nav-desc">Kernels & Filters</span>
                            </button>
                            <button class="nav-btn" data-tab="pooling">
                                <span class="nav-icon">📊</span>
                                <span class="nav-label">Pooling</span>
                                <span class="nav-desc">Downsampling</span>
                            </button>
                            <button class="nav-btn" data-tab="flatten-dense">
                                <span class="nav-icon">═══</span>
                                <span class="nav-label">Flatten & Dense</span>
                                <span class="nav-desc">Fully Connected</span>
                            </button>
                            <button class="nav-btn" data-tab="dropout">
                                <span class="nav-icon">⚫⚪</span>
                                <span class="nav-label">Dropout</span>
                                <span class="nav-desc">Regularization</span>
                            </button>
                        </div>
                    </nav>

                    <main class="learn-content">
                        <div class="tab-panel active" id="learn-tab-convolution"></div>
                        <div class="tab-panel" id="learn-tab-pooling"></div>
                        <div class="tab-panel" id="learn-tab-flatten-dense"></div>
                        <div class="tab-panel" id="learn-tab-dropout"></div>
                    </main>
                </div>
            </div>
        `;
    }

    attachEventListeners() {
        // Back button
        this.container.querySelector('#learn-back-btn').addEventListener('click', () => {
            this.dispose();
            this.onBack();
        });

        // Tab navigation
        this.container.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                this.switchTab(tabId);
            });
        });
    }

    switchTab(tabId) {
        // Update nav buttons
        this.container.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });

        // Update panels
        this.container.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `learn-tab-${tabId}`);
        });

        this.currentTab = tabId;

        // Initialize tab if needed
        if (!this.tabs[tabId]) {
            this.initTab(tabId);
        } else if (this.tabs[tabId].onActivate) {
            this.tabs[tabId].onActivate();
        }
    }

    initTab(tabId) {
        const container = this.container.querySelector(`#learn-tab-${tabId}`);
        if (!container) return;

        switch (tabId) {
            case 'convolution':
                this.tabs.convolution = new ConvolutionExplorer(container, this.dataLoader);
                break;
            case 'pooling':
                this.tabs.pooling = new PoolingExplorer(container, this.dataLoader, {});
                break;
            case 'flatten-dense':
                this.tabs['flatten-dense'] = new FlattenDenseExplorer(container, this.dataLoader);
                break;
            case 'dropout':
                this.tabs.dropout = new DropoutExplorer(container);
                break;
        }
    }

    show() {
        this.container.style.display = 'block';
    }

    hide() {
        this.container.style.display = 'none';
    }

    dispose() {
        Object.values(this.tabs).forEach(tab => {
            if (tab && tab.dispose) {
                tab.dispose();
            }
        });
        this.tabs = {};
    }
}
