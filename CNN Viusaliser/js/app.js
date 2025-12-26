/**
 * CNN Visualizer - Main Application
 * Interactive browser-based CNN visualization tool with tabbed interface
 */

import { DATASETS } from './utils/constants.js';
import { TooltipManager } from './utils/tooltips.js';
import { DataLoader } from './ml/dataLoader.js';
import { ConvolutionExplorer } from './tabs/convolutionExplorer.js';
import { PoolingExplorer } from './tabs/poolingExplorer.js';
import { FlattenDenseExplorer } from './tabs/flattenDenseExplorer.js';
import { DropoutExplorer } from './tabs/dropoutExplorer.js';
import { CNNBuilder } from './tabs/cnnBuilder.js';

class CNNVisualizer {
    constructor() {
        this.dataLoader = new DataLoader();
        this.tooltipManager = null;

        this.currentDataset = 'mnist';
        this.currentTab = 'convolution';

        // Tab instances
        this.tabs = {
            convolution: null,
            pooling: null,
            'flatten-dense': null,
            dropout: null,
            'cnn-builder': null
        };

        // Shared state between tabs
        this.sharedState = {
            convolutionImage: null
        };

        this.init();
    }

    async init() {
        // Wait for TensorFlow.js to be ready
        await tf.ready();
        this.updateTFStatus(true);

        // Initialize tooltips
        this.initTooltips();

        // Setup event listeners
        this.initEventListeners();

        // Load initial dataset
        await this.loadDataset('mnist');

        // Initialize the first tab
        this.initTab('convolution');

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

    initEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tabId = btn.dataset.tab;
                this.switchTab(tabId);
            });
        });

        // Dataset selector
        document.getElementById('dataset').addEventListener('change', async (e) => {
            await this.loadDataset(e.target.value);
            // Reinitialize current tab with new data
            this.initTab(this.currentTab, true);
        });
    }

    switchTab(tabId) {
        // Update nav buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });

        // Update content visibility
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tabId}`);
        });

        this.currentTab = tabId;

        // Initialize tab if not already
        if (!this.tabs[tabId]) {
            this.initTab(tabId);
        } else {
            // Call onActivate if the tab has it
            if (this.tabs[tabId].onActivate) {
                this.tabs[tabId].onActivate();
            }
        }
    }

    initTab(tabId, force = false) {
        const container = document.getElementById(`tab-${tabId}`);
        if (!container) return;

        // Dispose existing tab if forcing reinit
        if (force && this.tabs[tabId]) {
            if (this.tabs[tabId].dispose) {
                this.tabs[tabId].dispose();
            }
            this.tabs[tabId] = null;
        }

        // Skip if already initialized
        if (this.tabs[tabId]) return;

        switch (tabId) {
            case 'convolution':
                this.tabs.convolution = new ConvolutionExplorer(container, this.dataLoader);
                break;
            case 'pooling':
                this.tabs.pooling = new PoolingExplorer(container, this.dataLoader, this.sharedState);
                break;
            case 'flatten-dense':
                this.tabs['flatten-dense'] = new FlattenDenseExplorer(container, this.dataLoader);
                break;
            case 'dropout':
                this.tabs.dropout = new DropoutExplorer(container);
                break;
            case 'cnn-builder':
                this.tabs['cnn-builder'] = new CNNBuilder(container, this.dataLoader);
                break;
        }

        this.showStatus(`${tabId.replace('-', ' ')} loaded`);
    }

    async loadDataset(datasetName) {
        this.currentDataset = datasetName;

        this.showStatus(`Loading ${datasetName.toUpperCase()} dataset...`);

        try {
            await this.dataLoader.loadDataset(datasetName);
            this.showStatus(`${datasetName.toUpperCase()} loaded successfully`);
        } catch (error) {
            console.error('Failed to load dataset:', error);
            this.showStatus(`Failed to load ${datasetName}`);
        }
    }

    showStatus(text) {
        const progressText = document.getElementById('progress-text');
        if (progressText) {
            progressText.textContent = text;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.cnnVisualizer = new CNNVisualizer();
});
