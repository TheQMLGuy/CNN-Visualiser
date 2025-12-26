/**
 * CNN Platform - Main Application
 * 3-Mode Interactive CNN Learning Platform
 */

import { DataLoader } from './ml/dataLoader.js';
import { ModeSelector } from './modes/modeSelector.js';
import { LearnMode } from './modes/learnMode.js';
import { CreateMode } from './modes/createMode.js';
import { ExploreMode } from './modes/exploreMode.js';

class CNNPlatform {
    constructor() {
        this.dataLoader = new DataLoader();
        this.currentMode = null;
        this.modeInstances = {};

        this.init();
    }

    async init() {
        // Wait for TensorFlow.js
        await tf.ready();
        this.updateTFStatus(true);

        // Get container references
        this.landingContainer = document.getElementById('landing-container');
        this.modeContainer = document.getElementById('mode-container');

        // Initialize mode selector (landing page)
        this.modeSelector = new ModeSelector(this.landingContainer, (mode) => {
            this.switchToMode(mode);
        });

        // Preload MNIST dataset
        this.showStatus('Loading MNIST dataset...');
        await this.dataLoader.loadDataset('mnist');
        this.showStatus('Ready! Choose a learning mode.');

        console.log('CNN Platform initialized');
    }

    updateTFStatus(ready) {
        const statusEl = document.getElementById('tf-status');
        if (statusEl) {
            const statusText = statusEl.querySelector('.status-text');
            if (ready) {
                statusEl.classList.add('ready');
                statusText.textContent = 'TensorFlow.js Ready';
            }
        }
    }

    showStatus(text) {
        const statusEl = document.getElementById('app-status');
        if (statusEl) {
            statusEl.textContent = text;
        }
    }

    async switchToMode(mode) {
        // Hide landing page
        this.landingContainer.style.display = 'none';
        this.modeContainer.style.display = 'block';
        this.modeContainer.innerHTML = '';

        // Dispose previous mode
        if (this.currentMode && this.modeInstances[this.currentMode]) {
            if (this.modeInstances[this.currentMode].dispose) {
                this.modeInstances[this.currentMode].dispose();
            }
            this.modeInstances[this.currentMode] = null;
        }

        this.currentMode = mode;

        // Create new mode instance
        switch (mode) {
            case 'learn':
                this.modeInstances.learn = new LearnMode(
                    this.modeContainer,
                    this.dataLoader,
                    () => this.goHome()
                );
                break;
            case 'create':
                this.modeInstances.create = new CreateMode(
                    this.modeContainer,
                    this.dataLoader,
                    () => this.goHome()
                );
                break;
            case 'explore':
                this.modeInstances.explore = new ExploreMode(
                    this.modeContainer,
                    this.dataLoader,
                    () => this.goHome()
                );
                break;
        }
    }

    goHome() {
        // Show landing page
        this.landingContainer.style.display = 'block';
        this.modeContainer.style.display = 'none';

        // Re-render mode selector to reset animations
        this.modeSelector.render();
        this.modeSelector.attachEventListeners();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.cnnPlatform = new CNNPlatform();
});
