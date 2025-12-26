/**
 * CNN Visualizer - Architecture Builder
 * Manages the network architecture UI
 */

import { LAYER_TYPES, LAYER_CONFIGS, PRESETS, DATASETS } from '../utils/constants.js';

export class ArchitectureBuilder {
    constructor(containerElement, onArchitectureChange) {
        this.container = containerElement;
        this.onArchitectureChange = onArchitectureChange;
        this.layers = [];
        this.layerIdCounter = 0;
        this.emptyState = document.getElementById('empty-state');
        this.outputConnector = document.querySelector('.output-connector');
        this.outputLayer = document.querySelector('.output-layer');
    }

    addLayer(type, config = {}) {
        const layerConfig = LAYER_CONFIGS[type];
        if (!layerConfig) return;

        const layerId = ++this.layerIdCounter;
        const fullConfig = { ...layerConfig.defaults, ...config };

        const layer = {
            id: layerId,
            type: type,
            config: fullConfig
        };

        this.layers.push(layer);
        this.renderLayer(layer);
        this.updateUI();

        if (this.onArchitectureChange) {
            this.onArchitectureChange(this.layers);
        }

        return layer;
    }

    removeLayer(layerId) {
        const index = this.layers.findIndex(l => l.id === layerId);
        if (index === -1) return;

        // Remove from DOM
        const layerEl = this.container.querySelector(`[data-layer-id="${layerId}"]`);
        const connectorEl = layerEl?.previousElementSibling;

        if (connectorEl && connectorEl.classList.contains('layer-connector')) {
            connectorEl.remove();
        }
        layerEl?.remove();

        // Remove from array
        this.layers.splice(index, 1);
        this.updateUI();

        if (this.onArchitectureChange) {
            this.onArchitectureChange(this.layers);
        }
    }

    renderLayer(layer) {
        const layerConfig = LAYER_CONFIGS[layer.type];

        // Hide empty state
        if (this.emptyState) {
            this.emptyState.style.display = 'none';
        }

        // Create connector
        const connector = document.createElement('div');
        connector.className = 'layer-connector';
        connector.innerHTML = `
            <div class="connector-line"></div>
            <div class="connector-arrow"></div>
        `;

        // Create layer card
        const layerEl = document.createElement('div');
        layerEl.className = 'layer-card architecture-layer slide-in';
        layerEl.dataset.layerId = layer.id;
        layerEl.dataset.type = layer.type;
        layerEl.dataset.tooltip = layerConfig.tooltip;

        layerEl.innerHTML = `
            <button class="remove-layer" title="Remove layer">&times;</button>
            <div class="layer-visual">
                ${this.getLayerIcon(layer.type)}
            </div>
            <div class="layer-info">
                <span class="layer-title">${layerConfig.name}</span>
                <span class="layer-shape">${this.getLayerShapeText(layer)}</span>
            </div>
        `;

        // Add click handler for configuration
        layerEl.addEventListener('dblclick', (e) => {
            if (!e.target.classList.contains('remove-layer')) {
                this.openConfigModal(layer);
            }
        });

        // Add remove handler
        layerEl.querySelector('.remove-layer').addEventListener('click', () => {
            this.removeLayer(layer.id);
        });

        this.container.appendChild(connector);
        this.container.appendChild(layerEl);
    }

    getLayerIcon(type) {
        const icons = {
            [LAYER_TYPES.CONV2D]: `
                <svg viewBox="0 0 40 40" fill="none" width="32" height="32">
                    <rect x="4" y="4" width="14" height="14" rx="2" fill="var(--conv-color)" opacity="0.8"/>
                    <rect x="22" y="4" width="14" height="14" rx="2" fill="var(--conv-color)" opacity="0.6"/>
                    <rect x="4" y="22" width="14" height="14" rx="2" fill="var(--conv-color)" opacity="0.6"/>
                    <rect x="22" y="22" width="14" height="14" rx="2" fill="var(--conv-color)" opacity="0.4"/>
                </svg>
            `,
            [LAYER_TYPES.MAXPOOL]: `
                <svg viewBox="0 0 40 40" fill="none" width="32" height="32">
                    <rect x="8" y="8" width="24" height="24" rx="3" stroke="var(--pool-color)" stroke-width="2" fill="none"/>
                    <rect x="12" y="12" width="6" height="6" rx="1" fill="var(--pool-color)"/>
                </svg>
            `,
            [LAYER_TYPES.FLATTEN]: `
                <svg viewBox="0 0 40 40" fill="none" width="32" height="32">
                    <rect x="8" y="12" width="8" height="16" rx="2" fill="var(--flatten-color)" opacity="0.6"/>
                    <rect x="24" y="8" width="4" height="24" rx="2" fill="var(--flatten-color)"/>
                    <path d="M18 20h4" stroke="var(--flatten-color)" stroke-width="2" stroke-dasharray="2 2"/>
                </svg>
            `,
            [LAYER_TYPES.DENSE]: `
                <svg viewBox="0 0 40 40" fill="none" width="32" height="32">
                    <circle cx="12" cy="12" r="4" fill="var(--dense-color)"/>
                    <circle cx="12" cy="28" r="4" fill="var(--dense-color)"/>
                    <circle cx="28" cy="20" r="4" fill="var(--dense-color)"/>
                    <path d="M16 12l8 8M16 28l8-8" stroke="var(--dense-color)" stroke-width="1.5" opacity="0.5"/>
                </svg>
            `,
            [LAYER_TYPES.DROPOUT]: `
                <svg viewBox="0 0 40 40" fill="none" width="32" height="32">
                    <circle cx="12" cy="12" r="3" fill="var(--dropout-color)"/>
                    <circle cx="28" cy="12" r="3" fill="var(--dropout-color)" opacity="0.3"/>
                    <circle cx="12" cy="28" r="3" fill="var(--dropout-color)" opacity="0.3"/>
                    <circle cx="28" cy="28" r="3" fill="var(--dropout-color)"/>
                    <circle cx="20" cy="20" r="3" fill="var(--dropout-color)"/>
                </svg>
            `
        };
        return icons[type] || '';
    }

    getLayerShapeText(layer) {
        const config = layer.config;
        switch (layer.type) {
            case LAYER_TYPES.CONV2D:
                return `${config.filters}f, ${config.kernelSize}×${config.kernelSize}`;
            case LAYER_TYPES.MAXPOOL:
                return `${config.poolSize}×${config.poolSize}`;
            case LAYER_TYPES.FLATTEN:
                return '→ 1D';
            case LAYER_TYPES.DENSE:
                return `${config.units} units`;
            case LAYER_TYPES.DROPOUT:
                return `${Math.round(config.rate * 100)}%`;
            default:
                return '';
        }
    }

    updateUI() {
        // Show/hide empty state
        if (this.emptyState) {
            this.emptyState.style.display = this.layers.length === 0 ? 'flex' : 'none';
        }

        // Show/hide output layer
        const hasOutputLayer = this.layers.some(l =>
            l.type === LAYER_TYPES.DENSE || l.type === LAYER_TYPES.FLATTEN
        );

        if (this.outputConnector) {
            this.outputConnector.style.display = hasOutputLayer ? 'flex' : 'none';
        }
        if (this.outputLayer) {
            this.outputLayer.style.display = hasOutputLayer ? 'flex' : 'none';
        }
    }

    clear() {
        this.layers = [];

        // Remove all layer cards and connectors except empty state
        const elements = this.container.querySelectorAll('.architecture-layer, .layer-connector');
        elements.forEach(el => el.remove());

        this.updateUI();

        if (this.onArchitectureChange) {
            this.onArchitectureChange(this.layers);
        }
    }

    loadPreset(presetName) {
        const preset = PRESETS[presetName];
        if (!preset) return;

        this.clear();

        for (const layer of preset.layers) {
            this.addLayer(layer.type, layer.config);
        }
    }

    openConfigModal(layer) {
        const modal = document.getElementById('layer-modal');
        const title = document.getElementById('modal-title');
        const body = document.getElementById('modal-body');
        const saveBtn = document.getElementById('modal-save');
        const cancelBtn = document.getElementById('modal-cancel');
        const closeBtn = document.getElementById('modal-close');

        const layerConfig = LAYER_CONFIGS[layer.type];
        title.textContent = `Configure ${layerConfig.name}`;

        // Build form based on layer type
        body.innerHTML = this.getConfigForm(layer);

        modal.style.display = 'flex';

        const closeModal = () => {
            modal.style.display = 'none';
        };

        const saveConfig = () => {
            const form = body.querySelector('form');
            const formData = new FormData(form);

            for (const [key, value] of formData.entries()) {
                if (value !== '') {
                    const numValue = parseFloat(value);
                    layer.config[key] = isNaN(numValue) ? value : numValue;
                }
            }

            // Update layer display
            const layerEl = this.container.querySelector(`[data-layer-id="${layer.id}"]`);
            if (layerEl) {
                layerEl.querySelector('.layer-shape').textContent = this.getLayerShapeText(layer);
            }

            if (this.onArchitectureChange) {
                this.onArchitectureChange(this.layers);
            }

            closeModal();
        };

        saveBtn.onclick = saveConfig;
        cancelBtn.onclick = closeModal;
        closeBtn.onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
    }

    getConfigForm(layer) {
        const config = layer.config;
        let html = '<form>';

        switch (layer.type) {
            case LAYER_TYPES.CONV2D:
                html += `
                    <div class="control-group">
                        <label for="filters">Filters</label>
                        <input type="number" name="filters" value="${config.filters}" min="1" max="512">
                    </div>
                    <div class="control-group">
                        <label for="kernelSize">Kernel Size</label>
                        <input type="number" name="kernelSize" value="${config.kernelSize}" min="1" max="11">
                    </div>
                    <div class="control-group">
                        <label for="activation">Activation</label>
                        <select name="activation">
                            <option value="relu" ${config.activation === 'relu' ? 'selected' : ''}>ReLU</option>
                            <option value="sigmoid" ${config.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                            <option value="tanh" ${config.activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="padding">Padding</label>
                        <select name="padding">
                            <option value="same" ${config.padding === 'same' ? 'selected' : ''}>Same</option>
                            <option value="valid" ${config.padding === 'valid' ? 'selected' : ''}>Valid</option>
                        </select>
                    </div>
                `;
                break;

            case LAYER_TYPES.MAXPOOL:
                html += `
                    <div class="control-group">
                        <label for="poolSize">Pool Size</label>
                        <input type="number" name="poolSize" value="${config.poolSize}" min="2" max="4">
                    </div>
                `;
                break;

            case LAYER_TYPES.DENSE:
                html += `
                    <div class="control-group">
                        <label for="units">Units</label>
                        <input type="number" name="units" value="${config.units}" min="1" max="1024">
                    </div>
                    <div class="control-group">
                        <label for="activation">Activation</label>
                        <select name="activation">
                            <option value="relu" ${config.activation === 'relu' ? 'selected' : ''}>ReLU</option>
                            <option value="sigmoid" ${config.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
                            <option value="tanh" ${config.activation === 'tanh' ? 'selected' : ''}>Tanh</option>
                        </select>
                    </div>
                `;
                break;

            case LAYER_TYPES.DROPOUT:
                html += `
                    <div class="control-group">
                        <label for="rate">Dropout Rate</label>
                        <input type="number" name="rate" value="${config.rate}" min="0" max="0.9" step="0.05">
                    </div>
                `;
                break;

            case LAYER_TYPES.FLATTEN:
                html += '<p style="color: var(--text-secondary);">No configuration needed for Flatten layer.</p>';
                break;
        }

        html += '</form>';
        return html;
    }

    getArchitecture() {
        return this.layers.map(l => ({
            type: l.type,
            config: { ...l.config }
        }));
    }

    setDataset(datasetName) {
        const datasetInfo = DATASETS[datasetName];
        const inputShape = document.getElementById('input-shape');
        if (inputShape && datasetInfo) {
            inputShape.textContent = `${datasetInfo.inputShape[0]}×${datasetInfo.inputShape[1]}×${datasetInfo.inputShape[2]}`;
        }
    }
}
