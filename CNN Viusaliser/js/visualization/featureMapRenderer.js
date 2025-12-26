/**
 * CNN Visualizer - Feature Map Renderer
 * Visualizes intermediate layer activations
 */

export class FeatureMapRenderer {
    constructor(containerElement) {
        this.container = containerElement;
        this.section = document.getElementById('feature-maps-section');
    }

    render(activations, layerNames = []) {
        if (!activations || activations.length === 0) {
            this.hide();
            return;
        }

        this.show();
        this.container.innerHTML = '';

        for (const activation of activations) {
            const group = this.createFeatureMapGroup(activation);
            if (group) {
                this.container.appendChild(group);
            }
        }
    }

    createFeatureMapGroup(activation) {
        const { layerName, layerType, tensor, shape } = activation;

        // Only visualize 2D activations (height, width, channels)
        if (shape.length !== 4) return null;

        const [batch, height, width, channels] = shape;
        const numMapsToShow = Math.min(channels, 16); // Limit to 16 feature maps

        const group = document.createElement('div');
        group.className = 'feature-map-group';

        const title = document.createElement('h4');
        title.textContent = `${layerName} (${width}×${height}×${channels})`;
        group.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'feature-map-grid';

        // Get activation data
        const data = tensor.dataSync();

        // Determine canvas size based on feature map size
        const canvasSize = Math.max(20, Math.min(40, Math.floor(160 / Math.sqrt(numMapsToShow))));

        for (let c = 0; c < numMapsToShow; c++) {
            const canvas = document.createElement('canvas');
            canvas.width = canvasSize;
            canvas.height = canvasSize;
            canvas.title = `Filter ${c + 1}`;

            this.drawFeatureMap(canvas, data, width, height, channels, c);
            grid.appendChild(canvas);
        }

        group.appendChild(grid);
        return group;
    }

    drawFeatureMap(canvas, data, width, height, channels, channelIndex) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);

        // Find min/max for normalization
        let min = Infinity;
        let max = -Infinity;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * channels + channelIndex;
                const value = data[idx];
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }

        const range = max - min || 1;

        // Draw scaled feature map
        const scaleX = canvas.width / width;
        const scaleY = canvas.height / height;

        for (let cy = 0; cy < canvas.height; cy++) {
            for (let cx = 0; cx < canvas.width; cx++) {
                const x = Math.floor(cx / scaleX);
                const y = Math.floor(cy / scaleY);
                const idx = (y * width + x) * channels + channelIndex;
                const value = data[idx];

                // Normalize to 0-255
                const normalized = Math.floor(((value - min) / range) * 255);

                // Use viridis-like colormap
                const colors = this.getViridisColor(normalized / 255);

                const pixelIdx = (cy * canvas.width + cx) * 4;
                imageData.data[pixelIdx] = colors[0];
                imageData.data[pixelIdx + 1] = colors[1];
                imageData.data[pixelIdx + 2] = colors[2];
                imageData.data[pixelIdx + 3] = 255;
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }

    getViridisColor(t) {
        // Simplified viridis colormap
        const colors = [
            [68, 1, 84],      // 0.0
            [72, 40, 120],    // 0.2
            [62, 74, 137],    // 0.4
            [49, 104, 142],   // 0.5
            [38, 130, 142],   // 0.6
            [31, 158, 137],   // 0.7
            [53, 183, 121],   // 0.8
            [109, 205, 89],   // 0.9
            [180, 222, 44],   // 0.95
            [253, 231, 37]    // 1.0
        ];

        const idx = Math.min(Math.floor(t * (colors.length - 1)), colors.length - 2);
        const frac = t * (colors.length - 1) - idx;

        return [
            Math.floor(colors[idx][0] + frac * (colors[idx + 1][0] - colors[idx][0])),
            Math.floor(colors[idx][1] + frac * (colors[idx + 1][1] - colors[idx][1])),
            Math.floor(colors[idx][2] + frac * (colors[idx + 1][2] - colors[idx][2]))
        ];
    }

    show() {
        if (this.section) {
            this.section.style.display = 'block';
        }
    }

    hide() {
        if (this.section) {
            this.section.style.display = 'none';
        }
    }

    clear() {
        this.container.innerHTML = '';
        this.hide();
    }
}
