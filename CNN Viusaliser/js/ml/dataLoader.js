/**
 * CNN Visualizer - Data Loader
 * Handles MNIST and CIFAR-10 dataset loading
 */

import { DATASETS } from '../utils/constants.js';

export class DataLoader {
    constructor() {
        this.currentDataset = 'mnist';
        this.trainData = null;
        this.testData = null;
        this.isLoading = false;
    }

    async loadDataset(datasetName) {
        if (this.isLoading) return null;

        this.isLoading = true;
        this.currentDataset = datasetName;

        try {
            if (datasetName === 'mnist') {
                return await this.loadMNIST();
            } else if (datasetName === 'cifar10') {
                return await this.loadCIFAR10();
            }
        } finally {
            this.isLoading = false;
        }
    }

    async loadMNIST() {
        // Use TensorFlow.js built-in MNIST data
        const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
        const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

        const [imgResponse, labelsResponse] = await Promise.all([
            fetch(MNIST_IMAGES_SPRITE_PATH),
            fetch(MNIST_LABELS_PATH)
        ]);

        const imgBlob = await imgResponse.blob();
        const labelsArrayBuffer = await labelsResponse.arrayBuffer();
        const labels = new Uint8Array(labelsArrayBuffer);

        // Load image sprite
        const img = await this.loadImage(imgBlob);

        // Create canvas to extract pixel data
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const datasetSize = 65000;
        const imageSize = 28;

        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const numImages = img.height;

        // Extract images (use subset for faster training)
        const trainSize = 10000;
        const testSize = 1000;

        const trainImages = new Float32Array(trainSize * imageSize * imageSize);
        const trainLabels = new Uint8Array(trainSize);
        const testImages = new Float32Array(testSize * imageSize * imageSize);
        const testLabels = new Uint8Array(testSize);

        for (let i = 0; i < trainSize; i++) {
            const row = i;
            for (let j = 0; j < imageSize * imageSize; j++) {
                const x = j % imageSize;
                const y = Math.floor(j / imageSize);
                const srcIdx = (row * imageSize + y) * canvas.width * 4 + x * 4;
                trainImages[i * imageSize * imageSize + j] = imageData.data[srcIdx] / 255;
            }
            trainLabels[i] = labels[i];
        }

        for (let i = 0; i < testSize; i++) {
            const row = trainSize + i;
            for (let j = 0; j < imageSize * imageSize; j++) {
                const x = j % imageSize;
                const y = Math.floor(j / imageSize);
                const srcIdx = (row * imageSize + y) * canvas.width * 4 + x * 4;
                testImages[i * imageSize * imageSize + j] = imageData.data[srcIdx] / 255;
            }
            testLabels[i] = labels[trainSize + i];
        }

        this.trainData = {
            images: tf.tensor4d(trainImages, [trainSize, imageSize, imageSize, 1]),
            labels: tf.oneHot(tf.tensor1d(trainLabels, 'int32'), 10)
        };

        this.testData = {
            images: tf.tensor4d(testImages, [testSize, imageSize, imageSize, 1]),
            labels: tf.oneHot(tf.tensor1d(testLabels, 'int32'), 10)
        };

        return {
            train: this.trainData,
            test: this.testData,
            info: DATASETS.mnist
        };
    }

    async loadCIFAR10() {
        // For CIFAR-10, we'll generate synthetic data for demo purposes
        // In production, you'd load actual CIFAR-10 data
        const trainSize = 5000;
        const testSize = 500;
        const imageSize = 32;
        const channels = 3;

        // Generate random images with some structure for demo
        const trainImages = new Float32Array(trainSize * imageSize * imageSize * channels);
        const trainLabels = new Uint8Array(trainSize);
        const testImages = new Float32Array(testSize * imageSize * imageSize * channels);
        const testLabels = new Uint8Array(testSize);

        // Generate synthetic data with class-based patterns
        for (let i = 0; i < trainSize; i++) {
            const label = Math.floor(Math.random() * 10);
            trainLabels[i] = label;
            this.generateSyntheticImage(trainImages, i, imageSize, channels, label);
        }

        for (let i = 0; i < testSize; i++) {
            const label = Math.floor(Math.random() * 10);
            testLabels[i] = label;
            this.generateSyntheticImage(testImages, i, imageSize, channels, label);
        }

        this.trainData = {
            images: tf.tensor4d(trainImages, [trainSize, imageSize, imageSize, channels]),
            labels: tf.oneHot(tf.tensor1d(trainLabels, 'int32'), 10)
        };

        this.testData = {
            images: tf.tensor4d(testImages, [testSize, imageSize, imageSize, channels]),
            labels: tf.oneHot(tf.tensor1d(testLabels, 'int32'), 10)
        };

        return {
            train: this.trainData,
            test: this.testData,
            info: DATASETS.cifar10
        };
    }

    generateSyntheticImage(array, index, size, channels, label) {
        const offset = index * size * size * channels;
        const baseColor = [
            (label * 25 + 50) / 255,
            ((label * 37) % 255) / 255,
            ((label * 61) % 255) / 255
        ];

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const pixelOffset = offset + (y * size + x) * channels;
                const noise = Math.random() * 0.3;

                // Add some structure based on label
                const centerDist = Math.sqrt(Math.pow(x - size / 2, 2) + Math.pow(y - size / 2, 2)) / size;
                const pattern = label % 3 === 0 ?
                    Math.sin(x / 4 + label) * 0.2 :
                    label % 3 === 1 ?
                        (centerDist < 0.3 ? 0.5 : 0) :
                        (x > y ? 0.3 : 0);

                for (let c = 0; c < channels; c++) {
                    array[pixelOffset + c] = Math.min(1, Math.max(0, baseColor[c] + noise + pattern));
                }
            }
        }
    }

    loadImage(blob) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = URL.createObjectURL(blob);
        });
    }

    getSampleImages(count = 10) {
        if (!this.trainData) return null;

        const indices = [];
        const totalImages = this.trainData.images.shape[0];

        for (let i = 0; i < count; i++) {
            indices.push(Math.floor(Math.random() * totalImages));
        }

        return tf.tidy(() => {
            const samples = [];
            for (const idx of indices) {
                const image = this.trainData.images.slice([idx, 0, 0, 0], [1, -1, -1, -1]).squeeze();
                const label = this.trainData.labels.slice([idx, 0], [1, -1]).argMax(1).dataSync()[0];
                samples.push({ image, label, index: idx });
            }
            return samples;
        });
    }

    getImage(index) {
        if (!this.trainData) return null;

        return tf.tidy(() => {
            const image = this.trainData.images.slice([index, 0, 0, 0], [1, -1, -1, -1]).squeeze();
            const label = this.trainData.labels.slice([index, 0], [1, -1]).argMax(1).dataSync()[0];
            return { image, label };
        });
    }

    getBatch(batchSize, isTest = false) {
        const data = isTest ? this.testData : this.trainData;
        if (!data) return null;

        const totalImages = data.images.shape[0];
        const indices = [];

        for (let i = 0; i < batchSize; i++) {
            indices.push(Math.floor(Math.random() * totalImages));
        }

        return tf.tidy(() => {
            const batchImages = tf.stack(indices.map(idx =>
                data.images.slice([idx, 0, 0, 0], [1, -1, -1, -1]).squeeze()
            ));
            const batchLabels = tf.stack(indices.map(idx =>
                data.labels.slice([idx, 0], [1, -1]).squeeze()
            ));
            return { images: batchImages, labels: batchLabels };
        });
    }

    dispose() {
        if (this.trainData) {
            this.trainData.images.dispose();
            this.trainData.labels.dispose();
        }
        if (this.testData) {
            this.testData.images.dispose();
            this.testData.labels.dispose();
        }
    }

    getDatasetInfo() {
        return DATASETS[this.currentDataset];
    }
}
