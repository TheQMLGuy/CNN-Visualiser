/**
 * CNN Visualizer - Training Charts
 * Real-time loss and accuracy visualization
 */

import { CHART_COLORS } from '../utils/constants.js';

export class TrainingCharts {
    constructor(lossCanvas, accuracyCanvas) {
        this.lossCanvas = lossCanvas;
        this.accuracyCanvas = accuracyCanvas;
        this.lossCtx = lossCanvas.getContext('2d');
        this.accuracyCtx = accuracyCanvas.getContext('2d');

        this.lossData = [];
        this.accuracyData = [];
        this.valLossData = [];
        this.valAccuracyData = [];

        this.init();
    }

    init() {
        // Set up canvas with device pixel ratio for crisp rendering
        this.setupCanvas(this.lossCanvas, this.lossCtx);
        this.setupCanvas(this.accuracyCanvas, this.accuracyCtx);

        this.drawEmpty(this.lossCtx, this.lossCanvas, 'Loss');
        this.drawEmpty(this.accuracyCtx, this.accuracyCanvas, 'Accuracy');
    }

    setupCanvas(canvas, ctx) {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        ctx.scale(dpr, dpr);
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
    }

    drawEmpty(ctx, canvas, label) {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        ctx.clearRect(0, 0, width, height);

        // Draw grid
        this.drawGrid(ctx, width, height);

        // Draw label
        ctx.fillStyle = CHART_COLORS.text;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(label, 8, 14);

        // Draw placeholder text
        ctx.textAlign = 'center';
        ctx.fillStyle = '#3d4450';
        ctx.fillText('Training data will appear here', width / 2, height / 2);
        ctx.textAlign = 'left';
    }

    drawGrid(ctx, width, height) {
        ctx.strokeStyle = CHART_COLORS.grid;
        ctx.lineWidth = 1;

        // Horizontal lines
        const numHLines = 4;
        for (let i = 0; i <= numHLines; i++) {
            const y = (height / numHLines) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Vertical lines
        const numVLines = 5;
        for (let i = 0; i <= numVLines; i++) {
            const x = (width / numVLines) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
    }

    update(history) {
        this.lossData = history.loss || [];
        this.accuracyData = history.accuracy || [];
        this.valLossData = history.valLoss || [];
        this.valAccuracyData = history.valAccuracy || [];

        this.drawLossChart();
        this.drawAccuracyChart();
    }

    drawLossChart() {
        const ctx = this.lossCtx;
        const canvas = this.lossCanvas;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const padding = { top: 20, right: 10, bottom: 10, left: 10 };

        ctx.clearRect(0, 0, width, height);
        this.drawGrid(ctx, width, height);

        // Draw label
        ctx.fillStyle = CHART_COLORS.text;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('Loss', 8, 14);

        if (this.lossData.length === 0) return;

        // Calculate bounds
        const allLoss = [...this.lossData, ...this.valLossData];
        const maxLoss = Math.max(...allLoss) * 1.1;
        const minLoss = 0;

        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Draw training loss
        this.drawLine(ctx, this.lossData, maxLoss, minLoss, chartWidth, chartHeight, padding, CHART_COLORS.loss);

        // Draw validation loss if available
        if (this.valLossData.length > 0) {
            this.drawLine(ctx, this.valLossData, maxLoss, minLoss, chartWidth, chartHeight, padding, CHART_COLORS.loss, true);
        }

        // Draw current value
        const currentLoss = this.lossData[this.lossData.length - 1];
        ctx.fillStyle = CHART_COLORS.loss;
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(currentLoss.toFixed(4), width - 8, 14);
        ctx.textAlign = 'left';
    }

    drawAccuracyChart() {
        const ctx = this.accuracyCtx;
        const canvas = this.accuracyCanvas;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const padding = { top: 20, right: 10, bottom: 10, left: 10 };

        ctx.clearRect(0, 0, width, height);
        this.drawGrid(ctx, width, height);

        // Draw label
        ctx.fillStyle = CHART_COLORS.text;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('Accuracy', 8, 14);

        if (this.accuracyData.length === 0) return;

        // Calculate bounds (accuracy is 0-1)
        const maxAcc = 1;
        const minAcc = 0;

        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Draw training accuracy
        this.drawLine(ctx, this.accuracyData, maxAcc, minAcc, chartWidth, chartHeight, padding, CHART_COLORS.accuracy);

        // Draw validation accuracy if available
        if (this.valAccuracyData.length > 0) {
            this.drawLine(ctx, this.valAccuracyData, maxAcc, minAcc, chartWidth, chartHeight, padding, CHART_COLORS.accuracy, true);
        }

        // Draw current value
        const currentAcc = this.accuracyData[this.accuracyData.length - 1];
        ctx.fillStyle = CHART_COLORS.accuracy;
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText((currentAcc * 100).toFixed(1) + '%', width - 8, 14);
        ctx.textAlign = 'left';
    }

    drawLine(ctx, data, maxVal, minVal, chartWidth, chartHeight, padding, color, dashed = false) {
        if (data.length === 0) return;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        if (dashed) {
            ctx.setLineDash([4, 4]);
            ctx.globalAlpha = 0.6;
        } else {
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
        }

        const range = maxVal - minVal || 1;

        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / Math.max(data.length - 1, 1)) * chartWidth;
            const y = padding.top + (1 - (data[i] - minVal) / range) * chartHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.setLineDash([]);

        // Draw points
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / Math.max(data.length - 1, 1)) * chartWidth;
            const y = padding.top + (1 - (data[i] - minVal) / range) * chartHeight;

            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }
    }

    reset() {
        this.lossData = [];
        this.accuracyData = [];
        this.valLossData = [];
        this.valAccuracyData = [];

        this.init();
    }
}
