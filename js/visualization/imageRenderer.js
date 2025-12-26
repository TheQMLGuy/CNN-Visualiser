/**
 * CNN Visualizer - Image Renderer
 * Renders images to canvas for preview
 */

export class ImageRenderer {
    static renderToCanvas(canvas, tensor, isColor = false) {
        const ctx = canvas.getContext('2d');

        return tf.tidy(() => {
            let imageTensor = tensor;

            // Handle batch dimension
            if (imageTensor.rank === 4) {
                imageTensor = imageTensor.squeeze([0]);
            }

            const [height, width, channels] = imageTensor.shape;

            // Resize canvas if needed
            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }

            const data = imageTensor.dataSync();
            const imageData = ctx.createImageData(width, height);

            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const pixelIdx = (y * width + x) * 4;

                    if (channels === 1) {
                        // Grayscale
                        const value = Math.floor(data[y * width + x] * 255);
                        imageData.data[pixelIdx] = value;
                        imageData.data[pixelIdx + 1] = value;
                        imageData.data[pixelIdx + 2] = value;
                    } else {
                        // Color (RGB)
                        imageData.data[pixelIdx] = Math.floor(data[(y * width + x) * channels] * 255);
                        imageData.data[pixelIdx + 1] = Math.floor(data[(y * width + x) * channels + 1] * 255);
                        imageData.data[pixelIdx + 2] = Math.floor(data[(y * width + x) * channels + 2] * 255);
                    }
                    imageData.data[pixelIdx + 3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);
        });
    }

    static renderScaled(canvas, tensor, targetWidth, targetHeight) {
        // Create temp canvas at original size
        const tempCanvas = document.createElement('canvas');
        ImageRenderer.renderToCanvas(tempCanvas, tensor);

        // Scale to target size
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = false; // Nearest neighbor for pixel art
        ctx.drawImage(tempCanvas, 0, 0, targetWidth, targetHeight);
    }
}
