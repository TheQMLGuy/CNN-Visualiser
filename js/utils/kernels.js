/**
 * CNN Visualizer - Predefined Kernels
 * Common convolution kernels with explanations
 */

export const KERNELS = {
    // Identity - No change
    identity: {
        name: 'Identity',
        description: 'Returns the original image unchanged. Useful as a baseline comparison.',
        category: 'basic',
        kernel: [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        color: '#8b949e'
    },

    // Edge Detection
    sobelX: {
        name: 'Sobel X (Vertical Edges)',
        description: 'Detects vertical edges by computing the gradient in the X direction. Bright areas show where image intensity changes from left to right.',
        category: 'edge',
        kernel: [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        color: '#667eea'
    },

    sobelY: {
        name: 'Sobel Y (Horizontal Edges)',
        description: 'Detects horizontal edges by computing the gradient in the Y direction. Bright areas show where image intensity changes from top to bottom.',
        category: 'edge',
        kernel: [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        color: '#764ba2'
    },

    laplacian: {
        name: 'Laplacian',
        description: 'Detects edges in all directions simultaneously. Highlights regions of rapid intensity change, useful for finding object boundaries.',
        category: 'edge',
        kernel: [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ],
        color: '#f85149'
    },

    laplacianDiagonal: {
        name: 'Laplacian (with Diagonals)',
        description: 'Enhanced Laplacian that also considers diagonal neighbors. More sensitive to edges at all angles.',
        category: 'edge',
        kernel: [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ],
        color: '#da3633'
    },

    prewittX: {
        name: 'Prewitt X',
        description: 'Similar to Sobel but with equal weights. Simpler edge detection in the horizontal direction.',
        category: 'edge',
        kernel: [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        color: '#58a6ff'
    },

    prewittY: {
        name: 'Prewitt Y',
        description: 'Prewitt operator for vertical edges. Uses uniform weights unlike the Sobel operator.',
        category: 'edge',
        kernel: [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        color: '#7ee787'
    },

    // Sharpening
    sharpen: {
        name: 'Sharpen',
        description: 'Enhances edges and fine details by amplifying high-frequency components. Makes the image appear crisper.',
        category: 'sharpen',
        kernel: [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        color: '#ffa657'
    },

    sharpenStrong: {
        name: 'Sharpen (Strong)',
        description: 'Aggressive sharpening that heavily emphasizes edges. Can create halo effects if applied too strongly.',
        category: 'sharpen',
        kernel: [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ],
        color: '#ff7b72'
    },

    // Blur/Smoothing
    boxBlur: {
        name: 'Box Blur',
        description: 'Simple averaging filter that replaces each pixel with the mean of its neighbors. Creates a uniform blur effect.',
        category: 'blur',
        kernel: [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ],
        color: '#3fb950'
    },

    gaussianBlur: {
        name: 'Gaussian Blur',
        description: 'Weighted averaging that gives more importance to center pixels. Creates a natural, smooth blur effect.',
        category: 'blur',
        kernel: [
            [1 / 16, 2 / 16, 1 / 16],
            [2 / 16, 4 / 16, 2 / 16],
            [1 / 16, 2 / 16, 1 / 16]
        ],
        color: '#2ea043'
    },

    // Emboss/3D Effects
    emboss: {
        name: 'Emboss',
        description: 'Creates a 3D raised effect by highlighting edges in one direction and shadowing in the opposite direction.',
        category: 'effect',
        kernel: [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ],
        color: '#d29922'
    },

    embossTopLeft: {
        name: 'Emboss (Top-Left)',
        description: 'Emboss effect with light source from the top-left corner. Creates depth perception.',
        category: 'effect',
        kernel: [
            [2, 1, 0],
            [1, 1, -1],
            [0, -1, -2]
        ],
        color: '#e3b341'
    },

    // Ridge Detection
    ridge: {
        name: 'Ridge Detection',
        description: 'Highlights ridge-like structures in the image. Useful for detecting lines and linear features.',
        category: 'edge',
        kernel: [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        color: '#a371f7'
    },

    // Outline
    outline: {
        name: 'Outline',
        description: 'Extracts the outline of objects by detecting edges and removing the interior.',
        category: 'edge',
        kernel: [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ],
        color: '#bc8cff'
    },

    // Motion Blur
    motionBlurHorizontal: {
        name: 'Motion Blur (Horizontal)',
        description: 'Simulates horizontal camera motion. Blurs only in the horizontal direction.',
        category: 'blur',
        kernel: [
            [0, 0, 0],
            [1 / 3, 1 / 3, 1 / 3],
            [0, 0, 0]
        ],
        color: '#56d364'
    },

    motionBlurVertical: {
        name: 'Motion Blur (Vertical)',
        description: 'Simulates vertical camera motion. Blurs only in the vertical direction.',
        category: 'blur',
        kernel: [
            [0, 1 / 3, 0],
            [0, 1 / 3, 0],
            [0, 1 / 3, 0]
        ],
        color: '#46954a'
    }
};

// Kernel categories for UI grouping
export const KERNEL_CATEGORIES = {
    basic: { name: 'Basic', icon: '⬜' },
    edge: { name: 'Edge Detection', icon: '📐' },
    sharpen: { name: 'Sharpening', icon: '✨' },
    blur: { name: 'Blur/Smooth', icon: '🌫️' },
    effect: { name: 'Effects', icon: '🎨' }
};

// Helper to get all kernels as array
export function getKernelList() {
    return Object.entries(KERNELS).map(([key, value]) => ({
        id: key,
        ...value
    }));
}

// Helper to get kernels by category
export function getKernelsByCategory(category) {
    return Object.entries(KERNELS)
        .filter(([_, value]) => value.category === category)
        .map(([key, value]) => ({ id: key, ...value }));
}
