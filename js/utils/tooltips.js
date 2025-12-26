/**
 * CNN Visualizer - Tooltip Manager
 */

export class TooltipManager {
    constructor() {
        this.tooltip = document.getElementById('tooltip');
        this.showTimeout = null;
        this.hideTimeout = null;
        this.init();
    }

    init() {
        // Add event listeners to all elements with data-tooltip
        document.addEventListener('mouseover', (e) => {
            const target = e.target.closest('[data-tooltip]');
            if (target) {
                this.scheduleShow(target);
            }
        });

        document.addEventListener('mouseout', (e) => {
            const target = e.target.closest('[data-tooltip]');
            if (target) {
                this.scheduleHide();
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.tooltip.classList.contains('visible')) {
                this.position(e.clientX, e.clientY);
            }
        });
    }

    scheduleShow(element) {
        clearTimeout(this.hideTimeout);
        clearTimeout(this.showTimeout);

        this.showTimeout = setTimeout(() => {
            this.show(element);
        }, 400);
    }

    scheduleHide() {
        clearTimeout(this.showTimeout);
        clearTimeout(this.hideTimeout);

        this.hideTimeout = setTimeout(() => {
            this.hide();
        }, 100);
    }

    show(element) {
        const text = element.dataset.tooltip;
        if (!text) return;

        this.tooltip.textContent = text;
        this.tooltip.classList.add('visible');

        const rect = element.getBoundingClientRect();
        this.position(rect.left + rect.width / 2, rect.top);
    }

    position(x, y) {
        const padding = 10;
        const tooltipRect = this.tooltip.getBoundingClientRect();

        let left = x - tooltipRect.width / 2;
        let top = y - tooltipRect.height - padding;

        // Keep within viewport
        if (left < padding) left = padding;
        if (left + tooltipRect.width > window.innerWidth - padding) {
            left = window.innerWidth - tooltipRect.width - padding;
        }
        if (top < padding) {
            top = y + padding + 20; // Show below instead
        }

        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
    }

    hide() {
        this.tooltip.classList.remove('visible');
    }

    // Show custom tooltip at specific position
    showAt(x, y, text) {
        this.tooltip.textContent = text;
        this.tooltip.classList.add('visible');
        this.position(x, y);
    }
}
