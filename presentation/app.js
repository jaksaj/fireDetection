(function () {
    const slideOrder = ['overview', 'iteration1', 'iteration2', 'iteration3', 'iteration4', 'iteration5', 'comparison', 'simulator'];

    const slideMeta = {
        overview: { title: 'Project Overview', category: 'Methodology' },
        iteration1: { title: 'Iteration 1: Binary CNN', category: 'Foundations' },
        iteration2: { title: 'Iteration 2: MobileNetV3', category: 'Transfer Learning' },
        iteration3: { title: 'Iteration 3: Robust Model', category: 'Robustness' },
        iteration4: { title: 'Iteration 4: YOLO26 Detect', category: 'Localization' },
        iteration5: { title: 'Iteration 5: U-Net Segmentation', category: 'Segmentation' },
        comparison: { title: 'Comparative Dashboard', category: 'Evaluation' },
        simulator: { title: 'Hardware Deployment Simulator', category: 'Deployment' }
    };

    const modelStats = [
        { id: 'iter1', name: 'Iteration 1: FireCNN', params: 389153, sizeMb: 1.5, edgeScore: 95, baselineFps: { 'pi-zero': 11, 'pi-4': 45, 'jetson-nano': 62, 'jetson-orin': 110, 'desktop-gpu': 180 }, baselineLatency: { 'pi-zero': 91, 'pi-4': 22, 'jetson-nano': 16, 'jetson-orin': 9, 'desktop-gpu': 5 }, recommended: ['pi-zero', 'pi-4', 'jetson-nano'] },
        { id: 'iter3', name: 'Iteration 3: MobileNetV3 Robust', params: 1075748, sizeMb: 1.1, edgeScore: 88, baselineFps: { 'pi-zero': 4.2, 'pi-4': 30, 'jetson-nano': 51, 'jetson-orin': 95, 'desktop-gpu': 158 }, baselineLatency: { 'pi-zero': 238, 'pi-4': 33, 'jetson-nano': 20, 'jetson-orin': 11, 'desktop-gpu': 6 }, recommended: ['pi-4', 'jetson-nano', 'jetson-orin'] },
        { id: 'iter4', name: 'Iteration 4: YOLO26n Detector', params: 2572280, sizeMb: 9.8, edgeScore: 42, baselineFps: { 'pi-zero': 0.4, 'pi-4': 3.5, 'jetson-nano': 10.5, 'jetson-orin': 34, 'desktop-gpu': 72 }, baselineLatency: { 'pi-zero': 2600, 'pi-4': 285, 'jetson-nano': 95, 'jetson-orin': 29, 'desktop-gpu': 14 }, recommended: ['jetson-orin', 'desktop-gpu'] },
        { id: 'iter5', name: 'Iteration 5: Lightweight U-Net', params: 7849667, sizeMb: 30.0, edgeScore: 28, baselineFps: { 'pi-zero': 0.1, 'pi-4': 1.1, 'jetson-nano': 2.9, 'jetson-orin': 11, 'desktop-gpu': 29 }, baselineLatency: { 'pi-zero': 10000, 'pi-4': 910, 'jetson-nano': 345, 'jetson-orin': 91, 'desktop-gpu': 34 }, recommended: ['jetson-orin', 'desktop-gpu'] }
    ];

    const hardwareProfiles = {
        'pi-zero': {
            label: 'Raspberry Pi Zero 2W',
            cpu: 'Quad-core Cortex-A53',
            gpu: 'None',
            power: 'Very low, USB-powered',
            multiplier: 0.32,
            quantBoost: 1.55,
            quantLatencyBoost: 0.78,
            compat: { iter1: 'good', iter3: 'warn', iter4: 'bad', iter5: 'bad' }
        },
        'pi-4': {
            label: 'Raspberry Pi 4 / 5',
            cpu: '4-8 core ARM64',
            gpu: 'Integrated VideoCore',
            power: 'Low, fan recommended',
            multiplier: 1,
            quantBoost: 1.35,
            quantLatencyBoost: 0.72,
            compat: { iter1: 'good', iter3: 'good', iter4: 'warn', iter5: 'bad' }
        },
        'jetson-nano': {
            label: 'NVIDIA Jetson Nano',
            cpu: 'Quad-core ARM A57',
            gpu: '128-core Maxwell GPU',
            power: '5W - 10W mode',
            multiplier: 1.45,
            quantBoost: 1.7,
            quantLatencyBoost: 0.66,
            compat: { iter1: 'good', iter3: 'good', iter4: 'warn', iter5: 'warn' }
        },
        'jetson-orin': {
            label: 'NVIDIA Jetson Orin Nano',
            cpu: '6-core ARM64',
            gpu: 'Ada Lovelace GPU / NPU',
            power: '10W - 25W',
            multiplier: 3.9,
            quantBoost: 1.28,
            quantLatencyBoost: 0.72,
            compat: { iter1: 'good', iter3: 'good', iter4: 'good', iter5: 'good' }
        },
        'desktop-gpu': {
            label: 'Desktop NVIDIA GPU',
            cpu: 'Modern desktop x86_64',
            gpu: 'RTX-class CUDA GPU',
            power: 'High, mains powered',
            multiplier: 8,
            quantBoost: 1.12,
            quantLatencyBoost: 0.82,
            compat: { iter1: 'good', iter3: 'good', iter4: 'good', iter5: 'good' }
        }
    };

    function formatCompactNumber(value) {
        return new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value);
    }

    function getSlideIdFromHash() {
        const raw = (location.hash || '#overview').replace('#', '');
        if (slideOrder.includes(raw)) {
            return raw;
        }
        if (raw.startsWith('slide-')) {
            const candidate = raw.replace('slide-', '');
            if (slideOrder.includes(candidate)) {
                return candidate;
            }
        }
        return 'overview';
    }

    function setActiveNav(slideId) {
        document.querySelectorAll('.nav-item').forEach((item) => {
            item.classList.toggle('active', item.dataset.slide === slideId);
        });
    }

    function showSlide(slideId) {
        const safeSlideId = slideOrder.includes(slideId) ? slideId : 'overview';
        document.querySelectorAll('.slide-section').forEach((section) => {
            section.classList.toggle('active', section.id === `slide-${safeSlideId}`);
        });

        const meta = slideMeta[safeSlideId] || slideMeta.overview;
        const titleEl = document.getElementById('current-slide-title');
        const categoryEl = document.getElementById('current-slide-category');
        if (titleEl) titleEl.textContent = meta.title;
        if (categoryEl) categoryEl.textContent = meta.category;

        setActiveNav(safeSlideId);
        if (location.hash !== `#${safeSlideId}`) {
            history.replaceState(null, '', `#${safeSlideId}`);
        }
    }

    function createComparisonCharts() {
        if (typeof Chart === 'undefined') {
            return;
        }

        const paramsCanvas = document.getElementById('paramsChart');
        const latencyCanvas = document.getElementById('latencyChart');

        if (paramsCanvas && !paramsCanvas.dataset.chartReady) {
            const ctx = paramsCanvas.getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5'],
                    datasets: [{
                        label: 'Parameters',
                        data: [389153, 1075748, 1075748, 2572280, 7849667],
                        borderWidth: 1,
                        backgroundColor: [
                            'rgba(255, 90, 31, 0.7)',
                            'rgba(255, 155, 47, 0.7)',
                            'rgba(120, 183, 255, 0.7)',
                            'rgba(138, 92, 255, 0.7)',
                            'rgba(66, 211, 146, 0.7)'
                        ],
                        borderColor: 'rgba(255, 255, 255, 0.08)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => ` ${formatCompactNumber(context.raw)} params`
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#c7d2e3' },
                            grid: { display: false }
                        },
                        y: {
                            ticks: { color: '#c7d2e3' },
                            grid: { color: 'rgba(148, 163, 184, 0.12)' }
                        }
                    }
                }
            });
            paramsCanvas.dataset.chartReady = 'true';
        }

        if (latencyCanvas && !latencyCanvas.dataset.chartReady) {
            const ctx = latencyCanvas.getContext('2d');
            new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Iter 1', 'Iter 2', 'Iter 3', 'Iter 4', 'Iter 5'],
                    datasets: [{
                        label: 'Edge Suitability',
                        data: [95, 80, 88, 42, 28],
                        fill: true,
                        backgroundColor: 'rgba(66, 211, 146, 0.14)',
                        borderColor: 'rgba(66, 211, 146, 0.9)',
                        pointBackgroundColor: 'rgba(255, 155, 47, 0.95)',
                        pointBorderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(148, 163, 184, 0.14)' },
                            grid: { color: 'rgba(148, 163, 184, 0.14)' },
                            pointLabels: { color: '#dbe7f5', font: { size: 12 } },
                            ticks: { display: false, max: 100, min: 0 }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#dbe7f5' } }
                    }
                }
            });
            latencyCanvas.dataset.chartReady = 'true';
        }
    }

    function updateSimulator() {
        const hardwareSelect = document.getElementById('hw-select');
        const selectedHardware = hardwareProfiles[hardwareSelect?.value || 'pi-4'];
        const quantized = document.querySelector('input[name="opt-target"]:checked')?.value === 'quant';

        if (!selectedHardware) {
            return;
        }

        const specCpu = document.querySelector('#spec-cpu .val');
        const specGpu = document.querySelector('#spec-gpu .val');
        const specPower = document.querySelector('#spec-power .val');

        if (specCpu) specCpu.textContent = selectedHardware.cpu;
        if (specGpu) specGpu.textContent = selectedHardware.gpu;
        if (specPower) specPower.textContent = selectedHardware.power;

        let bestModel = 'iter3';
        let bestScore = -Infinity;

        modelStats.forEach((model) => {
            const compatState = selectedHardware.compat[model.id] || 'warn';
            let compatibilityScore = compatState === 'good' ? 28 : compatState === 'warn' ? 10 : -20;
            compatibilityScore += model.edgeScore;
            compatibilityScore += selectedHardware.multiplier * 6;
            if (quantized && model.id === 'iter3') {
                compatibilityScore += 12;
            }

            if (compatibilityScore > bestScore) {
                bestScore = compatibilityScore;
                bestModel = model.id;
            }

            const fpsBase = model.baselineFps[hardwareSelect.value];
            const latencyBase = model.baselineLatency[hardwareSelect.value];
            const fps = quantized ? fpsBase * selectedHardware.quantBoost : fpsBase;
            const latency = quantized ? latencyBase * selectedHardware.quantLatencyBoost : latencyBase;

            const fpsEl = document.getElementById(`fps-${model.id}`);
            const latEl = document.getElementById(`lat-${model.id}`);
            const barEl = document.getElementById(`bar-${model.id}`);
            const compatEl = document.getElementById(`compat-${model.id}`);

            if (fpsEl) fpsEl.textContent = `${fps.toFixed(fps < 10 ? 1 : 0)} FPS`;
            if (latEl) latEl.textContent = `${Math.round(latency)} ms`;
            if (barEl) {
                const normalized = Math.max(4, Math.min(96, fps * 2.2));
                barEl.style.width = `${normalized}%`;
            }
            if (compatEl) {
                const compatStateLabel = selectedHardware.compat[model.id] || 'warn';
                compatEl.className = `compatibility-badge ${compatStateLabel === 'good' ? 'good' : compatStateLabel === 'warn' ? 'warn' : 'bad'}`;
                compatEl.textContent = compatStateLabel === 'good' ? 'Recommended' : compatStateLabel === 'warn' ? 'Usable with constraints' : 'Not Recommended';
            }
        });

        const recommendationEl = document.getElementById('deployment-summary-text');
        if (recommendationEl) {
            const chosen = modelStats.find((model) => model.id === bestModel);
            const optimizationText = quantized ? 'INT8 quantization' : 'FP32 inference';
            recommendationEl.textContent = `${selectedHardware.label} pairs best with ${chosen?.name || 'Iteration 3'} using ${optimizationText}. The selected hardware profile favors ${chosen?.id === 'iter3' ? 'balanced throughput and compact size' : 'deployment efficiency'} for this project.`;
        }
    }

    function initTooltips() {
        const tooltip = document.getElementById('layer-tooltip');
        if (!tooltip) {
            return;
        }

        document.querySelectorAll('.arch-node').forEach((node) => {
            const text = node.getAttribute('data-tooltip');
            if (!text) {
                return;
            }

            node.addEventListener('mouseenter', () => {
                tooltip.textContent = text;
                tooltip.style.display = 'block';
            });

            node.addEventListener('mousemove', (event) => {
                const offset = 18;
                const x = Math.min(window.innerWidth - tooltip.offsetWidth - 12, event.clientX + offset);
                const y = Math.min(window.innerHeight - tooltip.offsetHeight - 12, event.clientY + offset);
                tooltip.style.left = `${Math.max(12, x)}px`;
                tooltip.style.top = `${Math.max(12, y)}px`;
            });

            node.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });
        });
    }

    function bindEvents() {
        document.querySelectorAll('.nav-item').forEach((item) => {
            item.addEventListener('click', (event) => {
                event.preventDefault();
                showSlide(item.dataset.slide || 'overview');
            });
        });

        document.querySelectorAll('.timeline-node').forEach((node) => {
            node.addEventListener('click', () => {
                showSlide(node.dataset.target || 'overview');
            });
        });

        document.querySelectorAll('tr.highlight-row-hover').forEach((row) => {
            row.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    const target = row.getAttribute('onclick')?.match(/showSlide\('([^']+)'\)/)?.[1];
                    if (target) {
                        showSlide(target);
                    }
                }
            });
            row.setAttribute('tabindex', '0');
            row.setAttribute('role', 'button');
        });

        const hwSelect = document.getElementById('hw-select');
        if (hwSelect) {
            hwSelect.addEventListener('change', updateSimulator);
        }

        document.querySelectorAll('input[name="opt-target"]').forEach((input) => {
            input.addEventListener('change', updateSimulator);
        });

        window.addEventListener('hashchange', () => {
            showSlide(getSlideIdFromHash());
        });
    }

    function init() {
        window.showSlide = showSlide;
        bindEvents();
        initTooltips();
        createComparisonCharts();
        updateSimulator();
        showSlide(getSlideIdFromHash());
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();