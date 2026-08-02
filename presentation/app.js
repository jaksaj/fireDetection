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

    // ------------------------------------------------------------------
    // MEASURED DATA ONLY.
    //
    // A previous version of this file hardcoded a full per-device FPS and
    // latency matrix (Pi Zero / Pi 4 / Jetson Nano / Orin / desktop GPU) plus
    // per-device quantization "boost" multipliers. Those numbers were never
    // produced by any code in this repository -- they were invented for the
    // slide deck. They have been removed.
    //
    // Latency now comes exclusively from results/benchmarks.csv, produced by
    // scripts/run_benchmarks.py using the harness in src/benchmark.py. Any
    // (model, device, precision) combination that has not actually been
    // measured renders as "not measured" rather than as a plausible number.
    //
    // Params and on-disk sizes below are real: params are counted from the
    // model definitions, sizeMb is the size of the saved checkpoint on disk.
    // ------------------------------------------------------------------

    const modelStats = [
        { id: 'iter1', key: 'iteration1', name: 'Iteration 1: FireCNN', params: 389153, sizeMb: 4.48 },
        { id: 'iter3', key: 'iteration3', name: 'Iteration 3: MobileNetV3 Robust', params: 1075748, sizeMb: 10.33 },
        { id: 'iter4', key: 'iteration4', name: 'Iteration 4: YOLO26n Detector', params: 2572280, sizeMb: 5.29 },
        { id: 'iter5', key: 'iteration5', name: 'Iteration 5: Lightweight U-Net', params: 7849667, sizeMb: 29.95 }
    ];

    // Hardware descriptions are specifications, not performance claims.
    // `measured: false` means no benchmark has been run on this device; the
    // simulator refuses to display numbers for it.
    const hardwareProfiles = {
        'desktop-gpu': {
            label: 'Desktop GPU (RTX 3060)',
            cpu: 'AMD Ryzen, 6 cores (Zen 3)',
            gpu: 'NVIDIA RTX 3060, 12 GB, SM 8.6',
            power: 'High, mains powered',
            benchDevice: 'cuda',
            measured: true
        },
        'desktop-cpu': {
            label: 'Desktop CPU (x86-64)',
            cpu: 'AMD Ryzen, 6 cores (Zen 3)',
            gpu: 'None (CPU inference)',
            power: 'High, mains powered',
            benchDevice: 'cpu',
            measured: true
        },
        'jetson-orin-gpu': {
            label: 'Jetson Orin Nano (GPU)',
            cpu: '6-core Arm Cortex-A78AE',
            gpu: 'Ampere iGPU, 1024 CUDA cores',
            power: '7 W / 15 W / MAXN',
            benchDevice: 'jetson-cuda',
            measured: false
        },
        'jetson-orin-cpu': {
            label: 'Jetson Orin Nano (ARM CPU)',
            cpu: '6-core Arm Cortex-A78AE',
            gpu: 'None (CPU inference)',
            power: '7 W / 15 W / MAXN',
            benchDevice: 'jetson-cpu',
            measured: false
        }
    };

    // Populated at load time from results/benchmarks.csv via loadBenchmarks().
    // Shape: benchmarks[benchDevice][precision][modelKey] = { latency_ms_median, fps, ... }
    let benchmarks = null;

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

        const precision = quantized ? 'int8' : 'fp32';
        const deviceRows = benchmarks?.[selectedHardware.benchDevice]?.[precision] || null;

        let bestModel = null;
        let bestLatency = Infinity;
        let measuredCount = 0;

        modelStats.forEach((model) => {
            const row = deviceRows?.[model.key] || null;

            const fpsEl = document.getElementById(`fps-${model.id}`);
            const latEl = document.getElementById(`lat-${model.id}`);
            const barEl = document.getElementById(`bar-${model.id}`);
            const compatEl = document.getElementById(`compat-${model.id}`);

            if (!row) {
                // No measurement exists for this combination. Say so plainly
                // rather than inventing a plausible-looking number.
                if (fpsEl) fpsEl.textContent = 'not measured';
                if (latEl) latEl.textContent = '—';
                if (barEl) barEl.style.width = '0%';
                if (compatEl) {
                    compatEl.className = 'compatibility-badge warn';
                    compatEl.textContent = 'No benchmark data';
                }
                return;
            }

            measuredCount += 1;
            const latency = row.latency_ms_median;
            const fps = row.fps;

            if (latency < bestLatency) {
                bestLatency = latency;
                bestModel = model;
            }

            if (fpsEl) fpsEl.textContent = `${fps.toFixed(fps < 10 ? 1 : 0)} FPS`;
            if (latEl) latEl.textContent = `${latency.toFixed(latency < 10 ? 2 : 1)} ms`;
            if (barEl) {
                // Bar is scaled against the fastest measured model on this
                // device so the comparison stays honest across scales.
                const fastest = Math.max(...Object.values(deviceRows).map((r) => r.fps));
                barEl.style.width = `${Math.max(4, Math.min(96, (fps / fastest) * 96))}%`;
            }
            if (compatEl) {
                // Realtime thresholds, stated explicitly: >=25 FPS realtime,
                // >=10 FPS usable for periodic monitoring, below that offline.
                const state = fps >= 25 ? 'good' : fps >= 10 ? 'warn' : 'bad';
                compatEl.className = `compatibility-badge ${state}`;
                compatEl.textContent = state === 'good'
                    ? 'Real-time (≥25 FPS)'
                    : state === 'warn'
                        ? 'Periodic monitoring (≥10 FPS)'
                        : 'Offline / batch only';
            }
        });

        const recommendationEl = document.getElementById('deployment-summary-text');
        if (recommendationEl) {
            if (!measuredCount) {
                recommendationEl.textContent = `No benchmarks have been run on ${selectedHardware.label} yet. `
                    + 'Run scripts/run_benchmarks.py on that device to populate this view. '
                    + 'This deck deliberately shows no estimated numbers.';
            } else {
                const optimizationText = quantized ? 'INT8' : 'FP32';
                recommendationEl.textContent = `On ${selectedHardware.label} at ${optimizationText}, `
                    + `${bestModel.name} has the lowest measured median latency `
                    + `(${bestLatency.toFixed(2)} ms, batch size 1). `
                    + `Measured for ${measuredCount} of ${modelStats.length} models; `
                    + 'unmeasured combinations are shown as "not measured".';
            }
        }
    }

    /**
     * Load measured benchmark rows from results/benchmarks.csv.
     *
     * Fails soft: if the file is absent (benchmarks not yet run) the simulator
     * shows "not measured" everywhere instead of falling back to estimates.
     */
    async function loadBenchmarks() {
        try {
            const response = await fetch('../results/benchmarks.csv');
            if (!response.ok) {
                return;
            }
            const text = await response.text();
            const lines = text.trim().split('\n');
            const header = lines[0].split(',').map((h) => h.trim());
            const parsed = {};

            lines.slice(1).forEach((line) => {
                const cells = line.split(',');
                const row = {};
                header.forEach((key, index) => {
                    const raw = (cells[index] || '').trim();
                    const num = Number(raw);
                    row[key] = raw !== '' && !Number.isNaN(num) ? num : raw;
                });
                const { bench_device: dev, precision: prec, model_key: key } = row;
                if (!dev || !prec || !key) {
                    return;
                }
                parsed[dev] = parsed[dev] || {};
                parsed[dev][prec] = parsed[dev][prec] || {};
                parsed[dev][prec][key] = row;
            });

            benchmarks = parsed;
        } catch (error) {
            // Opened via file:// -- fetch is blocked. Leave benchmarks null.
            benchmarks = null;
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