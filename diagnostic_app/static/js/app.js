document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentData = null;
    let selectedFiles = [];
    let animationInterval = null;
    let chartOriginal = null;
    let chartProcessed = null;
    let chartShap = null;
    


    // Elements
    const dom = {
        targetGroup: document.getElementById('target-group'),
        modelFile: document.getElementById('model-file'),
        dropzone: document.getElementById('file-dropzone'),
        fileInput: document.getElementById('file-upload'),
        fileList: document.getElementById('file-list'),
        btnAnalyze: document.getElementById('btn-analyze'),
        btnExport: document.getElementById('btn-export'),
        loader: document.getElementById('loader'),
        dashSection: document.getElementById('dashboard-section'),
        dashModelName: document.getElementById('dash-model-name'),
        resultsContainer: document.getElementById('patient-results-container'),
        tabProcessed: document.getElementById('tab-processed'),
        tabShap: document.getElementById('tab-shap'),
        frameSlider: document.getElementById('frame-slider'),
        frameDisplay: document.getElementById('frame-display'),
        btnAnimate: document.getElementById('btn-animate'),
        patientId: document.getElementById('patient-id')
    };

    // Load Models
    fetch('/api/models/')
        .then(r => r.json())
        .then(data => {
            window.modelsData = data.models;
            data.models.forEach(item => {
                const opt = document.createElement('option');
                opt.value = item.group;
                opt.textContent = item.group.replace(/_/g, ' ');
                dom.targetGroup.appendChild(opt);
            });
        });

    dom.targetGroup.addEventListener('change', (e) => {
        const group = e.target.value;
        dom.modelFile.innerHTML = '<option value="">Select Model...</option>';
        if (group) {
            const groupData = window.modelsData.find(m => m.group === group);
            if (groupData) {
                groupData.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    dom.modelFile.appendChild(opt);
                });
            }
        }
    });

    // File Upload
    dom.dropzone.addEventListener('click', () => dom.fileInput.click());
    dom.dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dom.dropzone.classList.add('bg-base-200'); });
    dom.dropzone.addEventListener('dragleave', () => dom.dropzone.classList.remove('bg-base-200'));
    dom.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.dropzone.classList.remove('bg-base-200');
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });
    dom.fileInput.addEventListener('change', () => handleFiles(dom.fileInput.files));

    function handleFiles(files) {
        selectedFiles = Array.from(files).filter(f => f.name.endsWith('.csv') || f.name.endsWith('.txt'));
        dom.fileList.textContent = selectedFiles.map(f => f.name).join(', ');
        if (selectedFiles.length > 0) {
            runPreprocess();
        }
    }

    // Run Preprocess Function (Fast, no ML)
    function runPreprocess() {
        if (!selectedFiles.length) return;

        const fd = new FormData();
        selectedFiles.forEach(f => fd.append('files', f));
        fd.append('use_baseline', document.getElementById('use-baseline').checked);
        fd.append('use_snv', document.getElementById('use-snv').checked);
        fd.append('use_minmax', document.getElementById('use-minmax').checked);

        fetch('/api/preprocess/', {
            method: 'POST',
            body: fd
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                // If we already have currentData with predictions, we want to clear them or just replace
                currentData = data.files;

                // Setup slider
                let maxFrames = 0;
                Object.values(currentData).forEach(f => {
                    if (f.runs.length > maxFrames) maxFrames = f.runs.length;
                });
                if (maxFrames > 0) {
                    dom.frameSlider.max = maxFrames - 1;
                    dom.frameSlider.value = maxFrames - 1;
                    dom.frameSlider.disabled = false;
                    dom.btnAnimate.disabled = false;
                    // Note: Export PDF requires ML data for the snapshot, but we can enable it or leave disabled.
                    // Let's enable it so they can at least see graphs
                    dom.btnExport.disabled = false;
                    renderDashboard(maxFrames - 1);
                } else {
                    dom.frameSlider.disabled = true;
                    dom.btnAnimate.disabled = true;
                    dom.btnExport.disabled = true;
                }
            })
            .catch(err => {
                console.error(err);
                alert('Preprocessing failed: ' + err.message);
            });
    }

    // Run Analysis Function (Slower, includes ML)
    function runAnalysis() {
        if (!selectedFiles.length) return;

        dom.loader.classList.remove('hidden');

        const fd = new FormData();
        selectedFiles.forEach(f => fd.append('files', f));
        fd.append('target_group', dom.targetGroup.value);
        fd.append('model_file', dom.modelFile.value);

        fetch('/api/analyze/', {
            method: 'POST',
            body: fd
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                if (!currentData) {
                    alert("Please wait for files to finish processing first.");
                    return;
                }

                // Merge predictions into currentData
                Object.keys(data.files).forEach(fname => {
                    if (currentData[fname]) {
                        currentData[fname].aggregate_prediction = data.files[fname].aggregate_prediction;
                        currentData[fname].aggregate_confidence = data.files[fname].aggregate_confidence;
                        currentData[fname].shap_values = data.files[fname].shap_values;
                        currentData[fname].shap_spec = data.files[fname].shap_spec;
                        
                        if (data.files[fname].runs) {
                            data.files[fname].runs.forEach((r, idx) => {
                                if (currentData[fname].runs[idx]) {
                                    currentData[fname].runs[idx].prediction = r.prediction;
                                    currentData[fname].runs[idx].confidence = r.confidence;
                                }
                            });
                        }
                    }
                });

                let maxFrames = 0;
                Object.values(currentData).forEach(f => {
                    if (f.runs.length > maxFrames) maxFrames = f.runs.length;
                });
                if (maxFrames > 0) {
                    renderDashboard(parseInt(dom.frameSlider.value) || 0);
                }
            })
            .catch(err => {
                console.error(err);
                alert('Analysis failed: ' + err.message);
            })
            .finally(() => {
                dom.loader.classList.add('hidden');
            });
    }

    // Triggers
    dom.btnAnalyze.addEventListener('click', runAnalysis);
    
    // Listen to changes on controls to auto-update preprocessing ONLY
    document.getElementById('use-baseline').addEventListener('change', runPreprocess);
    document.getElementById('use-snv').addEventListener('change', runPreprocess);
    document.getElementById('use-minmax').addEventListener('change', runPreprocess);
    
    // Changing model shouldn't re-run preprocessing, but it clears the prediction until they hit analyze
    // If you prefer it auto-analyzes when model changes:
    // dom.targetGroup.addEventListener('change', runAnalysis);
    // dom.modelFile.addEventListener('change', runAnalysis);

    // Just re-render dashboard when shapley toggle changes, no need to fetch again
    document.getElementById('display-shapley').addEventListener('change', () => {
        if (currentData) {
            renderDashboard(parseInt(dom.frameSlider.value));
        }
    });

    // Slider & Animation
    dom.frameSlider.addEventListener('input', (e) => {
        if (animationInterval) stopAnimation();
        renderDashboard(parseInt(e.target.value));
    });

    dom.btnAnimate.addEventListener('click', () => {
        if (animationInterval) {
            stopAnimation();
        } else {
            dom.btnAnimate.textContent = 'Stop Animation';
            dom.frameSlider.value = 0;
            animationInterval = setInterval(() => {
                let v = parseInt(dom.frameSlider.value);
                if (v >= parseInt(dom.frameSlider.max)) {
                    stopAnimation();
                } else {
                    v++;
                    dom.frameSlider.value = v;
                    renderDashboard(v);
                }
            }, 100);
        }
    });

    function stopAnimation() {
        clearInterval(animationInterval);
        animationInterval = null;
        dom.btnAnimate.textContent = 'Animate Sequence';
    }

    // Charting Setup
    const chartColors = ['#1e4a8d', '#e63946', '#2a9d8f', '#f4a261', '#9b5de5', '#00b4d8'];

    function initCharts() {
        if (chartOriginal) chartOriginal.destroy();
        if (chartProcessed) chartProcessed.destroy();
        if (chartShap) chartShap.destroy();

        const commOpt = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false, // critical for smooth fast sliding
            elements: { point: { radius: 0 }, line: { borderWidth: 1.5 } },
            plugins: { legend: { position: 'top', labels: { boxWidth: 10 } } },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'Raman Shift (cm⁻¹)' }
                }
            }
        };

        chartOriginal = new Chart(document.getElementById('chart-original').getContext('2d'), {
            type: 'scatter', data: { datasets: [] }, options: { ...commOpt }
        });
        chartProcessed = new Chart(document.getElementById('chart-processed').getContext('2d'), {
            type: 'scatter', data: { datasets: [] }, options: { ...commOpt }
        });
        chartShap = new Chart(document.getElementById('chart-shap').getContext('2d'), {
            type: 'scatter',
            data: { datasets: [] },
            options: {
                ...commOpt,
                scales: {
                    x: commOpt.scales.x,
                    y: { position: 'left', title: { display: true, text: 'SHAP Impact', color: 'darkred' } },
                    y1: { position: 'right', title: { display: true, text: 'Normalized Intensity', color: '#1e4a8d' }, grid: { drawOnChartArea: false } }
                }
            }
        });
    }
    


    function renderDashboard(frameIdx) {
        if (!currentData) return;
        dom.frameDisplay.textContent = frameIdx + 1;

        if (!chartOriginal) initCharts();

        dom.resultsContainer.innerHTML = '';
        dom.dashSection.classList.remove('hidden');
        dom.dashModelName.textContent = dom.modelFile.value || 'None';

        const needsProcessedChart = document.getElementById('use-baseline').checked ||
            document.getElementById('use-snv').checked ||
            document.getElementById('use-minmax').checked;

        let hasShap = false;

        const origDatasets = [];
        const procDatasets = [];
        const shapDatasets = [];

        Object.keys(currentData).forEach((fname, i) => {
            const fileData = currentData[fname];
            if (frameIdx >= fileData.runs.length) return;

            const run = fileData.runs[frameIdx];
            const color = chartColors[i % chartColors.length];

            // Metrics UI
            if (run.prediction) {
                dom.resultsContainer.innerHTML += `
                    <div class="stats shadow w-full">
                        <div class="stat">
                            <div class="stat-title text-base-content/70">Target Group</div>
                            <div class="stat-value text-lg">${dom.targetGroup.value || 'None'}</div>
                            <div class="stat-desc">Dataset: ${fname.replace(/_/g, ' ')}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-title text-base-content/70">Aggregate Diagnosis</div>
                            <div class="stat-value text-primary">${fileData.aggregate_prediction}</div>
                            <div class="stat-desc font-semibold text-primary/80">${(fileData.aggregate_confidence * 100).toFixed(1)}% Confidence</div>
                        </div>
                        <div class="stat">
                            <div class="stat-title text-base-content/70">Frame ${frameIdx + 1} Diagnosis</div>
                            <div class="stat-value text-secondary">${run.prediction}</div>
                            <div class="stat-desc font-semibold text-secondary/80">${(run.confidence * 100).toFixed(1)}% Confidence</div>
                        </div>
                    </div>
                `;
            }

            // Charts Data
            origDatasets.push({
                label: fname,
                data: run.shift.map((s, idx) => ({ x: s, y: run.intensity[idx] })),
                borderColor: color,
                backgroundColor: color,
                showLine: true
            });

            if (run.baseline) {
                origDatasets.push({
                    label: fname + ' (Baseline)',
                    data: run.shift.map((s, idx) => ({ x: s, y: run.baseline[idx] })),
                    borderColor: color,
                    borderDash: [5, 5],
                    borderWidth: 1,
                    showLine: true,
                    hidden: false // could default to hidden
                });
            }

            if (needsProcessedChart) {
                procDatasets.push({
                    label: fname,
                    data: run.shift.map((s, idx) => ({ x: s, y: run.normalized[idx] })),
                    borderColor: color,
                    backgroundColor: color,
                    showLine: true
                });
            }

            if (fileData.shap_values && fileData.shap_spec) {
                hasShap = true;
                shapDatasets.push({
                    label: "SHAP Feature Importance",
                    data: run.shift.map((s, idx) => ({ x: s, y: fileData.shap_values[idx] })),
                    borderColor: 'darkred',
                    backgroundColor: 'rgba(255,0,0,0.3)',
                    fill: true,
                    showLine: true,
                    yAxisID: 'y'
                });
                shapDatasets.push({
                    label: "Analyzed Spectra Array",
                    data: run.shift.map((s, idx) => ({ x: s, y: fileData.shap_spec[idx] })),
                    borderColor: '#1e4a8d',
                    borderDash: [5, 5],
                    showLine: true,
                    yAxisID: 'y1'
                });
            }
        });

        // Update Charts
        chartOriginal.data.datasets = origDatasets;
        chartOriginal.update();

        if (needsProcessedChart) {
            chartProcessed.data.datasets = procDatasets;
            chartProcessed.update();
        }

        const showShap = hasShap && document.getElementById('display-shapley').checked;
        if (showShap) {
            chartShap.data.datasets = shapDatasets;
            chartShap.update();
        }
        
        if (needsProcessedChart) {
            dom.tabProcessed.style.display = '';
            dom.tabProcessed.nextElementSibling.style.display = '';
        } else {
            dom.tabProcessed.style.display = 'none';
            dom.tabProcessed.nextElementSibling.style.display = 'none';
            if (dom.tabProcessed.checked) document.getElementById('tab-original').checked = true;
        }

        if (showShap) {
            dom.tabShap.style.display = '';
            dom.tabShap.nextElementSibling.style.display = '';
        } else {
            dom.tabShap.style.display = 'none';
            dom.tabShap.nextElementSibling.style.display = 'none';
            if (dom.tabShap.checked) document.getElementById('tab-original').checked = true;
        }
    }

    // Export PDF
    dom.btnExport.addEventListener('click', () => {
        if (!currentData) return;

        dom.btnExport.disabled = true;
        dom.btnExport.textContent = 'Generating...';

        const fd = new FormData();
        fd.append('data', JSON.stringify(currentData));
        fd.append('patient_id', dom.patientId.value || 'Diagnostic');
        fd.append('target_group', dom.targetGroup.value);

        fetch('/api/export-report/', {
            method: 'POST',
            body: fd
        })
            .then(res => {
                if (!res.ok) throw new Error("Failed to export");
                return res.blob();
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                // Generate filename based on date
                const d = new Date();
                const timeStr = `${d.getHours()}${d.getMinutes()}${d.getSeconds()}`;
                a.download = `Diagnostic_Reports_${timeStr}.zip`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(err => alert(err.message))
            .finally(() => {
                dom.btnExport.disabled = false;
                dom.btnExport.textContent = 'Export PDF Report';
            });
    });
});
