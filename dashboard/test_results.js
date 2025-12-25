// Sidebar Toggle
document.addEventListener('DOMContentLoaded', function () {
    const el = document.getElementById("wrapper");
    const toggleButton = document.getElementById("menu-toggle");

    if (toggleButton) {
        toggleButton.onclick = function () {
            el.classList.toggle("toggled");
        };
    }

    // Load test results data
    loadTestResults();
});

async function loadTestResults() {
    try {
        const response = await fetch('/api/model-results');
        const data = await response.json();

        // Update summary metrics
        updateSummaryMetrics(data);

        // Update profession tables
        updateProfessionTables(data.model_comparison);

        // Update external test results
        updateExternalTestResults(data.external_test);

        // Update augmentation results
        updateAugmentationResults(data.augmentation);

        // Create charts
        createR2Chart(data.model_comparison);
        createMAEChart(data.model_comparison);
        createClassificationCharts(data.model_comparison);

    } catch (error) {
        console.error('Error loading test results:', error);
    }
}

function updateSummaryMetrics(data) {
    let bestR2 = 0;
    let bestMAE = Infinity;

    // Find best metrics across all models
    if (data.model_comparison) {
        for (const prof of Object.values(data.model_comparison)) {
            for (const model of Object.values(prof)) {
                if (model.r2 > bestR2) bestR2 = model.r2;
                if (model.mae < bestMAE) bestMAE = model.mae;
            }
        }
    }

    document.getElementById('best-r2').textContent = bestR2.toFixed(3);
    document.getElementById('best-mae').textContent = bestMAE.toFixed(3);
}

function updateProfessionTables(modelComparison) {
    const professions = ['veteriner', 'gida', 'ziraat'];
    const modelNames = {
        'mlp': 'MLP',
        'resnet': 'ResNet',
        'attention': 'Attention'
    };

    professions.forEach(prof => {
        const tbody = document.getElementById(`${prof}-results`);
        if (!tbody) return;

        let html = '';
        const profData = modelComparison[prof];

        if (profData) {
            for (const [model, metrics] of Object.entries(profData)) {
                const r2Class = getR2Class(metrics.r2);
                html += `
                    <tr>
                        <td><span class="model-badge badge-${model}">${modelNames[model]}</span></td>
                        <td class="${r2Class}">${metrics.r2.toFixed(4)}</td>
                        <td>${metrics.mae.toFixed(4)}</td>
                        <td>${metrics.rmse.toFixed(4)}</td>
                    </tr>
                `;
            }
        }

        tbody.innerHTML = html || '<tr><td colspan="4" class="text-center text-muted">Veri bulunamadi</td></tr>';
    });
}

function getR2Class(r2) {
    if (r2 >= 0.9) return 'r2-excellent';
    if (r2 >= 0.7) return 'r2-good';
    if (r2 >= 0.5) return 'r2-moderate';
    return 'r2-poor';
}

function updateExternalTestResults(externalTest) {
    const tbody = document.getElementById('external-test-results');
    if (!tbody || !externalTest) return;

    const profNames = {
        'veteriner': 'Veteriner',
        'gida': 'Gida',
        'ziraat': 'Ziraat'
    };

    let html = '';

    // Holdout results
    if (externalTest.holdout) {
        for (const [prof, metrics] of Object.entries(externalTest.holdout)) {
            html += `
                <tr>
                    <td>${profNames[prof]}</td>
                    <td class="${getR2Class(metrics.internal_r2)}">${metrics.internal_r2.toFixed(4)}</td>
                    <td class="${getR2Class(metrics.external_r2)}">${metrics.external_r2.toFixed(4)}</td>
                    <td>${metrics.internal_mae.toFixed(4)}</td>
                    <td>${metrics.external_mae.toFixed(4)}</td>
                </tr>
            `;
        }
    }

    tbody.innerHTML = html || '<tr><td colspan="5" class="text-center text-muted">Veri bulunamadi</td></tr>';
}

function updateAugmentationResults(augmentation) {
    const tbody = document.getElementById('augmentation-results');
    if (!tbody || !augmentation || !augmentation.professions) return;

    const profNames = {
        'veteriner': 'Veteriner',
        'gida': 'Gida',
        'ziraat': 'Ziraat'
    };

    let html = '';

    for (const [prof, data] of Object.entries(augmentation.professions)) {
        const baseline = data.baseline;
        const best = data.best;
        const improvement = ((best.r2 - baseline.r2) / baseline.r2 * 100).toFixed(2);
        const improvementClass = improvement > 0 ? 'text-success' : 'text-danger';

        html += `
            <tr>
                <td>${profNames[prof]}</td>
                <td><span class="badge bg-primary">${best.augmentation}</span></td>
                <td>${baseline.r2.toFixed(4)}</td>
                <td class="${getR2Class(best.r2)}">${best.r2.toFixed(4)}</td>
                <td class="${improvementClass}">${improvement > 0 ? '+' : ''}${improvement}%</td>
            </tr>
        `;
    }

    tbody.innerHTML = html || '<tr><td colspan="5" class="text-center text-muted">Veri bulunamadi</td></tr>';
}

function createR2Chart(modelComparison) {
    const ctx = document.getElementById('r2Chart');
    if (!ctx) return;

    const labels = ['Veteriner', 'Gida', 'Ziraat'];
    const datasets = [];
    const modelColors = {
        'mlp': '#1565c0',
        'resnet': '#7b1fa2',
        'attention': '#2e7d32'
    };
    const modelNames = {
        'mlp': 'MLP',
        'resnet': 'ResNet',
        'attention': 'Attention'
    };

    const models = ['mlp', 'resnet', 'attention'];

    models.forEach(model => {
        const data = [];
        ['veteriner', 'gida', 'ziraat'].forEach(prof => {
            if (modelComparison[prof] && modelComparison[prof][model]) {
                data.push(modelComparison[prof][model].r2);
            } else {
                data.push(0);
            }
        });

        datasets.push({
            label: modelNames[model],
            data: data,
            backgroundColor: modelColors[model],
            borderRadius: 5
        });
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: { display: true, text: 'R2 Skoru' }
                }
            }
        }
    });
}

function createMAEChart(modelComparison) {
    const ctx = document.getElementById('maeChart');
    if (!ctx) return;

    const labels = ['Veteriner', 'Gida', 'Ziraat'];
    const datasets = [];
    const modelColors = {
        'mlp': 'rgba(21, 101, 192, 0.7)',
        'resnet': 'rgba(123, 31, 162, 0.7)',
        'attention': 'rgba(46, 125, 50, 0.7)'
    };
    const modelNames = {
        'mlp': 'MLP',
        'resnet': 'ResNet',
        'attention': 'Attention'
    };

    const models = ['mlp', 'resnet', 'attention'];

    models.forEach(model => {
        const data = [];
        ['veteriner', 'gida', 'ziraat'].forEach(prof => {
            if (modelComparison[prof] && modelComparison[prof][model]) {
                data.push(modelComparison[prof][model].mae);
            } else {
                data.push(0);
            }
        });

        datasets.push({
            label: modelNames[model],
            data: data,
            borderColor: modelColors[model],
            backgroundColor: modelColors[model],
            fill: false,
            tension: 0.1
        });
    });

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'MAE (Ortalama Mutlak Hata)' }
                }
            }
        }
    });
}

function createClassificationCharts(modelComparison) {
    const professions = [
        { id: 'vetClassChart', key: 'veteriner' },
        { id: 'gidaClassChart', key: 'gida' },
        { id: 'ziraatClassChart', key: 'ziraat' }
    ];

    professions.forEach(prof => {
        const ctx = document.getElementById(prof.id);
        if (!ctx) return;

        // Use attention model data for classification
        const profData = modelComparison[prof.key];
        let classData = { norm_fazlasi: 0, norm_eksigi: 0, dengede: 0 };

        if (profData && profData.attention && profData.attention.classification_counts) {
            classData = profData.attention.classification_counts;
        }

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Norm Fazlasi', 'Norm Eksigi', 'Dengede'],
                datasets: [{
                    data: [
                        classData.norm_fazlasi || 0,
                        classData.norm_eksigi || 0,
                        classData.dengede || 0
                    ],
                    backgroundColor: ['#0d6efd', '#6f42c1', '#20c997'],
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 10 } } }
                }
            }
        });
    });
}
