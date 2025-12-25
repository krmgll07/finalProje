// Sidebar Toggle
const el = document.getElementById("wrapper");
const toggleButton = document.getElementById("menu-toggle");

toggleButton.onclick = function () {
    el.classList.toggle("toggled");
};

document.addEventListener('DOMContentLoaded', function () {
    
    // --- 1. Load Real Data from API ---
    loadRealData();
    
    // Sidebar Toggle
    const el = document.getElementById("wrapper");
    const toggleButton = document.getElementById("menu-toggle");

    toggleButton.onclick = function () {
        el.classList.toggle("toggled");
    };
});

async function loadRealData() {
    try {
        // Load summary data from API
        const summaryResponse = await fetch('/api/summary');
        const summaryData = await summaryResponse.json();
        
        // Update KPI cards with real data
        updateKPICards(summaryData);
        
        // Load district data for table
        const districtsResponse = await fetch('/api/districts?limit=1000');
        const districtsData = await districtsResponse.json();
        
        // Create DataTable with real data
        createDataTable(districtsData.districts);
        
        // Create charts with real data
        createCharts(summaryData);
        
    } catch (error) {
        console.error('Error loading real data:', error);
        // Fallback to static data if API fails
        loadStaticData();
    }
}

function updateKPICards(summaryData) {
    let totalSurplus = 0;
    let totalDeficit = 0;
    let totalBalanced = 0;
    let totalCount = 960; // TÜRKİYE'DEKİ TOPLAM İLÇE SAYISI - DOĞRUDAN 960
    
    // Sadece meslek başına durumları topla, ilçe sayısını sabit 960 olarak göster
    for (const [profession, stats] of Object.entries(summaryData.professions)) {
        totalSurplus += stats.surplus;
        totalDeficit += stats.deficit;
        totalBalanced += stats.balanced;
    }
    
    document.getElementById('total-count').innerText = totalCount.toLocaleString();
    document.getElementById('total-surplus').innerText = totalSurplus.toLocaleString();
    document.getElementById('total-deficit').innerText = totalDeficit.toLocaleString();
    document.getElementById('total-balanced').innerText = totalBalanced.toLocaleString();
}

function createDataTable(districts) {
    const tableData = [];
    
    districts.forEach(district => {
        // Process each profession for this district
        const professions = ['veteriner', 'gida', 'ziraat'];
        
        professions.forEach(prof => {
            const currentCol = prof === 'veteriner' ? 'veteriner_hekim' : `${prof}_muhendisi`;
            const normCol = `${prof}_tahmini_norm_yuvarlak`;
            const diffCol = `${prof}_norm_farki`;
            const statusCol = `${prof}_durumu`;
            
            // Check if this profession data exists for this district
            if (district[currentCol] !== null && district[currentCol] !== undefined) {
                const mevcut = district[currentCol] || 0;
                const norm = district[normCol] || 0;
                const fark = district[diffCol] || 0;
                const durum = district[statusCol] || 'belirsiz';
                
                let durumHtml = '';
                if (durum === 'norm_fazlasi') {
                    durumHtml = '<span class="badge-fazla">Fazla</span>';
                } else if (durum === 'norm_eksigi') {
                    durumHtml = '<span class="badge-eksik">Eksik</span>';
                } else if (durum === 'dengede') {
                    durumHtml = '<span class="badge-denge">Dengede</span>';
                } else {
                    durumHtml = '<span class="badge-secondary">Belirsiz</span>';
                }
                
                const professionNames = {
                    'veteriner': 'Veteriner Hekim',
                    'gida': 'Gıda Mühendisi',
                    'ziraat': 'Ziraat Mühendisi'
                };
                
                tableData.push([
                    professionNames[prof],
                    district.il_adi || 'Bilinmiyor',
                    district.ilce_adi || 'Bilinmiyor',
                    mevcut.toFixed(1),
                    norm.toFixed(1),
                    fark.toFixed(1),
                    durumHtml
                ]);
            }
        });
    });
    
    // DataTable Başlatma
    $('#dataTable').DataTable({
        data: tableData,
        language: {
            url: '//cdn.datatables.net/plug-ins/1.13.4/i18n/tr.json' // Türkçe Arayüz
        },
        columns: [
            { title: "Meslek" },
            { title: "Şehir" },
            { title: "İlçe" },
            { title: "Mevcut" },
            { title: "Norm" },
            { title: "Fark" },
            { title: "Durum" }
        ],
        pageLength: 25,
        order: [[5, 'desc']], // Farka göre sıralı başla
        responsive: true,
        lengthChange: true,
        searching: true,
        info: true
    });
}

function createCharts(summaryData) {
    // Calculate totals for pie chart
    let totalSurplus = 0;
    let totalDeficit = 0;
    let totalBalanced = 0;
    
    for (const [profession, stats] of Object.entries(summaryData.professions)) {
        totalSurplus += stats.surplus;
        totalDeficit += stats.deficit;
        totalBalanced += stats.balanced;
    }
    
    // Bar Chart (Top 10 cities with most surplus - using static data as fallback)
    const ctxBar = document.getElementById('barChart').getContext('2d');
    new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: dashboardData.surplus.labels.slice(0, 10), // Top 10
            datasets: [{
                label: 'Norm Fazlası Personel Sayısı',
                data: dashboardData.surplus.data.slice(0, 10),
                backgroundColor: '#009d63',
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // Pie Chart (Overall Status)
    const ctxPie = document.getElementById('pieChart').getContext('2d');
    new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: ['Fazla', 'Eksik', 'Dengede'],
            datasets: [{
                data: [totalSurplus, totalDeficit, totalBalanced],
                backgroundColor: ['#0d6efd', '#6f42c1', '#20c997'],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}

// Fallback function to use static data if API fails
function loadStaticData() {
    // --- 1. KPI Hesaplamaları ---
    let totalSurplus = 0;
    let totalDeficit = 0;
    let totalBalanced = 0;
    
    // Status verisinden toplamları çek
    dashboardData.status.norm_fazlasi.forEach(num => totalSurplus += num);
    dashboardData.status.norm_eksigi.forEach(num => totalDeficit += num);
    dashboardData.status.dengede.forEach(num => totalBalanced += num);

    document.getElementById('total-count').innerText = (totalSurplus + totalDeficit + totalBalanced).toLocaleString();
    document.getElementById('total-surplus').innerText = totalSurplus.toLocaleString();
    document.getElementById('total-deficit').innerText = totalDeficit.toLocaleString();
    document.getElementById('total-balanced').innerText = totalBalanced.toLocaleString();

    // --- 2. Akıllı Tablo (DataTables) Hazırlığı ---
    // Scatter verisi detaylı ilçe verisini içeriyor, oradan tabloyu dolduralım.
    const tableData = [];
    
    for (const [meslek, items] of Object.entries(dashboardData.scatter)) {
        items.forEach(item => {
            // Label formatı "Sehir-Ilce" şeklinde geliyordu, ayıralım
            const parts = item.label.split('-');
            const sehir = parts[0];
            const ilce = parts.length > 1 ? parts[1] : '';
            
            // Fark hesabı (Mevcut - Norm)
            const fark = item.y - item.x;
            let durumHtml = '';
            
            if (fark > 0.5) {
                durumHtml = '<span class="badge-fazla">Fazla</span>';
            } else if (fark < -0.5) {
                durumHtml = '<span class="badge-eksik">Eksik</span>';
            } else {
                durumHtml = '<span class="badge-denge">Dengede</span>';
            }

            tableData.push([
                meslek,
                sehir,
                ilce,
                item.y, // Mevcut
                item.x, // Norm
                fark.toFixed(1),
                durumHtml
            ]);
        });
    }

    // DataTable Başlatma
    $('#dataTable').DataTable({
        data: tableData,
        language: {
            url: '//cdn.datatables.net/plug-ins/1.13.4/i18n/tr.json' // Türkçe Arayüz
        },
        columns: [
            { title: "Meslek" },
            { title: "Şehir" },
            { title: "İlçe" },
            { title: "Mevcut" },
            { title: "Norm" },
            { title: "Fark" },
            { title: "Durum" }
        ],
        pageLength: 25,
        order: [[5, 'desc']], // Farka göre sıralı başla
        responsive: true,
        lengthChange: true,
        searching: true,
        info: true
    });

    // --- 3. Grafikler ---

    // Bar Chart (Top Surplus Cities)
    const ctxBar = document.getElementById('barChart').getContext('2d');
    new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: dashboardData.surplus.labels.slice(0, 10), // Top 10
            datasets: [{
                label: 'Norm Fazlası Personel Sayısı',
                data: dashboardData.surplus.data.slice(0, 10),
                backgroundColor: '#009d63',
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });

    // Pie Chart (Overall Status)
    const ctxPie = document.getElementById('pieChart').getContext('2d');
    new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: ['Fazla', 'Eksik', 'Dengede'],
            datasets: [{
                data: [totalSurplus, totalDeficit, totalBalanced],
                backgroundColor: ['#0d6efd', '#6f42c1', '#20c997'],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}