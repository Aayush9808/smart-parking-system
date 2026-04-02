/**
 * ParkSense AI — Dashboard Logic
 * Handles detection, analytics, charts, and slot grid rendering.
 */

const API = "";

// ─── DOM Elements ────────────────────────────────────────────
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const sampleBtn = document.getElementById("sample-btn");
const simulateBtn = document.getElementById("simulate-btn");
const trainBtn = document.getElementById("train-btn");

const resultImage = document.getElementById("result-image");
const placeholder = document.getElementById("placeholder");
const diagSection = document.getElementById("diagnostics");
const diagGrid = document.getElementById("diag-grid");
const slotGrid = document.getElementById("slot-grid");
const trainResult = document.getElementById("train-result");
const peakSection = document.getElementById("peak-hours");
const peakList = document.getElementById("peak-list");

// ─── Charts ──────────────────────────────────────────────────
let historyChart = null;
let forecastChart = null;
let heatmapChart = null;

// ─── Event Listeners ─────────────────────────────────────────
fileInput.addEventListener("change", handleUpload);
sampleBtn.addEventListener("click", handleSample);
simulateBtn.addEventListener("click", handleSimulate);
trainBtn.addEventListener("click", handleTrain);

// ─── Upload Image ────────────────────────────────────────────
async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    showLoading("Detecting cars...");
    const form = new FormData();
    form.append("file", file);

    try {
        const res = await fetch(`${API}/api/detect`, { method: "POST", body: form });
        const data = await res.json();
        if (data.success) {
            renderDetectionResult(data);
        } else {
            alert(data.error || "Detection failed");
        }
    } catch (err) {
        alert("Error: " + err.message);
    }
    fileInput.value = "";
}

// ─── Generate Sample ─────────────────────────────────────────
async function handleSample() {
    showLoading("Generating sample...");
    try {
        const res = await fetch(`${API}/api/detect/sample`);
        const data = await res.json();
        if (data.success) {
            renderDetectionResult(data);
        }
    } catch (err) {
        alert("Error: " + err.message);
    }
}

// ─── Simulate History ────────────────────────────────────────
async function handleSimulate() {
    simulateBtn.disabled = true;
    simulateBtn.textContent = "⏳ Simulating...";
    try {
        const res = await fetch(`${API}/api/parking/simulate`, { method: "POST" });
        const data = await res.json();
        simulateBtn.textContent = `✅ ${data.records_generated} records`;
        loadHistory();
        loadHeatmap();
        setTimeout(() => { simulateBtn.textContent = "📊 Simulate 14 Days"; simulateBtn.disabled = false; }, 2000);
    } catch (err) {
        simulateBtn.textContent = "❌ Failed";
        setTimeout(() => { simulateBtn.textContent = "📊 Simulate 14 Days"; simulateBtn.disabled = false; }, 2000);
    }
}

// ─── Train Predictor ─────────────────────────────────────────
async function handleTrain() {
    trainBtn.disabled = true;
    trainBtn.textContent = "⏳ Training...";
    try {
        const res = await fetch(`${API}/api/prediction/train`, { method: "POST" });
        const data = await res.json();
        if (data.success) {
            trainResult.classList.remove("hidden");
            trainResult.innerHTML = `
                <div class="metric"><span>Status</span><span class="metric-val">✅ Trained</span></div>
                <div class="metric"><span>Samples</span><span class="metric-val">${data.samples}</span></div>
                <div class="metric"><span>R² Score</span><span class="metric-val">${data.r2_score}</span></div>
                <div class="metric"><span>Features</span><span class="metric-val">${data.features}</span></div>
            `;
            loadForecast();
            loadPeakHours();
        } else {
            trainResult.classList.remove("hidden");
            trainResult.innerHTML = `<div class="metric"><span>Error</span><span class="metric-val" style="color:var(--red)">${data.error}</span></div>`;
        }
        trainBtn.textContent = "🧠 Train Predictor";
        trainBtn.disabled = false;
    } catch (err) {
        trainBtn.textContent = "❌ Failed";
        setTimeout(() => { trainBtn.textContent = "🧠 Train Predictor"; trainBtn.disabled = false; }, 2000);
    }
}

// ─── Render Detection Result ─────────────────────────────────
function renderDetectionResult(data) {
    // Show image
    placeholder.classList.add("hidden");
    resultImage.classList.remove("hidden");
    resultImage.src = "data:image/jpeg;base64," + data.result_image;

    // Update stats
    const a = data.analytics;
    document.getElementById("stat-cars").textContent = a.cars_detected;
    document.getElementById("stat-slots").textContent = a.total_slots;
    document.getElementById("stat-occupied").textContent = a.occupied;
    document.getElementById("stat-empty").textContent = a.empty;
    document.getElementById("stat-rate").textContent = a.occupancy_percent + "%";

    // Render diagnostics
    if (data.diagnostics) {
        diagSection.classList.remove("hidden");
        diagGrid.innerHTML = "";
        for (const [key, val] of Object.entries(data.diagnostics)) {
            const item = document.createElement("div");
            item.className = "diag-item";
            item.innerHTML = `<span class="diag-label">${key.replace(/_/g, " ")}</span><br><span class="diag-value">${val}</span>`;
            diagGrid.appendChild(item);
        }
    }

    // Render slot grid
    renderSlotGrid(data.slots, data.grid_info);
}

// ─── Render Slot Grid ────────────────────────────────────────
function renderSlotGrid(slots, gridInfo) {
    slotGrid.innerHTML = "";

    if (!slots || slots.length === 0) {
        slotGrid.innerHTML = '<p class="slot-empty-msg">No slots generated</p>';
        return;
    }

    const cols = gridInfo ? gridInfo.cols : Math.ceil(Math.sqrt(slots.length));
    slotGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    for (const slot of slots) {
        const card = document.createElement("div");
        card.className = `slot-card ${slot.status}`;
        card.innerHTML = `
            <div class="slot-name">${slot.name}</div>
            <div class="slot-status">${slot.status}</div>
        `;
        slotGrid.appendChild(card);
    }
}

// ─── Load History Chart ──────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch(`${API}/api/parking/history`);
        const data = await res.json();
        if (!data.records || data.records.length === 0) return;

        const labels = data.records.map(r => {
            const d = new Date(r.timestamp);
            return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit" });
        });
        const values = data.records.map(r => Math.round(r.occupancy_rate * 100));

        const ctx = document.getElementById("history-chart").getContext("2d");
        if (historyChart) historyChart.destroy();
        historyChart = new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [{
                    label: "Occupancy %",
                    data: values,
                    borderColor: "#3b82f6",
                    backgroundColor: "rgba(59,130,246,0.1)",
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    borderWidth: 2,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { min: 0, max: 100, ticks: { color: "#8892a4" }, grid: { color: "#2a2e3f" } },
                },
            },
        });
    } catch (err) { console.error("History:", err); }
}

// ─── Load Forecast Chart ─────────────────────────────────────
async function loadForecast() {
    try {
        const res = await fetch(`${API}/api/prediction/forecast?hours=12`);
        const data = await res.json();
        if (!data.success) return;

        const labels = data.predictions.map(p => p.hour + ":00");
        const values = data.predictions.map(p => p.predicted_percent);

        const ctx = document.getElementById("forecast-chart").getContext("2d");
        if (forecastChart) forecastChart.destroy();
        forecastChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels,
                datasets: [{
                    label: "Predicted %",
                    data: values,
                    backgroundColor: values.map(v =>
                        v >= 80 ? "rgba(239,68,68,0.7)" :
                        v >= 50 ? "rgba(234,179,8,0.7)" :
                        "rgba(34,197,94,0.7)"
                    ),
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: "#8892a4" }, grid: { display: false } },
                    y: { min: 0, max: 100, ticks: { color: "#8892a4" }, grid: { color: "#2a2e3f" } },
                },
            },
        });
    } catch (err) { console.error("Forecast:", err); }
}

// ─── Load Heatmap ────────────────────────────────────────────
async function loadHeatmap() {
    try {
        const res = await fetch(`${API}/api/parking/heatmap`);
        const data = await res.json();

        const ctx = document.getElementById("heatmap-chart").getContext("2d");
        if (heatmapChart) heatmapChart.destroy();

        // Flatten matrix into scatter data points
        const points = [];
        for (let d = 0; d < 7; d++) {
            for (let h = 0; h < 24; h++) {
                points.push({ x: h, y: d, v: data.matrix[d][h] });
            }
        }

        heatmapChart = new Chart(ctx, {
            type: "scatter",
            data: {
                datasets: [{
                    data: points.map(p => ({ x: p.x, y: p.y })),
                    backgroundColor: points.map(p => {
                        const v = p.v;
                        if (v >= 0.8) return "rgba(239,68,68,0.8)";
                        if (v >= 0.6) return "rgba(234,179,8,0.8)";
                        if (v >= 0.3) return "rgba(59,130,246,0.8)";
                        return "rgba(34,197,94,0.4)";
                    }),
                    pointRadius: 8,
                    pointStyle: "rect",
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false }, tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const p = points[ctx.dataIndex];
                            return `${data.days[p.y]} ${p.x}:00 — ${Math.round(p.v * 100)}%`;
                        }
                    }
                }},
                scales: {
                    x: { min: -0.5, max: 23.5, ticks: { color: "#8892a4", stepSize: 3 }, grid: { color: "#2a2e3f" }, title: { display: true, text: "Hour", color: "#8892a4" } },
                    y: { min: -0.5, max: 6.5, ticks: { color: "#8892a4", callback: (v) => data.days[v] || "" }, grid: { color: "#2a2e3f" } },
                },
            },
        });
    } catch (err) { console.error("Heatmap:", err); }
}

// ─── Load Peak Hours ─────────────────────────────────────────
async function loadPeakHours() {
    try {
        const res = await fetch(`${API}/api/prediction/peak_hours`);
        const data = await res.json();
        if (!data.success) return;

        peakSection.classList.remove("hidden");
        peakList.innerHTML = "";
        for (const p of data.peak_hours) {
            const item = document.createElement("div");
            item.className = "peak-item";
            item.innerHTML = `
                <span class="peak-label">${p.label}</span>
                <span class="peak-value">${Math.round(p.predicted_occupancy * 100)}%</span>
            `;
            peakList.appendChild(item);
        }
    } catch (err) { console.error("Peak hours:", err); }
}

// ─── Helpers ─────────────────────────────────────────────────
function showLoading(msg) {
    placeholder.classList.remove("hidden");
    resultImage.classList.add("hidden");
    placeholder.innerHTML = `<div class="loading"><div class="spinner"></div>${msg}</div>`;
}

// ─── Initial Load ────────────────────────────────────────────
loadHistory();
loadHeatmap();
