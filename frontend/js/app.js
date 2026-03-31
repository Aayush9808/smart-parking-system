/* =====================================================================
   ParkSense AI — Dashboard Logic
   ===================================================================== */

const API = "";  // same origin

// ── State ────────────────────────────────────────────────────────────────────
let historyChart = null;
let forecastChart = null;

// Chart.js global defaults for dark theme
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#1e293b';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// ── Utilities ────────────────────────────────────────────────────────────────
function toast(msg, ms = 3000) {
    const el = document.getElementById("toast");
    document.getElementById("toastMessage").textContent = msg;
    el.classList.add("toast-show");
    setTimeout(() => el.classList.remove("toast-show"), ms);
}

function setLoading(btnEl, loading) {
    if (loading) {
        btnEl.dataset.orig = btnEl.innerHTML;
        btnEl.innerHTML = '<span class="spinner"></span>Working…';
        btnEl.disabled = true;
    } else {
        btnEl.innerHTML = btnEl.dataset.orig || btnEl.innerHTML;
        btnEl.disabled = false;
    }
}

// ── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
    const el = document.getElementById("clockDisplay");
    if (el) el.textContent = new Date().toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
}
setInterval(updateClock, 1000);
updateClock();

// ── Generate Sample ──────────────────────────────────────────────────────────
async function generateSample() {
    const btn = event.currentTarget;
    setLoading(btn, true);
    try {
        const res = await fetch(`${API}/api/detect/sample`);
        const data = await res.json();
        renderDetectionResult(data);
        toast("Sample parking scene generated!");
    } catch (e) {
        toast("Error: " + e.message);
    }
    setLoading(btn, false);
}

// ── Upload Image ─────────────────────────────────────────────────────────────
async function uploadImage(evt) {
    const file = evt.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    toast("Uploading & detecting…");
    try {
        const res = await fetch(`${API}/api/detect`, { method: "POST", body: form });
        const data = await res.json();
        renderDetectionResult(data);
        toast(`Detected ${data.detections.length} vehicles (${data.mode})`);
    } catch (e) {
        toast("Error: " + e.message);
    }
    evt.target.value = "";
}

// ── Run Simulation ───────────────────────────────────────────────────────────
async function runSimulation() {
    const btn = event.currentTarget;
    setLoading(btn, true);
    try {
        const res = await fetch(`${API}/api/parking/simulate`, { method: "POST" });
        const data = await res.json();
        toast(data.message);
        await loadHistory();
        await loadHeatmap();
    } catch (e) {
        toast("Error: " + e.message);
    }
    setLoading(btn, false);
}

// ── Train Model ──────────────────────────────────────────────────────────────
async function trainModel() {
    const btn = event.currentTarget;
    setLoading(btn, true);
    try {
        const res = await fetch(`${API}/api/prediction/train`, { method: "POST" });
        const data = await res.json();
        if (data.status === "trained") {
            document.getElementById("modelAccuracy")?.textContent && (document.getElementById("modelAccuracy").textContent = data.test_score.toFixed(2));
            toast(`Model trained! R²=${data.test_score.toFixed(3)}`);
            await loadForecast();
            await loadSlotPredictions();
            await loadPeakHours();
        } else {
            toast(data.message || "Training failed");
        }
    } catch (e) {
        toast("Error: " + e.message);
    }
    setLoading(btn, false);
}

// ── Refresh All ──────────────────────────────────────────────────────────────
async function refreshAll() {
    await Promise.all([loadHistory(), loadHeatmap(), loadForecast(), loadSlotPredictions(), loadPeakHours()]);
    toast("Refreshed");
}

// ── Render detection result ──────────────────────────────────────────────────
function renderDetectionResult(data) {
    // Image
    const container = document.getElementById("parkingImageContainer");
    if (data.result_image) {
        container.innerHTML = `<img src="data:image/jpeg;base64,${data.result_image}" class="w-full rounded-lg" alt="Parking lot detection result">`;
    }

    // Stats
    const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    setEl("vehicleCount", data.total_vehicles ?? data.total_occupied);
    setEl("occupiedCount", data.total_occupied);
    setEl("availableCount", data.total_empty);
    setEl("totalSlots", data.total_slots);
    setEl("occupancyPct", data.occupancy_pct !== undefined ? data.occupancy_pct + "%" : "--");

    // Confidence note
    const noteEl = document.getElementById("confidenceNote");
    const noteText = document.getElementById("confidenceNoteText");
    if (noteEl && data.confidence_note) {
        noteText.textContent = data.confidence_note;
        noteEl.classList.remove("hidden");
    } else if (noteEl) {
        noteEl.classList.add("hidden");
    }

    // Detection mode badge
    const modeEl = document.getElementById("detectionMode");
    if (modeEl) {
        if (data.mode === "yolo_detection") {
            modeEl.textContent = "YOLOv8 + SAHI";
            modeEl.className = "ml-auto text-[10px] font-medium text-blue-400/70";
        } else {
            modeEl.textContent = "Simulated";
            modeEl.className = "ml-auto text-[10px] font-medium text-amber-400/70";
        }
    }

    // Status badge
    const badge = document.getElementById("statusBadge");
    badge.className = "badge badge-live";
    badge.textContent = "Live";

    // Slot grid
    renderSlotGrid(data.slots);
}

// ── Slot grid ────────────────────────────────────────────────────────────────
function renderSlotGrid(slots) {
    const grid = document.getElementById("slotGrid");
    grid.innerHTML = "";
    if (!slots || !slots.length) {
        grid.innerHTML = '<div class="empty-state col-span-3"><p>No data</p></div>';
        return;
    }
    slots.forEach(s => {
        const isOcc = s.status === "occupied";
        const cls = isOcc ? "slot-occupied" : "slot-empty";
        const icon = isOcc
            ? '<svg class="w-5 h-5 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 17a2 2 0 11-4 0 2 2 0 014 0zM19 17a2 2 0 11-4 0 2 2 0 014 0z"/><path stroke-linecap="round" stroke-linejoin="round" d="M13 16V6a1 1 0 00-1-1H4a1 1 0 00-1 1v10m10 0h-3m3 0h5a1 1 0 001-1v-4a1 1 0 00-.8-.97l-3.2-.64A1 1 0 0015 10V6"/></svg>'
            : '<svg class="w-5 h-5 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>';
        const conf = s.confidence !== undefined ? `${Math.round(s.confidence * 100)}%` : "";
        const classLabel = (isOcc && s.class_name) ? s.class_name : "";
        grid.innerHTML += `
            <div class="slot-card ${cls}">
                ${icon}
                <div class="slot-name mt-1">${s.name}</div>
                <div class="slot-status">${isOcc ? classLabel || 'OCCUPIED' : 'FREE'}</div>
                <div class="slot-conf">${conf}</div>
            </div>`;
    });
}

// ── History chart ────────────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch(`${API}/api/parking/history?hours=24`);
        const data = await res.json();
        if (!data.data || !data.data.length) return;

        const labels = data.data.map(d => {
            const dt = new Date(d.timestamp);
            return dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        });
        const occ = data.data.map(d => d.total_occupied);
        const emp = data.data.map(d => d.total_empty);

        const ctx = document.getElementById("historyChart").getContext("2d");
        if (historyChart) historyChart.destroy();
        historyChart = new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [
                    { label: "Occupied", data: occ, borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,0.1)", fill: true, tension: 0.3 },
                    { label: "Empty", data: emp, borderColor: "#10b981", backgroundColor: "rgba(16,185,129,0.1)", fill: true, tension: 0.3 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#9ca3af" } } },
                scales: {
                    x: { ticks: { color: "#6b7280", maxTicksLimit: 12 }, grid: { color: "#1f2937" } },
                    y: { ticks: { color: "#6b7280" }, grid: { color: "#1f2937" }, beginAtZero: true },
                },
            },
        });
    } catch (_) {}
}

// ── Forecast chart ───────────────────────────────────────────────────────────
async function loadForecast() {
    try {
        const res = await fetch(`${API}/api/prediction/forecast?hours=6`);
        const data = await res.json();
        if (data.status !== "ok" || !data.predictions.length) return;

        const labels = data.predictions.map(p => p.hour);
        const occ = data.predictions.map(p => p.predicted_occupied);
        const emp = data.predictions.map(p => p.predicted_empty);

        const ctx = document.getElementById("forecastChart").getContext("2d");
        if (forecastChart) forecastChart.destroy();
        forecastChart = new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [
                    { label: "Predicted Occupied", data: occ, borderColor: "#f59e0b", backgroundColor: "rgba(245,158,11,0.1)", fill: true, tension: 0.3, borderDash: [5, 3] },
                    { label: "Predicted Empty", data: emp, borderColor: "#06b6d4", backgroundColor: "rgba(6,182,212,0.1)", fill: true, tension: 0.3, borderDash: [5, 3] },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#9ca3af" } } },
                scales: {
                    x: { ticks: { color: "#6b7280", maxTicksLimit: 12 }, grid: { color: "#1f2937" } },
                    y: { ticks: { color: "#6b7280" }, grid: { color: "#1f2937" }, beginAtZero: true },
                },
            },
        });
    } catch (_) {}
}

// ── Slot predictions ─────────────────────────────────────────────────────────
async function loadSlotPredictions() {
    const container = document.getElementById("predictionsContainer");
    try {
        const res = await fetch(`${API}/api/prediction/slots`);
        const data = await res.json();
        if (data.status !== "ok" || !data.slot_predictions.length) return;

        container.innerHTML = "";
        data.slot_predictions.forEach(s => {
            let msg = "";
            let icon = '<svg class="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>';
            let accent = "text-emerald-400";
            if (s.current_status === "occupied") {
                icon = '<svg class="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';
                accent = "text-amber-400";
                if (s.prediction && s.prediction.estimated_free_in_minutes) {
                    msg = `Free in ~${s.prediction.estimated_free_in_minutes} min`;
                } else {
                    msg = "No free-up predicted";
                    accent = "text-red-400";
                }
            } else {
                msg = "Available now";
            }
            container.innerHTML += `
                <div class="pred-card">
                    <div class="flex items-center gap-2">
                        ${icon}
                        <span class="font-semibold text-sm">${s.slot_name}</span>
                    </div>
                    <p class="text-xs ${accent} mt-1.5">${msg}</p>
                </div>`;
        });
    } catch (_) {}
}

// ── Heatmap ──────────────────────────────────────────────────────────────────
async function loadHeatmap() {
    try {
        const res = await fetch(`${API}/api/parking/heatmap`);
        const data = await res.json();
        if (data.status !== "ok") return;

        const canvas = document.getElementById("heatmapCanvas");
        const ctx = canvas.getContext("2d");
        const grid = data.heatmap;        // 7 x 24
        const days = data.days;

        const cellW = 28, cellH = 24;
        const padLeft = 80, padTop = 30;
        canvas.width = padLeft + 24 * cellW + 10;
        canvas.height = padTop + 7 * cellH + 10;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#111827";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Hour labels
        ctx.fillStyle = "#6b7280";
        ctx.font = "11px Inter, sans-serif";
        for (let h = 0; h < 24; h++) {
            ctx.fillText(`${h}`, padLeft + h * cellW + 8, padTop - 8);
        }

        // Day labels + cells
        for (let d = 0; d < 7; d++) {
            ctx.fillStyle = "#9ca3af";
            ctx.fillText(days[d].slice(0, 3), 10, padTop + d * cellH + 17);
            for (let h = 0; h < 24; h++) {
                const val = grid[d][h];
                const r = Math.round(239 * val + 16 * (1 - val));
                const g = Math.round(68 * val + 185 * (1 - val));
                const b = Math.round(68 * val + 129 * (1 - val));
                ctx.fillStyle = `rgb(${r},${g},${b})`;
                ctx.fillRect(padLeft + h * cellW, padTop + d * cellH, cellW - 2, cellH - 2);
                // value text
                if (val > 0) {
                    ctx.fillStyle = val > 0.5 ? "#fff" : "#64748b";
                    ctx.font = "9px Inter, sans-serif";
                    ctx.fillText((val * 100).toFixed(0), padLeft + h * cellW + 5, padTop + d * cellH + 15);
                    ctx.font = "11px Inter, sans-serif";
                }
            }
        }
    } catch (_) {}
}

// ── Peak hours ───────────────────────────────────────────────────────────────
async function loadPeakHours() {
    const container = document.getElementById("peakHoursContainer");
    try {
        const res = await fetch(`${API}/api/prediction/peak_hours`);
        const data = await res.json();
        if (data.status !== "ok") return;

        let html = "";
        if (data.busiest_hour) {
            html += `<div class="pred-card mb-3">
                <div class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Busiest Hour</div>
                <div class="text-2xl font-bold text-white mt-1">${data.busiest_hour.hour_label}</div>
                <div class="text-xs text-amber-400 mt-0.5">${(data.busiest_hour.predicted_rate * 100).toFixed(0)}% occupancy</div>
            </div>`;
        }
        if (data.peak_hours.length) {
            html += `<div class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-2">Peak hours (&gt;70% full)</div>`;
            html += '<div class="flex flex-wrap gap-1.5">';
            data.peak_hours.forEach(p => {
                html += `<span class="px-2.5 py-1 bg-red-500/10 text-red-400 border border-red-500/20 rounded-md text-xs font-medium">${p.hour_label}</span>`;
            });
            html += "</div>";
        } else {
            html += '<p class="text-slate-600 text-sm mt-2">No peak hours detected</p>';
        }
        container.innerHTML = html;
    } catch (_) {}
}

// ── Auto-load on page ready ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadHistory();
    loadHeatmap();
    loadForecast();
    loadSlotPredictions();
    loadPeakHours();
});
