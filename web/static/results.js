/**
 * ProteaFlow Results — unified JS for rankings, scatter, radar, tools, detail, rerank.
 *
 * Expects these globals injected by Jinja2:
 *   JOB_ID, TOOL_COLORS, SCATTER_PRESETS, HIGHER_BETTER, LOWER_BETTER,
 *   DEFAULT_COLUMNS, RADAR_AXES, TOOL_METRICS
 */

// ============================================================
// Section 1: Data Store
// ============================================================
let allData = [];
let filteredData = [];
let filterState = {
    min_combined: -10, min_iptm: 0, min_plddt: 0,
    max_dg: 100, max_site_pae: 50,
    active_tools: [],
    ss_bias: 'any',
    include_unranked: false,
};
let selectedDesign = null;
let gridApi = null;
let activeTab = 'rankings';
const dirtyTabs = new Set();

// ============================================================
// Section 2: Filter Engine
// ============================================================
function applyFilters() {
    filteredData = allData.filter(row => {
        // Unranked filter
        if (!filterState.include_unranked && row.rank == null) return false;

        // Score thresholds (pass NaN through)
        if (row.combined_score != null && row.combined_score < filterState.min_combined) return false;
        if (row.boltz_iptm != null && row.boltz_iptm < filterState.min_iptm) return false;
        if (row.esmfold_plddt != null && row.esmfold_plddt < filterState.min_plddt) return false;
        if (row.rosetta_dG != null && row.rosetta_dG > filterState.max_dg) return false;
        if (row.boltz_site_mean_pae != null && row.boltz_site_mean_pae > filterState.max_site_pae) return false;

        // Tool filter
        if (filterState.active_tools.length > 0 && !filterState.active_tools.includes(row.tool)) return false;

        // SS bias
        if (filterState.ss_bias === 'helix' && row.binder_helix_frac != null && row.binder_helix_frac < 0.4) return false;
        if (filterState.ss_bias === 'sheet' && row.binder_sheet_frac != null && row.binder_sheet_frac < 0.3) return false;
        if (filterState.ss_bias === 'balanced') {
            if (row.binder_helix_frac != null && row.binder_helix_frac >= 0.6) return false;
            if (row.binder_sheet_frac != null && row.binder_sheet_frac >= 0.4) return false;
        }
        return true;
    });

    // Update active tab, mark others dirty
    updateActiveTab();
    ['rankings', 'scatter', 'radar', 'tools'].forEach(t => {
        if (t !== activeTab) dirtyTabs.add(t);
    });
    updateFilterStatus();
}

function updateActiveTab() {
    switch (activeTab) {
        case 'rankings': updateGrid(); break;
        case 'scatter': updateScatter(); break;
        case 'radar': updateRadar(); break;
        case 'tools': updateToolComparison(); break;
    }
}

function updateFilterStatus() {
    const el = document.getElementById('filter-status');
    if (el) el.textContent = `Showing ${filteredData.length} / ${allData.length} designs`;
}

function initFilterPanel() {
    // Tool checkboxes
    const toolDiv = document.getElementById('filter-tools');
    if (!toolDiv) return;
    const tools = [...new Set(allData.map(d => d.tool).filter(Boolean))].sort();
    filterState.active_tools = tools.slice();
    toolDiv.innerHTML = '';
    tools.forEach(tool => {
        const color = TOOL_COLORS[tool] || '#888';
        const label = document.createElement('label');
        label.className = 'flex items-center gap-2 cursor-pointer text-sm';
        label.innerHTML = `<input type="checkbox" checked data-tool="${tool}" class="tool-cb border-slate-600">
            <span style="color:${color}">${tool}</span>`;
        toolDiv.appendChild(label);
    });

    // Wire filter controls
    document.querySelectorAll('.filter-input').forEach(el => {
        el.addEventListener('change', readFiltersAndApply);
    });
    document.querySelectorAll('.tool-cb').forEach(el => {
        el.addEventListener('change', readFiltersAndApply);
    });
    document.querySelectorAll('input[name="filter_ss"]').forEach(el => {
        el.addEventListener('change', readFiltersAndApply);
    });
    document.getElementById('f-include-unranked')?.addEventListener('change', readFiltersAndApply);

    // All/None buttons
    document.getElementById('tools-all')?.addEventListener('click', () => {
        document.querySelectorAll('.tool-cb').forEach(cb => cb.checked = true);
        readFiltersAndApply();
    });
    document.getElementById('tools-none')?.addEventListener('click', () => {
        document.querySelectorAll('.tool-cb').forEach(cb => cb.checked = false);
        readFiltersAndApply();
    });
    document.getElementById('filter-reset')?.addEventListener('click', resetFilters);
}

function readFiltersAndApply() {
    filterState.min_combined = parseFloat(document.getElementById('f-min-combined')?.value) || -10;
    filterState.min_iptm = parseFloat(document.getElementById('f-min-iptm')?.value) || 0;
    filterState.min_plddt = parseFloat(document.getElementById('f-min-plddt')?.value) || 0;
    filterState.max_dg = parseFloat(document.getElementById('f-max-dg')?.value) || 100;
    filterState.max_site_pae = parseFloat(document.getElementById('f-max-site-pae')?.value) || 50;
    filterState.active_tools = [...document.querySelectorAll('.tool-cb:checked')].map(cb => cb.dataset.tool);
    const ss = document.querySelector('input[name="filter_ss"]:checked');
    filterState.ss_bias = ss ? ss.value : 'any';
    filterState.include_unranked = document.getElementById('f-include-unranked')?.checked || false;
    applyFilters();
}

function resetFilters() {
    document.getElementById('f-min-combined').value = -10;
    document.getElementById('f-min-iptm').value = 0;
    document.getElementById('f-min-plddt').value = 0;
    document.getElementById('f-max-dg').value = 100;
    document.getElementById('f-max-site-pae').value = 50;
    const unrankedCb = document.getElementById('f-include-unranked');
    if (unrankedCb) unrankedCb.checked = false;
    document.querySelectorAll('.tool-cb').forEach(cb => cb.checked = true);
    document.querySelector('input[name="filter_ss"][value="any"]').checked = true;
    readFiltersAndApply();
}

// ============================================================
// Section 3: Rankings Grid
// ============================================================
function scoreColor(col, val, allVals) {
    if (val == null) return null;
    const nums = allVals.filter(v => v != null);
    if (!nums.length) return null;
    const min = Math.min(...nums), max = Math.max(...nums);
    if (min === max) return null;
    let t = (val - min) / (max - min);
    if (LOWER_BETTER.has(col)) t = 1 - t;
    if (!HIGHER_BETTER.has(col) && !LOWER_BETTER.has(col)) return null;
    const r = Math.round(220 - 180 * t);
    const g = Math.round(60 + 160 * t);
    return `rgba(${r}, ${g}, 80, 0.25)`;
}

let colArrays = {};

function initGrid() {
    const allCols = Object.keys(allData[0] || {});
    allCols.forEach(c => { colArrays[c] = allData.map(r => r[c]); });

    const columnDefs = allCols
        .filter(c => DEFAULT_COLUMNS.includes(c) || c === 'binder_sequence')
        .map(c => ({
            field: c, headerName: c, sortable: true, filter: true, resizable: true,
            width: c === 'binder_sequence' ? 200 : c === 'design_id' ? 180 : 110,
            hide: c === 'binder_sequence',
            cellStyle: params => {
                const bg = scoreColor(c, params.value, colArrays[c]);
                return bg ? { backgroundColor: bg } : null;
            },
            valueFormatter: params => {
                if (params.value == null) return '';
                if (typeof params.value === 'number' && params.value.toFixed) return params.value.toFixed(3);
                return params.value;
            },
        }));

    const gridDiv = document.getElementById('rankings-grid');
    gridApi = agGrid.createGrid(gridDiv, {
        columnDefs, rowData: filteredData,
        defaultColDef: { sortable: true, filter: true, resizable: true },
        rowSelection: 'single',
        onRowClicked: (e) => selectDesign(e.data),
        animateRows: true, theme: 'legacy',
    });
}

function updateGrid() {
    if (gridApi) gridApi.setGridOption('rowData', filteredData);
}

// ============================================================
// Section 4: Scatter Plot
// ============================================================
function updateScatter() {
    const xCol = document.getElementById('scatter-x')?.value;
    const yCol = document.getElementById('scatter-y')?.value;
    if (!xCol || !yCol || !filteredData.length) return;

    const tools = [...new Set(filteredData.map(d => d.tool))];
    const traces = tools.map(tool => {
        const pts = filteredData.filter(d => d.tool === tool && d[xCol] != null && d[yCol] != null);
        return {
            x: pts.map(d => d[xCol]), y: pts.map(d => d[yCol]),
            text: pts.map(d => d.design_id), customdata: pts,
            mode: 'markers', type: 'scatter', name: tool,
            marker: { color: TOOL_COLORS[tool] || '#888', size: 8, opacity: 0.8 },
        };
    });

    const layout = {
        xaxis: { title: xCol, color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: yCol, color: '#94a3b8', gridcolor: '#1e293b' },
        plot_bgcolor: '#0f172a', paper_bgcolor: '#0f172a',
        font: { color: '#94a3b8' }, legend: { font: { color: '#cbd5e1' } },
        margin: { t: 30 },
    };

    Plotly.newPlot('scatter-plot', traces, layout, { responsive: true });

    // Click to select design
    document.getElementById('scatter-plot').on('plotly_click', (data) => {
        if (data.points.length > 0) {
            const d = data.points[0].customdata;
            if (d) selectDesign(d);
        }
    });
}

function initScatterControls() {
    const numCols = Object.keys(allData[0] || {}).filter(c =>
        allData.some(r => typeof r[c] === 'number'));
    const xSel = document.getElementById('scatter-x');
    const ySel = document.getElementById('scatter-y');
    numCols.forEach(c => { xSel.add(new Option(c, c)); ySel.add(new Option(c, c)); });
    xSel.value = 'boltz_iptm'; ySel.value = 'combined_score';

    const presetSel = document.getElementById('scatter-preset');
    Object.keys(SCATTER_PRESETS).forEach(name => presetSel.add(new Option(name, name)));
    presetSel.addEventListener('change', () => {
        const p = SCATTER_PRESETS[presetSel.value];
        if (p) { xSel.value = p[0]; ySel.value = p[1]; updateScatter(); }
    });
    xSel.addEventListener('change', updateScatter);
    ySel.addEventListener('change', updateScatter);
}

// ============================================================
// Section 5: Radar Chart — Adaptive Normalization
// ============================================================

// Precomputed per-axis scaling params (set once in initRadarScaling)
let radarScaling = {};  // col -> { floor, ceiling, higher }

/**
 * Compute adaptive scaling for each radar axis based on strategy:
 *   "fixed"        — use axis.min / axis.max directly
 *   "threshold"    — anchor user's filter threshold at 50%
 *   "data_driven"  — use dataset percentiles (5th=95%, median=50%, 95th=5%)
 */
function initRadarScaling() {
    radarScaling = {};
    for (const axis of RADAR_AXES) {
        const col = axis.col;
        const vals = allData.map(d => d[col]).filter(v => v != null).sort((a, b) => a - b);
        const s = axis.strategy || 'fixed';

        if (s === 'fixed') {
            radarScaling[col] = { floor: axis.min, ceiling: axis.max, higher: axis.higher };

        } else if (s === 'threshold') {
            // Anchor the threshold at midPct of the radar (default 50%)
            const thresh = axis.threshold_default || 0.5;
            const midPct = axis.threshold_pct || 0.5;
            if (axis.higher) {
                // higher=better: floor=0, threshold=midPct, ceiling based on remaining range
                radarScaling[col] = {
                    floor: 0,
                    mid: thresh,
                    midPct: midPct,
                    ceiling: Math.min(thresh + (1.0 - thresh), 1.0),  // scale to max 1.0
                    higher: true,
                    piecewise: true,
                };
            } else {
                // lower=better: 0→100%, threshold→(1-midPct), ceiling→0%
                radarScaling[col] = {
                    floor: 0,
                    mid: thresh,
                    midPct: midPct,
                    ceiling: thresh * (1 / midPct),  // so that ceiling maps to 0%
                    higher: false,
                    piecewise: true,
                };
            }

        } else if (s === 'data_driven') {
            if (vals.length < 5) {
                radarScaling[col] = { floor: 0, ceiling: 1, higher: axis.higher };
            } else {
                const p5  = vals[Math.floor(vals.length * 0.05)];
                const p50 = vals[Math.floor(vals.length * 0.50)];
                const p95 = vals[Math.floor(vals.length * 0.95)];
                radarScaling[col] = {
                    p5, p50, p95,
                    higher: axis.higher,
                    data_driven: true,
                };
            }
        }
    }
}

function radarNormalize(val, axis) {
    if (val == null) return null;
    const col = axis.col;
    const sc = radarScaling[col];
    if (!sc) return 0.5;

    let normed;

    if (sc.data_driven) {
        // Percentile-based: p5→95%, p50→50%, p95→5%
        // Linear interpolation between anchors
        if (sc.p5 === sc.p95) return 0.5;
        if (sc.higher) {
            // higher=better: high value = high radar (p5→5%, p95→95%)
            normed = (val - sc.p5) / (sc.p95 - sc.p5);
        } else {
            // lower=better: low value = high radar (p5→95%, p95→5%)
            normed = (sc.p95 - val) / (sc.p95 - sc.p5);
        }
        normed = Math.max(0, Math.min(1, normed));

    } else if (sc.piecewise) {
        // Threshold-anchored piecewise: mid→midPct (default 50%)
        const mp = sc.midPct || 0.5;
        if (sc.higher) {
            if (val <= sc.floor)       normed = 0;
            else if (val <= sc.mid)    normed = mp * (val - sc.floor) / (sc.mid - sc.floor);
            else if (val <= sc.ceiling) normed = mp + (1 - mp) * (val - sc.mid) / (sc.ceiling - sc.mid);
            else                        normed = 1;
        } else {
            // lower=better: 0→100%, mid→(1-mp), ceiling→0%
            if (val <= sc.floor)       normed = 1;
            else if (val <= sc.mid)    normed = 1 - mp * (val - sc.floor) / (sc.mid - sc.floor);
            else if (val <= sc.ceiling) normed = (1 - mp) - (1 - mp) * (val - sc.mid) / (sc.ceiling - sc.mid);
            else                        normed = 0;
        }

    } else {
        // Fixed linear
        const clamped = Math.max(sc.floor, Math.min(sc.ceiling, val));
        normed = (sc.ceiling > sc.floor) ? (clamped - sc.floor) / (sc.ceiling - sc.floor) : 0.5;
        if (!sc.higher) normed = 1.0 - normed;
    }

    return Math.max(0, Math.min(1, normed));
}

function radarMedianValues(dataSubset, axes) {
    return axes.map(axis => {
        const vals = dataSubset.map(d => d[axis.col]).filter(v => v != null);
        if (!vals.length) return 0.5;
        vals.sort((a, b) => a - b);
        const med = vals.length % 2 === 0
            ? (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2
            : vals[Math.floor(vals.length / 2)];
        return radarNormalize(med, axis) ?? 0.5;
    });
}

function radarDesignValues(design, axes) {
    return axes.map(axis => radarNormalize(design[axis.col], axis) ?? 0.5);
}

function updateRadar() {
    if (!filteredData.length) return;
    const mode = document.getElementById('radar-mode')?.value || 'none';
    const axes = RADAR_AXES.filter(a => allData.some(d => d[a.col] != null));
    if (axes.length < 3) return;

    const labels = axes.map(a => a.name);
    const traces = [];

    // Background traces
    if (mode === 'tiers') {
        const tierConfig = [
            { tier: 1, color: '#4CAF50', name: 'Tier 1 (Top 7.5%)' },
            { tier: 2, color: '#FFC107', name: 'Tier 2' },
            { tier: 3, color: '#F44336', name: 'Tier 3' },
        ];
        tierConfig.forEach(({ tier, color, name }) => {
            const subset = filteredData.filter(d => d.tier === tier);
            if (!subset.length) return;
            const vals = radarMedianValues(subset, axes);
            traces.push({
                type: 'scatterpolar', r: [...vals, vals[0]], theta: [...labels, labels[0]],
                name, line: { color, width: 1.5 }, fill: 'toself',
                fillcolor: color + '15', opacity: 0.7, marker: { size: 3 },
            });
        });
    } else if (mode === 'tools') {
        const tools = [...new Set(filteredData.map(d => d.tool))].sort();
        tools.forEach(tool => {
            const subset = filteredData.filter(d => d.tool === tool);
            if (subset.length < 3) return;
            const color = TOOL_COLORS[tool] || '#888';
            const vals = radarMedianValues(subset, axes);
            traces.push({
                type: 'scatterpolar', r: [...vals, vals[0]], theta: [...labels, labels[0]],
                name: tool, line: { color, width: 1.2 }, fill: 'toself',
                fillcolor: color + '10', opacity: 0.6, marker: { size: 2 },
            });
        });
    } else if (mode === 'top10') {
        const ranked = filteredData.filter(d => d.rank != null).sort((a, b) => a.rank - b.rank).slice(0, 10);
        if (ranked.length) {
            const vTop = radarMedianValues(ranked, axes);
            const vAll = radarMedianValues(filteredData, axes);
            traces.push({
                type: 'scatterpolar', r: [...vTop, vTop[0]], theta: [...labels, labels[0]],
                name: 'Top 10', line: { color: '#4CAF50', width: 1.5 }, fill: 'toself',
                fillcolor: '#4CAF5015', marker: { size: 3 },
            });
            traces.push({
                type: 'scatterpolar', r: [...vAll, vAll[0]], theta: [...labels, labels[0]],
                name: 'All', line: { color: '#9E9E9E', width: 1 }, fill: 'toself',
                fillcolor: '#9E9E9E0A', opacity: 0.5, marker: { size: 2 },
            });
        }
    }

    // Selected design overlay
    if (selectedDesign) {
        const vals = radarDesignValues(selectedDesign, axes);
        const tool = selectedDesign.tool || 'unknown';
        const color = TOOL_COLORS[tool] || '#FFFFFF';
        const rawTexts = axes.map(a => {
            const v = selectedDesign[a.col];
            return v != null ? (typeof v === 'number' ? v.toFixed(2) : String(v)) : 'N/A';
        });
        traces.push({
            type: 'scatterpolar', r: [...vals, vals[0]], theta: [...labels, labels[0]],
            name: selectedDesign.design_id || '?',
            line: { color, width: 3 }, fill: 'toself', fillcolor: color + '40',
            marker: { size: 6 }, text: [...rawTexts, rawTexts[0]],
            textposition: 'top center', textfont: { color, size: 10, family: 'monospace' },
            mode: 'lines+markers+text',
        });
    }

    Plotly.newPlot('radar-plot', traces, {
        polar: {
            radialaxis: { range: [0, 1.05], tickvals: [0.25, 0.5, 0.75, 1.0],
                         ticktext: ['25%', '50%', '75%', '100%'],
                         gridcolor: '#334155', color: '#94a3b8' },
            angularaxis: { gridcolor: '#334155', color: '#cbd5e1' },
            bgcolor: '#0f172a',
        },
        paper_bgcolor: '#0f172a', font: { color: '#94a3b8' },
        legend: { font: { color: '#cbd5e1' } }, margin: { t: 40 },
        title: { text: selectedDesign ? `Developability — ${selectedDesign.design_id}` : 'Developability Profile',
                 font: { color: '#e2e8f0', size: 14 } },
    }, { responsive: true });

    // Update axis legend
    updateRadarLegend(axes);
}

function initRadarControls() {
    document.getElementById('radar-mode')?.addEventListener('change', updateRadar);

    // Populate design selector
    const sel = document.getElementById('radar-design');
    if (!sel) return;
    sel.innerHTML = '<option value="">None</option>';
    const ranked = allData.filter(d => d.rank != null).sort((a, b) => a.rank - b.rank).slice(0, 50);
    ranked.forEach(d => {
        sel.add(new Option(`#${d.rank} ${d.design_id} (${d.tool})`, d.design_id));
    });
    sel.addEventListener('change', () => {
        const did = sel.value;
        if (!did) { selectedDesign = null; }
        else { selectedDesign = allData.find(d => d.design_id === did) || null; }
        updateRadar();
        if (selectedDesign) updateDetail();
    });
}

function updateRadarLegend(axes) {
    const el = document.getElementById('radar-legend');
    if (!el) return;

    let html = '<table class="text-xs w-full"><tbody>';
    for (const axis of axes) {
        const sc = radarScaling[axis.col];
        let scaleDesc = '';
        if (sc && sc.data_driven) {
            scaleDesc = `Data-driven (p5=${sc.p5?.toFixed?.(0) ?? '?'}, med=${sc.p50?.toFixed?.(0) ?? '?'}, p95=${sc.p95?.toFixed?.(0) ?? '?'})`;
        } else if (sc && sc.piecewise) {
            const t = sc.mid;
            scaleDesc = `Threshold @ ${typeof t === 'number' && t < 1 ? (t * 100).toFixed(0) + '%' : t} → 50%`;
        } else if (sc) {
            scaleDesc = `Fixed ${sc.floor}–${sc.ceiling}`;
        }
        const dir = axis.higher ? '↑ higher = better' : '↓ lower = better';
        const dirColor = axis.higher ? 'text-green-400' : 'text-amber-400';

        html += `<tr class="border-b border-slate-800">` +
            `<td class="py-1 pr-2 font-medium text-slate-300 whitespace-nowrap">${axis.name}</td>` +
            `<td class="py-1 pr-2 ${dirColor} whitespace-nowrap">${dir}</td>` +
            `<td class="py-1 pr-2 text-slate-500">${scaleDesc}</td>` +
            `<td class="py-1 text-slate-600">${axis.tip || ''}</td>` +
            `</tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
}

// ============================================================
// Section 6: Tool Comparison
// ============================================================
function updateToolComparison() {
    const mode = document.getElementById('tool-mode')?.value || 'box';
    const metric = document.getElementById('tool-metric')?.value || 'combined_score';

    if (mode === 'score_vs_length') {
        plotScoreVsLength();
    } else if (mode === 'histogram') {
        plotHistogram(metric);
    } else {
        plotBoxPlot(metric);
    }
}

function plotBoxPlot(metric) {
    const tools = [...new Set(filteredData.map(d => d.tool))].sort();
    const traces = tools.map(tool => {
        const vals = filteredData.filter(d => d.tool === tool && d[metric] != null).map(d => d[metric]);
        return {
            y: vals, type: 'box', name: `${tool} (n=${vals.length})`,
            marker: { color: TOOL_COLORS[tool] || '#888' },
            line: { color: TOOL_COLORS[tool] || '#888' }, fillcolor: (TOOL_COLORS[tool] || '#888') + '99',
        };
    });
    Plotly.newPlot('tool-plot', traces, {
        title: { text: `${metric} by tool`, font: { color: '#e2e8f0' } },
        yaxis: { title: metric, color: '#94a3b8', gridcolor: '#1e293b' },
        xaxis: { color: '#94a3b8' },
        plot_bgcolor: '#0f172a', paper_bgcolor: '#0f172a',
        font: { color: '#94a3b8' }, margin: { t: 40 },
    }, { responsive: true });
}

function plotHistogram(metric) {
    const tools = [...new Set(filteredData.map(d => d.tool))].sort();
    const traces = tools.map(tool => {
        const vals = filteredData.filter(d => d.tool === tool && d[metric] != null).map(d => d[metric]);
        return {
            x: vals, type: 'histogram', name: `${tool} (n=${vals.length})`,
            marker: { color: TOOL_COLORS[tool] || '#888' }, opacity: 0.5, nbinsx: 30,
        };
    });
    Plotly.newPlot('tool-plot', traces, {
        barmode: 'overlay',
        title: { text: `${metric} distribution by tool`, font: { color: '#e2e8f0' } },
        xaxis: { title: metric, color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'Count', color: '#94a3b8', gridcolor: '#1e293b' },
        plot_bgcolor: '#0f172a', paper_bgcolor: '#0f172a',
        font: { color: '#94a3b8' }, margin: { t: 40 },
    }, { responsive: true });
}

function plotScoreVsLength() {
    const tools = [...new Set(filteredData.map(d => d.tool))].sort();
    const traces = tools.map(tool => {
        const pts = filteredData.filter(d => d.tool === tool && d.binder_length != null && d.combined_score != null);
        return {
            x: pts.map(d => d.binder_length), y: pts.map(d => d.combined_score),
            text: pts.map(d => d.design_id), mode: 'markers', type: 'scatter',
            name: `${tool} (${pts.length})`,
            marker: { color: TOOL_COLORS[tool] || '#888', size: 6, opacity: 0.5 },
        };
    });
    Plotly.newPlot('tool-plot', traces, {
        title: { text: 'Score vs Length by Tool', font: { color: '#e2e8f0' } },
        xaxis: { title: 'Binder Length (aa)', color: '#94a3b8', gridcolor: '#1e293b' },
        yaxis: { title: 'Combined Score', color: '#94a3b8', gridcolor: '#1e293b' },
        plot_bgcolor: '#0f172a', paper_bgcolor: '#0f172a',
        font: { color: '#94a3b8' }, margin: { t: 40 },
    }, { responsive: true });
}

function initToolControls() {
    const metricSel = document.getElementById('tool-metric');
    if (metricSel) {
        const available = TOOL_METRICS.filter(m => allData.some(d => d[m] != null));
        available.forEach(m => metricSel.add(new Option(m, m)));
    }
    document.getElementById('tool-mode')?.addEventListener('change', updateToolComparison);
    document.getElementById('tool-metric')?.addEventListener('change', updateToolComparison);
}

// ============================================================
// Section 7: Design Detail + Structure Viewer (3Dmol.js)
// ============================================================
let viewer3d = null;
let cachedSiteResidues = null;

function selectDesign(d) {
    if (!d) return;
    selectedDesign = d;
    updateDetail();
    // Sync radar if visible
    if (activeTab === 'radar') updateRadar();
    dirtyTabs.add('radar');
}

function updateDetail() {
    const d = selectedDesign;
    if (!d) return;
    const cards = document.getElementById('score-cards');
    const fields = [
        ['Combined Score', d.combined_score], ['Boltz iPTM', d.boltz_iptm],
        ['Boltz pLDDT', d.boltz_binder_plddt], ['ESMFold pLDDT', d.esmfold_plddt],
        ['pDockQ', d.pDockQ], ['Tier', d.tier],
        ['Rosetta dG', d.rosetta_dG], ['Rosetta Sc', d.rosetta_sc],
        ['Site PAE', d.boltz_site_mean_pae], ['SIF', d.site_interface_fraction],
        ['Refolding RMSD', d.refolding_rmsd], ['Solubility', d.netsolp_solubility],
        ['SAP', d.rosetta_sap], ['Interface KE', d.interface_KE_fraction],
        ['Binder Length', d.binder_length], ['Tool', d.tool],
    ];
    cards.innerHTML = fields.map(([label, val]) => `
        <div class="bg-slate-900 border border-slate-700 rounded p-2">
            <p class="text-xs text-slate-500">${label}</p>
            <p class="text-lg font-mono ${val == null ? 'text-slate-600' : 'text-slate-100'}">${
                val != null ? (typeof val === 'number' ? val.toFixed(3) : val) : 'N/A'
            }</p>
        </div>
    `).join('');

    document.getElementById('detail-seq').textContent = d.binder_sequence || 'N/A';

    // SS bars
    const ssDiv = document.getElementById('ss-bars');
    if (ssDiv) {
        const h = d.binder_helix_frac != null ? (d.binder_helix_frac * 100).toFixed(0) : '?';
        const s = d.binder_sheet_frac != null ? (d.binder_sheet_frac * 100).toFixed(0) : '?';
        const l = d.binder_loop_frac != null ? (d.binder_loop_frac * 100).toFixed(0) : '?';
        ssDiv.innerHTML = `
            <div class="flex items-center gap-2"><span class="text-xs text-slate-500 w-12">Helix</span>
                <div class="flex-1 bg-slate-800 rounded h-3"><div class="bg-red-500 h-3 rounded" style="width:${h}%"></div></div>
                <span class="text-xs text-slate-400 w-8">${h}%</span></div>
            <div class="flex items-center gap-2"><span class="text-xs text-slate-500 w-12">Sheet</span>
                <div class="flex-1 bg-slate-800 rounded h-3"><div class="bg-blue-500 h-3 rounded" style="width:${s}%"></div></div>
                <span class="text-xs text-slate-400 w-8">${s}%</span></div>
            <div class="flex items-center gap-2"><span class="text-xs text-slate-500 w-12">Loop</span>
                <div class="flex-1 bg-slate-800 rounded h-3"><div class="bg-gray-500 h-3 rounded" style="width:${l}%"></div></div>
                <span class="text-xs text-slate-400 w-8">${l}%</span></div>`;
    }

    // Update detail selector
    const detSel = document.getElementById('detail-select');
    if (detSel) {
        for (let i = 0; i < detSel.options.length; i++) {
            if (allData[detSel.options[i].value]?.design_id === d.design_id) {
                detSel.selectedIndex = i; break;
            }
        }
    }

    // Sync radar selector
    const radarSel = document.getElementById('radar-design');
    if (radarSel) radarSel.value = d.design_id || '';

    loadStructure(d);
    loadPlipForDesign(d);
}

async function loadStructure(d) {
    const container = document.getElementById('mol-viewer');
    if (!container || typeof $3Dmol === 'undefined') return;

    // Destroy and recreate viewer to prevent ghost structures
    container.innerHTML = '';
    viewer3d = $3Dmol.createViewer(container, { backgroundColor: '#0f172a' });

    try {
        const structs = await (await fetch('/api/results/' + JOB_ID + '/structures')).json();
        const match = structs.find(s => s.filename.includes(d.design_id));
        if (!match) { viewer3d.render(); return; }

        const resp = await fetch('/api/results/' + JOB_ID + '/structure/' + match.path);
        const data = await resp.text();
        const fmt = match.filename.endsWith('.cif') ? 'cif' : 'pdb';

        viewer3d.addModel(data, fmt);
        // Detect binder vs target by chain length (binder is shorter)
        const toolColor = TOOL_COLORS[d.tool] || '#3b82f6';
        const chains = {};
        viewer3d.selectedAtoms({}).forEach(a => {
            chains[a.chain] = (chains[a.chain] || 0) + 1;
        });
        const chainIds = Object.keys(chains);
        let binderChain, targetChain;
        if (chainIds.length >= 2) {
            // Shorter chain = binder
            chainIds.sort((a, b) => chains[a] - chains[b]);
            binderChain = chainIds[0];
            targetChain = chainIds[1];
        } else {
            binderChain = 'A';
            targetChain = 'B';
        }
        viewer3d.setStyle({ chain: targetChain }, { cartoon: { color: '#64748b' } });
        viewer3d.setStyle({ chain: binderChain }, { cartoon: { color: toolColor } });

        // Site residues as red sticks (Boltz-renumbered) on target chain
        const siteResidues = await fetchSiteResidues();
        if (siteResidues.length > 0) {
            // Apply on target chain (detected above), fall back to trying both
            const tryCh = [targetChain, binderChain];
            let applied = 0;
            for (const ch of tryCh) {
                for (const resi of siteResidues) {
                    const atoms = viewer3d.selectedAtoms({ chain: ch, resi: resi });
                    if (atoms.length > 0) {
                        viewer3d.addStyle(
                            { chain: ch, resi: resi },
                            { stick: { color: 'red', radius: 0.25 }, sphere: { color: 'red', radius: 0.4 } }
                        );
                        applied++;
                    }
                }
                if (applied > 0) break; // found the right chain, stop
            }
            console.log(`Site: ${applied}/${siteResidues.length} residues highlighted on chain ${applied > 0 ? tryCh[0] : '?'}`);
        }

        viewer3d.zoomTo();
        viewer3d.render();
    } catch (e) {
        console.error('Structure load error:', e);
    }
}

async function fetchSiteResidues() {
    if (cachedSiteResidues !== null) return cachedSiteResidues;
    try {
        const resp = await fetch('/api/results/' + JOB_ID + '/site_residues');
        const data = await resp.json();
        cachedSiteResidues = data.residues || [];
    } catch (e) {
        console.warn('Failed to fetch site residues:', e);
        cachedSiteResidues = [];
    }
    return cachedSiteResidues;
}

function initDetailControls() {
    const detSel = document.getElementById('detail-select');
    if (!detSel) return;
    allData.forEach((d, i) => {
        detSel.add(new Option(`#${d.rank || i + 1} ${d.design_id} (${d.tool})`, i));
    });
    detSel.addEventListener('change', () => selectDesign(allData[detSel.value]));
}

// ============================================================
// Section 8: PLIP
// ============================================================
async function loadPlipForDesign(d) {
    const plipDiv = document.getElementById('plip-content');
    if (!plipDiv) return;
    try {
        const resp = await fetch('/api/results/' + JOB_ID + '/plip');
        const data = await resp.json();
        if (data.summary) {
            // Check if this design is mentioned in the summary
            const lines = data.summary.split('\n');
            const relevant = lines.filter(l =>
                l.includes(d.design_id) || l.startsWith('===') || l.startsWith('Design') || l.startsWith('PLIP')
            );
            if (relevant.length > 0) {
                plipDiv.innerHTML = `<pre class="text-xs text-green-300 whitespace-pre-wrap">${relevant.join('\n')}</pre>`;
            } else {
                plipDiv.innerHTML = '<p class="text-xs text-slate-500">No PLIP data for this design</p>';
            }
        } else {
            const hasPLIP = data.designs && data.designs.some(name => name.includes(d.design_id));
            plipDiv.innerHTML = hasPLIP
                ? '<p class="text-xs text-green-400">PLIP analysis available</p>'
                : '<p class="text-xs text-slate-500">No PLIP data for this design</p>';
        }
    } catch {
        plipDiv.innerHTML = '<p class="text-xs text-slate-500">PLIP not available</p>';
    }
}

// ============================================================
// Section 9: Rerank Form
// ============================================================
function initRerankForm() {
    // Drag-drop file upload for rerank tab
    const rrDropZone = document.getElementById('rr-drop-zone');
    const rrFileInput = document.getElementById('rr-file-input');
    const rrDropText = document.getElementById('rr-drop-text');
    if (rrDropZone) {
        rrDropZone.addEventListener('click', () => rrFileInput.click());
        rrDropZone.addEventListener('dragover', e => { e.preventDefault(); rrDropZone.classList.add('border-blue-500'); });
        rrDropZone.addEventListener('dragleave', () => rrDropZone.classList.remove('border-blue-500'));
        rrDropZone.addEventListener('drop', e => {
            e.preventDefault();
            rrDropZone.classList.remove('border-blue-500');
            if (e.dataTransfer.files.length) {
                rrFileInput.files = e.dataTransfer.files;
                rrDropText.textContent = e.dataTransfer.files[0].name;
            }
        });
        rrFileInput.addEventListener('change', () => {
            if (rrFileInput.files.length) rrDropText.textContent = rrFileInput.files[0].name;
        });
    }

    document.getElementById('rr_reprediction')?.addEventListener('change', async (e) => {
        const gpuDiv = document.getElementById('rr-gpu-select');
        const plddtDiv = document.getElementById('rr-plddt-threshold');
        if (e.target.checked) {
            gpuDiv.classList.remove('hidden');
            plddtDiv?.classList.remove('hidden');
            const sel = document.getElementById('rr_gpu_id');
            sel.innerHTML = '';
            const gpus = await (await fetch('/api/gpus')).json();
            gpus.forEach(g => {
                const opt = new Option(
                    `GPU ${g.index}: ${g.name} (${g.memory_used_mb}/${g.memory_total_mb} MB)${g.busy ? ' — BUSY' : ''}`,
                    g.index);
                if (g.busy) opt.disabled = true;
                sel.add(opt);
            });
        } else { gpuDiv.classList.add('hidden'); plddtDiv?.classList.add('hidden'); }
    });

    document.getElementById('rerank-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData();
        const suffix = form.rr_suffix.value || 'reranked';
        const resultsDir = form.results_dir.value;
        const outDir = resultsDir + '/reranked_' + suffix;

        formData.append('job_name', form.job_name.value);
        formData.append('job_type', 'rerank');
        formData.append('project', form.project?.value || 'unassigned');
        // Upload file takes priority over path
        const rrFile = document.getElementById('rr-file-input');
        if (rrFile && rrFile.files.length) {
            formData.append('target', rrFile.files[0]);
        }
        formData.append('target_path', form.target_path.value);
        formData.append('site', form.site.value);
        formData.append('length', '60-80');
        formData.append('tools', 'rfdiffusion,boltzgen,bindcraft,pxdesign,proteina,proteina_complexa');
        formData.append('mode', 'test');
        formData.append('score_weights', form.score_weights.value);
        formData.append('ss_bias', form.ss_bias.value);
        formData.append('plip_top', form.plip_top.value || '10');
        formData.append('top_n', '50');
        formData.append('results_dir', resultsDir);
        formData.append('out_dir_override', outDir);

        const repred = form.reprediction?.checked || false;
        formData.append('reprediction', repred);
        formData.append('gpu_id', repred ? form.gpu_id.value : '0');
        if (repred && form.esmfold_plddt_threshold?.value) {
            formData.append('esmfold_plddt_threshold', form.esmfold_plddt_threshold.value);
        }

        if (form.no_cys.checked) formData.append('no_cys', 'true');
        ['max_aa_fraction', 'min_sc', 'max_refolding_rmsd', 'max_interface_ke',
         'min_site_interface_fraction', 'max_site_dist', 'min_site_fraction'].forEach(name => {
            const val = form[name]?.value;
            if (val) formData.append(name, val);
        });

        const statusDiv = document.getElementById('rr-status');
        statusDiv.classList.remove('hidden');
        statusDiv.innerHTML = '<span class="text-blue-400">Launching rerank...</span>';

        try {
            const resp = await fetch('/api/jobs', { method: 'POST', body: formData });
            const data = await resp.json();
            if (resp.ok) {
                window.location.href = '/monitor/' + data.job_id;
            } else {
                statusDiv.innerHTML = `<span class="text-red-400">Error: ${data.detail || JSON.stringify(data)}</span>`;
                statusDiv.scrollIntoView({behavior: 'smooth'});
            }
        } catch (err) {
            statusDiv.innerHTML = `<span class="text-red-400">Network error: ${err.message}</span>`;
            statusDiv.scrollIntoView({behavior: 'smooth'});
        }
    });

    document.getElementById('rr-dryrun-btn')?.addEventListener('click', () => {
        const form = document.getElementById('rerank-form');
        const parts = ['python rerank_binders.py'];
        parts.push('--target ' + form.target_path.value);
        parts.push('--site "' + form.site.value + '"');
        parts.push('--results_dir ' + form.results_dir.value);
        parts.push('--out_dir ' + form.results_dir.value + '/reranked_' + form.rr_suffix.value);
        parts.push('--rank_only');
        parts.push('--score_weights ' + form.score_weights.value);
        parts.push('--ss_bias ' + form.ss_bias.value);
        if (form.no_cys.checked) parts.push('--no_cys');
        ['max_aa_fraction', 'min_sc', 'max_refolding_rmsd', 'max_interface_ke',
         'min_site_interface_fraction', 'max_site_dist', 'min_site_fraction'].forEach(f => {
            if (form[f]?.value) parts.push('--' + f + ' ' + form[f].value);
        });
        if (form.plip_top.value) parts.push('--plip_top ' + form.plip_top.value);
        if (form.reprediction?.checked) {
            parts.push('--reprediction');
            const plddt = form.esmfold_plddt_threshold?.value;
            if (plddt && plddt !== '80') parts.push('--esmfold_plddt_threshold ' + plddt);
        }
        const out = document.getElementById('rr-dryrun-output');
        out.textContent = parts.join(' \\\n  ');
        out.classList.remove('hidden');
    });
}

// ============================================================
// Section 10: Tab Switching + Init
// ============================================================
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Notify Live Rerank if leaving that tab
            if (activeTab === 'livererank' && window.LiveRerank) {
                window.LiveRerank.onTabHide();
            }

            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active', 'border-blue-500', 'text-blue-400',
                    'border-emerald-500', 'text-emerald-400',
                    'border-amber-500', 'text-amber-400');
                b.classList.add('border-transparent', 'text-slate-400');
            });
            const tab = btn.dataset.tab;
            // Use per-tab accent colors
            if (tab === 'livererank') {
                btn.classList.add('active', 'border-emerald-500', 'text-emerald-400');
            } else if (tab === 'rerank') {
                btn.classList.add('active', 'border-amber-500', 'text-amber-400');
            } else {
                btn.classList.add('active', 'border-blue-500', 'text-blue-400');
            }
            btn.classList.remove('border-transparent', 'text-slate-400');
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
            document.getElementById('tab-' + tab).classList.remove('hidden');
            activeTab = tab;
            // Render dirty tab
            if (dirtyTabs.has(tab)) {
                dirtyTabs.delete(tab);
                updateActiveTab();
            } else if (tab === 'scatter' && filteredData.length) {
                updateScatter();
            }
            // Initialize Live Rerank lazily on first view
            if (tab === 'livererank' && window.LiveRerank) {
                window.LiveRerank.init();
            }
        });
    });
}

// Filter sidebar toggle
function initFilterToggle() {
    const btn = document.getElementById('filter-toggle');
    const panel = document.getElementById('filter-sidebar');
    if (btn && panel) {
        btn.addEventListener('click', () => {
            panel.classList.toggle('hidden');
            btn.textContent = panel.classList.contains('hidden') ? 'Filters' : 'Hide Filters';
        });
    }
}

async function loadData() {
    const resp = await fetch('/api/results/' + JOB_ID + '/rankings?include_unranked=true');
    allData = await resp.json();
    if (!allData.length) return;

    filteredData = allData.slice();

    initGrid();
    initScatterControls();
    initRadarScaling();
    initRadarControls();
    initToolControls();
    initDetailControls();
    initFilterPanel();
    initFilterToggle();
    updateFilterStatus();
}

// Boot
initTabs();
initRerankForm();
loadData();
