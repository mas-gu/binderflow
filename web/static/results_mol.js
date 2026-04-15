/**
 * ProteaFlow Molecule Results — JS for rankings, scatter, tools, detail.
 *
 * Expects these globals injected by Jinja2:
 *   JOB_ID, TOOL_COLORS, SCATTER_PRESETS, HIGHER_BETTER, LOWER_BETTER,
 *   DEFAULT_COLUMNS, TOOL_METRICS, IS_MOLECULE
 */

// ============================================================
// Section 1: Data Store
// ============================================================
let allData = [];
let filteredData = [];
let filterState = {
    min_combined: -10, min_qed: 0, max_vina: 0,
    max_sa: 10, max_lipinski: 5,
    active_tools: [],
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
        // Score thresholds (pass NaN through)
        if (row.combined_score != null && row.combined_score < filterState.min_combined) return false;
        if (row.qed != null && row.qed < filterState.min_qed) return false;
        if (row.vina_score != null && row.vina_score > filterState.max_vina) return false;
        if (row.sa_score != null && row.sa_score > filterState.max_sa) return false;
        if (row.lipinski_violations != null && row.lipinski_violations > filterState.max_lipinski) return false;

        // Tool filter
        if (filterState.active_tools.length > 0 && !filterState.active_tools.includes(row.tool)) return false;

        return true;
    });

    updateActiveTab();
    ['rankings', 'scatter', 'tools'].forEach(t => {
        if (t !== activeTab) dirtyTabs.add(t);
    });
    updateFilterStatus();
}

function updateActiveTab() {
    switch (activeTab) {
        case 'rankings': updateGrid(); break;
        case 'scatter': updateScatter(); break;
        case 'tools': updateToolComparison(); break;
    }
}

function updateFilterStatus() {
    const el = document.getElementById('filter-status');
    if (el) el.textContent = `Showing ${filteredData.length} / ${allData.length} molecules`;
}

function initFilterPanel() {
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

    document.querySelectorAll('.filter-input').forEach(el => {
        el.addEventListener('change', readFiltersAndApply);
    });
    document.querySelectorAll('.tool-cb').forEach(el => {
        el.addEventListener('change', readFiltersAndApply);
    });

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
    filterState.min_qed = parseFloat(document.getElementById('f-min-qed')?.value) || 0;
    filterState.max_vina = parseFloat(document.getElementById('f-max-vina')?.value) || 0;
    filterState.max_sa = parseFloat(document.getElementById('f-max-sa')?.value) || 10;
    filterState.max_lipinski = parseInt(document.getElementById('f-max-lipinski')?.value) || 5;
    filterState.active_tools = [...document.querySelectorAll('.tool-cb:checked')].map(cb => cb.dataset.tool);
    applyFilters();
}

function resetFilters() {
    document.getElementById('f-min-combined').value = -10;
    document.getElementById('f-min-qed').value = 0;
    document.getElementById('f-max-vina').value = 0;
    document.getElementById('f-max-sa').value = 10;
    document.getElementById('f-max-lipinski').value = 5;
    document.querySelectorAll('.tool-cb').forEach(cb => cb.checked = true);
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
        .filter(c => DEFAULT_COLUMNS.includes(c) || c === 'smiles')
        .map(c => ({
            field: c, headerName: c, sortable: true, filter: true, resizable: true,
            width: c === 'smiles' ? 250 : c === 'design_id' ? 180 : 110,
            hide: c === 'smiles' && !DEFAULT_COLUMNS.includes('smiles'),
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
    // Default axes for molecules
    xSel.value = 'qed'; ySel.value = 'vina_score';

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
// Section 5: Tool Comparison
// ============================================================
function updateToolComparison() {
    const mode = document.getElementById('tool-mode')?.value || 'box';
    const metric = document.getElementById('tool-metric')?.value || 'combined_score';

    if (mode === 'histogram') {
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
// Section 6: Molecule Detail
// ============================================================
function selectDesign(d) {
    if (!d) return;
    selectedDesign = d;
    updateDetail();
}

function updateDetail() {
    const d = selectedDesign;
    if (!d) return;
    const cards = document.getElementById('score-cards');
    const fields = [
        ['Combined Score', d.combined_score], ['QED', d.qed],
        ['SA Score', d.sa_score], ['Vina Score', d.vina_score],
        ['MW', d.mw], ['LogP', d.logp],
        ['HBD', d.hbd], ['HBA', d.hba],
        ['TPSA', d.tpsa], ['Lipinski Violations', d.lipinski_violations],
        ['Rotatable Bonds', d.rotatable_bonds], ['Tool', d.tool],
    ];
    cards.innerHTML = fields.map(([label, val]) => `
        <div class="bg-slate-900 border border-slate-700 rounded p-2">
            <p class="text-xs text-slate-500">${label}</p>
            <p class="text-lg font-mono ${val == null ? 'text-slate-600' : 'text-slate-100'}">${
                val != null ? (typeof val === 'number' ? val.toFixed(3) : val) : 'N/A'
            }</p>
        </div>
    `).join('');

    document.getElementById('detail-smiles').textContent = d.smiles || 'N/A';

    // Properties table
    const propsDiv = document.getElementById('detail-props');
    if (propsDiv) {
        const props = Object.keys(d).filter(k =>
            !['rank', 'design_id', 'tool', 'smiles'].includes(k) && d[k] != null
        );
        propsDiv.innerHTML = props.map(k => `
            <div class="flex justify-between">
                <span class="text-slate-400">${k}</span>
                <span class="text-slate-200 font-mono">${typeof d[k] === 'number' ? d[k].toFixed(3) : d[k]}</span>
            </div>
        `).join('');
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
// Section 7: Tab Switching + Init
// ============================================================
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active', 'border-blue-500', 'text-blue-400');
                b.classList.add('border-transparent', 'text-slate-400');
            });
            btn.classList.add('active', 'border-blue-500', 'text-blue-400');
            btn.classList.remove('border-transparent', 'text-slate-400');
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
            const tab = btn.dataset.tab;
            document.getElementById('tab-' + tab).classList.remove('hidden');
            activeTab = tab;
            if (dirtyTabs.has(tab)) {
                dirtyTabs.delete(tab);
                updateActiveTab();
            } else if (tab === 'scatter' && filteredData.length) {
                updateScatter();
            }
        });
    });
}

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
    initToolControls();
    initDetailControls();
    initFilterPanel();
    initFilterToggle();
    updateFilterStatus();
}

// Boot
initTabs();
loadData();
