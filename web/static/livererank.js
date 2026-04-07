/**
 * BinderFlow Live Rerank — Three.js 3D visualization + real-time filtering.
 *
 * Expects globals injected by Jinja2:
 *   JOB_ID, TOOL_COLORS, JOB_SITE
 *
 * Three.js 0.164.0 (UMD) + OrbitControls loaded via <script> tags in template.
 * Exposes window.LiveRerank.init() for lazy initialization from the tab switcher.
 */
(function () {
    'use strict';

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    let initialized = false;
    let animating = false;
    let designs = [];
    let geometryData = null;

    // Three.js handles
    let renderer, scene, camera, controls;
    let targetMesh, targetWire, siteSpheres = [], siteCentroidSphere;
    let instancedMeshes = {};   // tool -> THREE.InstancedMesh
    let designIndexMap = {};    // tool -> [designIndex, ...]  (parallel to instance ids)
    let raycaster, mouse;
    let selectedDesignIdx = null;
    let selectedMarker = null;
    let approachArrow = null;

    // Camera state for view presets
    let sceneCenter = new THREE.Vector3();  // target protein center
    let siteCentroid3 = null;               // site centroid as Vector3
    let boundingRadius = 20;

    // Filter state
    let filters = {
        min_iptm: 0, min_plddt: 0, max_refolding_rmsd: 10,
        max_interface_pae: 30, max_site_pae: 30, min_sc: 0,
        max_interface_ke: 1, max_aa_fraction: 1,
        min_sif: 0, max_centroid_dist: 40, min_cos_angle: -1,
        min_contact_fraction: 0,
        no_cys: false, ss_bias: 'any',
        w_plddt: 0.3, w_iptm: 0.6, w_dg: 0.1,
        active_tools: new Set()
    };

    // Per-design derived data (computed once after load)
    let designPass = [];   // bool[]
    let designScores = []; // number|null[]

    // Tooltip / detail elements (created once)
    let tooltipEl, statsEl, histCanvas, histCtx, detailEl;

    // Debounce handle for filter updates
    let filterRAF = null;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
    function fmt(v, d) { return v == null ? '--' : v.toFixed(d); }
    function darkenColor(hex, factor) {
        const c = new THREE.Color(hex);
        c.multiplyScalar(factor);
        return c;
    }

    // Brighten tool colors for white background — boost saturation and lightness
    function brightenColor(hex) {
        const c = new THREE.Color(hex);
        const hsl = {};
        c.getHSL(hsl);
        // Increase saturation to 1.0, push lightness to 0.5-0.6 range
        c.setHSL(hsl.h, Math.min(1.0, hsl.s * 1.3 + 0.2), clamp(hsl.l * 0.85 + 0.15, 0.45, 0.6));
        return '#' + c.getHexString();
    }

    // Build bright palette once on init
    let BRIGHT_COLORS = {};

    // ------------------------------------------------------------------
    // 1. Data loading
    // ------------------------------------------------------------------
    async function fetchGeometry() {
        const spinner = document.getElementById('livererank-spinner');
        if (spinner) {
            spinner.classList.remove('hidden');
            spinner.innerHTML = `
                <div class="text-center">
                    <div class="text-slate-400 text-sm mb-2" id="lr-progress-text">Connecting...</div>
                    <div class="w-64 bg-slate-800 rounded-full h-2.5 mx-auto">
                        <div id="lr-progress-bar" class="bg-emerald-500 h-2.5 rounded-full transition-all duration-150" style="width: 0%"></div>
                    </div>
                    <div class="text-slate-500 text-xs mt-1" id="lr-progress-detail"></div>
                </div>`;
        }

        const progressText = document.getElementById('lr-progress-text');
        const progressBar = document.getElementById('lr-progress-bar');
        const progressDetail = document.getElementById('lr-progress-detail');

        return new Promise((resolve) => {
            try {
                const evtSource = new EventSource(`/api/results/${JOB_ID}/geometry/stream`);

                evtSource.onmessage = function (event) {
                    try {
                        const msg = JSON.parse(event.data);

                        if (msg.type === 'progress') {
                            const pct = msg.total > 0 ? Math.round(msg.done / msg.total * 100) : 0;
                            if (progressBar) progressBar.style.width = pct + '%';
                            if (progressText) progressText.textContent = `Processing structures: ${msg.done} / ${msg.total}`;
                            if (progressDetail) progressDetail.textContent = `${msg.found} structures found`;
                        } else if (msg.type === 'done') {
                            evtSource.close();
                            // Data is cached on server — fetch it via normal GET
                            if (progressText) progressText.textContent = 'Loading geometry data...';
                            if (progressBar) progressBar.style.width = '100%';
                            fetch(`/api/results/${JOB_ID}/geometry`)
                                .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
                                .then(data => {
                                    geometryData = data;
                                    designs = geometryData.designs || [];
                                    if (spinner) spinner.classList.add('hidden');
                                    resolve(true);
                                })
                                .catch(e => {
                                    console.error('LiveRerank: fetch after SSE done failed', e);
                                    const vp = document.getElementById('livererank-viewport');
                                    if (vp) vp.innerHTML = `<p class="text-red-400 p-4">Failed to load geometry: ${e.message}</p>`;
                                    resolve(false);
                                });
                        } else if (msg.type === 'error') {
                            evtSource.close();
                            console.error('LiveRerank: geometry error', msg.message);
                            const vp = document.getElementById('livererank-viewport');
                            if (vp) vp.innerHTML = `<p class="text-red-400 p-4">Geometry computation failed: ${msg.message}</p>`;
                            resolve(false);
                        }
                    } catch (parseErr) {
                        console.error('LiveRerank: failed to parse SSE message', parseErr);
                    }
                };

                evtSource.onerror = function () {
                    evtSource.close();
                    // SSE error — fall back to direct GET
                    console.warn('LiveRerank: SSE failed, falling back to direct fetch');
                    if (progressText) progressText.textContent = 'Loading (fallback)...';
                    fetch(`/api/results/${JOB_ID}/geometry`)
                        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
                        .then(data => {
                            geometryData = data;
                            designs = geometryData.designs || [];
                            if (spinner) spinner.classList.add('hidden');
                            resolve(true);
                        })
                        .catch(e => {
                            console.error('LiveRerank: fallback fetch failed', e);
                            const vp = document.getElementById('livererank-viewport');
                            if (vp) vp.innerHTML = `<p class="text-red-400 p-4">Failed to load geometry data: ${e.message}</p>`;
                            resolve(false);
                        });
                };
            } catch (e) {
                console.error('LiveRerank: failed to start SSE', e);
                if (spinner) spinner.classList.add('hidden');
                resolve(false);
            }
        });
    }

    // ------------------------------------------------------------------
    // 2. Three.js Scene Setup
    // ------------------------------------------------------------------
    function initScene() {
        const container = document.getElementById('livererank-viewport');
        if (!container) return;

        const w = container.clientWidth || 800;
        const h = container.clientHeight || 600;

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0xffffff);
        container.innerHTML = '';
        container.appendChild(renderer.domElement);

        // Scene
        scene = new THREE.Scene();

        // Camera
        camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 2000);

        // Lighting
        scene.add(new THREE.AmbientLight(0x808090, 0.7));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
        dirLight.position.set(50, 80, 60);
        scene.add(dirLight);

        // Controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.12;

        // Raycaster
        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();

        // Events
        renderer.domElement.addEventListener('mousemove', onMouseMove);
        renderer.domElement.addEventListener('click', onMouseClick);
        window.addEventListener('resize', onResize);
    }

    function buildScene() {
        if (!geometryData) return;
        const tgt = geometryData.target || {};
        const site = geometryData.site || {};

        // --- Target convex hull ---
        if (tgt.hull_vertices && tgt.hull_faces && tgt.hull_vertices.length && tgt.hull_faces.length) {
            const geo = new THREE.BufferGeometry();
            const verts = new Float32Array(tgt.hull_vertices.flat());
            geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
            const indices = [];
            for (const f of tgt.hull_faces) indices.push(f[0], f[1], f[2]);
            geo.setIndex(indices);
            geo.computeVertexNormals();

            targetMesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
                color: 0xb0bec5, transparent: false,
                side: THREE.DoubleSide, depthWrite: true,
                shininess: 30, specular: 0x222222
            }));
            scene.add(targetMesh);

            targetWire = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({
                color: 0x78909c, wireframe: true, opacity: 0.3, transparent: true
            }));
            scene.add(targetWire);
        }

        // --- Site residues ---
        const sitePositions = site.residue_positions || [];
        const sphereGeo = new THREE.SphereGeometry(0.8, 12, 12);
        const siteMat = new THREE.MeshPhongMaterial({ color: 0xff1744 });
        for (const pos of sitePositions) {
            const mesh = new THREE.Mesh(sphereGeo, siteMat);
            mesh.position.set(pos[0], pos[1], pos[2]);
            scene.add(mesh);
            siteSpheres.push(mesh);
        }
        if (site.centroid) {
            const bigGeo = new THREE.SphereGeometry(1.5, 16, 16);
            const bigMat = new THREE.MeshPhongMaterial({
                color: 0xff1744, transparent: true, opacity: 0.45
            });
            siteCentroidSphere = new THREE.Mesh(bigGeo, bigMat);
            siteCentroidSphere.position.set(site.centroid[0], site.centroid[1], site.centroid[2]);
            scene.add(siteCentroidSphere);
        }

        // --- Binder ellipsoids (InstancedMesh per tool) ---
        buildInstancedMeshes();

        // --- Camera framing ---
        const center = tgt.center || [0, 0, 0];
        sceneCenter = new THREE.Vector3(center[0], center[1], center[2]);
        if (site.centroid) {
            siteCentroid3 = new THREE.Vector3(site.centroid[0], site.centroid[1], site.centroid[2]);
        }
        controls.target.copy(sceneCenter);

        // Compute bounding radius
        boundingRadius = 20;
        if (tgt.hull_vertices) {
            for (const v of tgt.hull_vertices) {
                const d = new THREE.Vector3(v[0], v[1], v[2]).distanceTo(sceneCenter);
                if (d > boundingRadius) boundingRadius = d;
            }
        }
        setCameraView('initial');
        controls.update();
    }

    function buildInstancedMeshes() {
        // Group designs by tool
        const byTool = {};
        designs.forEach((d, i) => {
            const t = d.tool || 'unknown';
            if (!byTool[t]) byTool[t] = [];
            byTool[t].push(i);
        });

        const baseGeo = new THREE.SphereGeometry(1, 16, 12);

        for (const [tool, indices] of Object.entries(byTool)) {
            const color = BRIGHT_COLORS[tool] || TOOL_COLORS[tool] || '#888888';
            const mat = new THREE.MeshPhongMaterial({
                color: new THREE.Color(color),
                transparent: true,
                opacity: 0.75,
                depthWrite: false
            });
            // InstancedMesh with per-instance color
            const im = new THREE.InstancedMesh(baseGeo, mat, indices.length);
            im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            // Enable per-instance color
            const colors = new Float32Array(indices.length * 3);
            const bright = new THREE.Color(color);
            for (let j = 0; j < indices.length; j++) {
                colors[j * 3] = bright.r;
                colors[j * 3 + 1] = bright.g;
                colors[j * 3 + 2] = bright.b;
            }
            im.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);

            // Set transforms
            const dummy = new THREE.Object3D();
            for (let j = 0; j < indices.length; j++) {
                const d = designs[indices[j]];
                const c = d.centroid || [0, 0, 0];
                const axes = d.ellipsoid_axes || [2, 2, 2];
                const quat = d.ellipsoid_rotation || [1, 0, 0, 0]; // [w, x, y, z]

                dummy.position.set(c[0], c[1], c[2]);
                dummy.quaternion.set(quat[1], quat[2], quat[3], quat[0]); // THREE uses (x,y,z,w)
                dummy.scale.set(axes[0], axes[1], axes[2]);
                dummy.updateMatrix();
                im.setMatrixAt(j, dummy.matrix);
            }
            im.instanceMatrix.needsUpdate = true;
            im.userData = { tool: tool };

            scene.add(im);
            instancedMeshes[tool] = im;
            designIndexMap[tool] = indices;
        }
    }

    // ------------------------------------------------------------------
    // 3. Filter Logic
    // ------------------------------------------------------------------
    function passesFilters(design) {
        const m = design.metrics || {};

        // Tool toggle
        if (!filters.active_tools.has(design.tool)) return false;

        // Quality filters — NaN pass-through
        if (m.boltz_iptm != null && m.boltz_iptm < filters.min_iptm) return false;
        if (m.esmfold_plddt != null && m.esmfold_plddt < filters.min_plddt) return false;
        if (m.refolding_rmsd != null && m.refolding_rmsd > filters.max_refolding_rmsd) return false;
        if (m.boltz_interface_pae != null && m.boltz_interface_pae > filters.max_interface_pae) return false;
        if (m.boltz_site_mean_pae != null && m.boltz_site_mean_pae > filters.max_site_pae) return false;
        if (m.rosetta_sc != null && m.rosetta_sc < filters.min_sc) return false;
        if (m.interface_ke != null && m.interface_ke > filters.max_interface_ke) return false;
        if (m.max_aa_fraction != null && m.max_aa_fraction > filters.max_aa_fraction) return false;

        // Geometric filters
        if (m.site_interface_fraction != null && m.site_interface_fraction < filters.min_sif) return false;
        if (m.site_centroid_dist != null && m.site_centroid_dist > filters.max_centroid_dist) return false;
        if (m.site_cos_angle != null && m.site_cos_angle < filters.min_cos_angle) return false;
        if (m.site_contact_fraction != null && m.site_contact_fraction < filters.min_contact_fraction) return false;

        // No cysteine
        if (filters.no_cys && m.binder_sequence && m.binder_sequence.includes('C')) return false;

        // SS bias
        if (filters.ss_bias === 'helix' && m.binder_helix_frac != null && m.binder_helix_frac < 0.4) return false;
        if (filters.ss_bias === 'sheet' && m.binder_sheet_frac != null && m.binder_sheet_frac < 0.3) return false;
        if (filters.ss_bias === 'balanced') {
            if (m.binder_helix_frac != null && m.binder_helix_frac >= 0.6) return false;
            if (m.binder_sheet_frac != null && m.binder_sheet_frac >= 0.4) return false;
        }

        return true;
    }

    function recalcScore(m) {
        if (!m) return null;
        const plddt = m.boltz_binder_plddt != null ? m.boltz_binder_plddt
            : m.esmfold_plddt != null ? m.esmfold_plddt : null;
        const iptm = m.boltz_iptm != null ? m.boltz_iptm : null;
        const dg = m.rosetta_dG != null ? m.rosetta_dG : null;
        if (iptm == null) return null;
        const dg_norm = dg != null ? clamp(-dg / 40, 0, 1) : 0;
        const plddt_norm = plddt != null ? plddt / 100 : 0;
        return filters.w_plddt * plddt_norm + filters.w_iptm * iptm + filters.w_dg * dg_norm;
    }

    function runFilters() {
        designPass = new Array(designs.length);
        designScores = new Array(designs.length);
        for (let i = 0; i < designs.length; i++) {
            designPass[i] = passesFilters(designs[i]);
            designScores[i] = recalcScore(designs[i].metrics);
        }
        updateInstanceVisibility();
        updateStats();
        drawHistogram();
        // If selected design no longer passes, keep it visible but note it
        if (selectedDesignIdx != null) updateDetailCard(selectedDesignIdx);
    }

    function updateInstanceVisibility() {
        const dummy = new THREE.Object3D();
        const _zeroScale = new THREE.Vector3(0, 0, 0);

        for (const [tool, im] of Object.entries(instancedMeshes)) {
            const indices = designIndexMap[tool];

            for (let j = 0; j < indices.length; j++) {
                const d = designs[indices[j]];
                const pass = designPass[indices[j]];

                if (pass) {
                    // Restore real transform
                    const c = d.centroid || [0, 0, 0];
                    const axes = d.ellipsoid_axes || [2, 2, 2];
                    const quat = d.ellipsoid_rotation || [1, 0, 0, 0];
                    dummy.position.set(c[0], c[1], c[2]);
                    dummy.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
                    dummy.scale.set(axes[0], axes[1], axes[2]);
                } else {
                    // Scale to zero — invisible
                    dummy.position.set(0, 0, 0);
                    dummy.quaternion.set(0, 0, 0, 1);
                    dummy.scale.copy(_zeroScale);
                }
                dummy.updateMatrix();
                im.setMatrixAt(j, dummy.matrix);
            }
            im.instanceMatrix.needsUpdate = true;
        }
    }

    // ------------------------------------------------------------------
    // 4. Stats bar + histogram
    // ------------------------------------------------------------------
    function updateStats() {
        if (!statsEl) return;
        const total = designs.length;
        const passing = designPass.filter(Boolean).length;

        // Per-tool counts
        const toolCounts = {};
        designs.forEach((d, i) => {
            if (!designPass[i]) return;
            const t = d.tool || 'unknown';
            toolCounts[t] = (toolCounts[t] || 0) + 1;
        });

        let html = `<span class="font-semibold">${passing.toLocaleString()} / ${total.toLocaleString()} pass</span>`;
        html += '<span class="mx-3 text-slate-600">|</span>';

        const tools = Object.keys(TOOL_COLORS);
        for (const t of tools) {
            if (!designIndexMap[t]) continue;
            const cnt = toolCounts[t] || 0;
            const color = TOOL_COLORS[t] || '#888';
            html += `<span class="inline-flex items-center gap-1 mr-3">` +
                `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color}"></span>` +
                `<span class="text-xs">${t}: ${cnt}</span></span>`;
        }
        statsEl.innerHTML = html;
    }

    function drawHistogram() {
        if (!histCanvas || !histCtx) return;
        const ctx = histCtx;
        const W = histCanvas.width;
        const H = histCanvas.height;
        ctx.clearRect(0, 0, W, H);

        // Collect passing scores
        const vals = [];
        for (let i = 0; i < designs.length; i++) {
            if (designPass[i] && designScores[i] != null) vals.push(designScores[i]);
        }
        if (vals.length === 0) {
            ctx.fillStyle = '#475569';
            ctx.font = '11px sans-serif';
            ctx.fillText('No passing designs', 10, H / 2 + 4);
            return;
        }

        const bins = 30;
        const lo = Math.min(...vals);
        const hi = Math.max(...vals);
        const range = hi - lo || 1;
        const counts = new Array(bins).fill(0);
        for (const v of vals) {
            const b = Math.min(bins - 1, Math.floor((v - lo) / range * bins));
            counts[b]++;
        }
        const maxC = Math.max(...counts, 1);

        // Draw bars
        const barW = W / bins;
        for (let b = 0; b < bins; b++) {
            const barH = (counts[b] / maxC) * (H - 14);
            const x = b * barW;
            ctx.fillStyle = '#3b82f6';
            ctx.fillRect(x + 0.5, H - 12 - barH, barW - 1, barH);
        }

        // Axis labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '9px sans-serif';
        ctx.fillText(lo.toFixed(2), 0, H - 1);
        ctx.fillText(hi.toFixed(2), W - 30, H - 1);
    }

    // ------------------------------------------------------------------
    // 5. Detail card
    // ------------------------------------------------------------------
    function updateDetailCard(idx) {
        if (!detailEl) return;
        if (idx == null || idx < 0 || idx >= designs.length) {
            detailEl.classList.add('hidden');
            return;
        }
        detailEl.classList.remove('hidden');
        const d = designs[idx];
        const m = d.metrics || {};
        const color = TOOL_COLORS[d.tool] || '#888';
        const pass = designPass[idx];
        const score = designScores[idx];
        const seq = m.binder_sequence || '';
        const seqDisplay = seq.length > 80 ? seq.substring(0, 80) + '...' : seq;

        detailEl.innerHTML = `
            <div class="flex items-center gap-3 mb-2">
                <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${color}"></span>
                <span class="font-semibold text-sm">${d.design_id || 'unknown'}</span>
                <span class="text-xs px-2 py-0.5 rounded" style="background:${color}33;color:${color}">${d.tool}</span>
                ${pass ? '<span class="text-xs text-green-400">PASS</span>' : '<span class="text-xs text-red-400">FILTERED</span>'}
            </div>
            <div class="grid grid-cols-4 gap-2 text-xs mb-2">
                <div><span class="text-slate-500">iPTM</span><br><span class="text-white">${fmt(m.boltz_iptm, 3)}</span></div>
                <div><span class="text-slate-500">pLDDT</span><br><span class="text-white">${fmt(m.esmfold_plddt, 1)}</span></div>
                <div><span class="text-slate-500">dG</span><br><span class="text-white">${fmt(m.rosetta_dG, 1)}</span></div>
                <div><span class="text-slate-500">Sc</span><br><span class="text-white">${fmt(m.rosetta_sc, 3)}</span></div>
                <div><span class="text-slate-500">SIF</span><br><span class="text-white">${fmt(m.site_interface_fraction, 3)}</span></div>
                <div><span class="text-slate-500">RMSD</span><br><span class="text-white">${fmt(m.refolding_rmsd, 2)}</span></div>
                <div><span class="text-slate-500">Site PAE</span><br><span class="text-white">${fmt(m.boltz_site_mean_pae, 1)}</span></div>
                <div><span class="text-slate-500">Score</span><br><span class="font-semibold text-blue-400">${fmt(score, 3)}</span></div>
            </div>
            ${seq ? `<div class="text-xs font-mono text-green-400 bg-slate-950 rounded px-2 py-1 break-all">${seqDisplay}</div>` : ''}
            <button id="lr-view-detail" class="mt-2 text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1 rounded">View in Detail Tab</button>
        `;

        // Wire the "View in Detail Tab" button
        const btn = document.getElementById('lr-view-detail');
        if (btn) {
            btn.addEventListener('click', () => {
                // Switch to detail tab and select this design
                const detailSelect = document.getElementById('detail-select');
                if (detailSelect) {
                    detailSelect.value = d.design_id;
                    detailSelect.dispatchEvent(new Event('change'));
                }
                const tabBtn = document.querySelector('.tab-btn[data-tab="detail"]');
                if (tabBtn) tabBtn.click();
            });
        }
    }

    // ------------------------------------------------------------------
    // 6. Mouse interaction (hover + click)
    // ------------------------------------------------------------------
    function getIntersection(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        let closest = null;
        let closestDist = Infinity;

        for (const [tool, im] of Object.entries(instancedMeshes)) {
            const hits = raycaster.intersectObject(im);
            if (hits.length > 0 && hits[0].distance < closestDist) {
                closestDist = hits[0].distance;
                const instanceId = hits[0].instanceId;
                const globalIdx = designIndexMap[tool][instanceId];
                closest = { tool, instanceId, globalIdx, point: hits[0].point };
            }
        }
        return closest;
    }

    function onMouseMove(event) {
        const hit = getIntersection(event);
        if (hit) {
            showTooltip(event, hit.globalIdx);
            renderer.domElement.style.cursor = 'pointer';
        } else {
            hideTooltip();
            renderer.domElement.style.cursor = 'default';
        }
    }

    function onMouseClick(event) {
        const hit = getIntersection(event);
        if (hit) {
            selectDesign(hit.globalIdx);
        }
    }

    function showTooltip(event, idx) {
        if (!tooltipEl) return;
        const d = designs[idx];
        const m = d.metrics || {};
        const score = designScores[idx];
        tooltipEl.innerHTML =
            `<b>${d.design_id || 'unknown'}</b> <span style="color:${TOOL_COLORS[d.tool] || '#888'}">[${d.tool}]</span><br>` +
            `iPTM: ${fmt(m.boltz_iptm, 3)} | pLDDT: ${fmt(m.esmfold_plddt, 1)} | Score: ${fmt(score, 3)}` +
            (m.site_interface_fraction != null ? `<br>SIF: ${fmt(m.site_interface_fraction, 3)}` : '');
        tooltipEl.style.left = (event.clientX + 14) + 'px';
        tooltipEl.style.top = (event.clientY + 14) + 'px';
        tooltipEl.classList.remove('hidden');
    }

    function hideTooltip() {
        if (tooltipEl) tooltipEl.classList.add('hidden');
    }

    function selectDesign(idx) {
        selectedDesignIdx = idx;

        // Remove old marker
        if (selectedMarker) { scene.remove(selectedMarker); selectedMarker = null; }
        if (approachArrow) { scene.remove(approachArrow); approachArrow = null; }

        if (idx == null) {
            updateDetailCard(null);
            return;
        }

        const d = designs[idx];

        // Approach vector arrow
        if (d.interface_centroid && d.approach_vector) {
            const ic = d.interface_centroid;
            const av = d.approach_vector;
            const origin = new THREE.Vector3(ic[0], ic[1], ic[2]);
            const dir = new THREE.Vector3(av[0], av[1], av[2]).normalize();
            approachArrow = new THREE.ArrowHelper(dir, origin, 15, 0x00e5ff, 2, 1.2);
            scene.add(approachArrow);
        }

        updateDetailCard(idx);
    }

    // ------------------------------------------------------------------
    // 7. Camera views
    // ------------------------------------------------------------------
    function setCameraView(name) {
        const r = boundingRadius * 2.2;
        const c = (name === 'site' && siteCentroid3) ? siteCentroid3 : sceneCenter;
        const dist = (name === 'site') ? r : r;

        controls.target.copy(c);

        switch (name) {
            case 'front':   camera.position.set(c.x, c.y, c.z + r); break;
            case 'back':    camera.position.set(c.x, c.y, c.z - r); break;
            case 'top':     camera.position.set(c.x, c.y + r, c.z); camera.up.set(0, 0, -1); break;
            case 'bottom':  camera.position.set(c.x, c.y - r, c.z); camera.up.set(0, 0, 1); break;
            case 'left':    camera.position.set(c.x - r, c.y, c.z); break;
            case 'right':   camera.position.set(c.x + r, c.y, c.z); break;
            case 'site':
                if (siteCentroid3) {
                    // Look at site from the outward-facing normal direction
                    const sn = geometryData.site.normal || [0, 0, 1];
                    camera.position.set(
                        c.x + sn[0] * dist,
                        c.y + sn[1] * dist,
                        c.z + sn[2] * dist
                    );
                }
                break;
            case 'initial':
            default:
                camera.position.set(
                    sceneCenter.x + boundingRadius * 1.2,
                    sceneCenter.y + boundingRadius * 0.6,
                    sceneCenter.z + boundingRadius * 1.5
                );
                controls.target.copy(sceneCenter);
                break;
        }

        // Reset up vector for non-top/bottom views
        if (name !== 'top' && name !== 'bottom') {
            camera.up.set(0, 1, 0);
        }
        camera.lookAt(c);
        controls.update();
    }

    function wireViewButtons() {
        document.querySelectorAll('[data-camera-view]').forEach(btn => {
            btn.addEventListener('click', () => {
                setCameraView(btn.dataset.cameraView);
            });
        });
    }

    // ------------------------------------------------------------------
    // 8. Render loop
    // ------------------------------------------------------------------
    function animate() {
        if (!animating) return;
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    function startAnimation() {
        if (!animating) {
            animating = true;
            animate();
        }
    }

    function stopAnimation() {
        animating = false;
    }

    // ------------------------------------------------------------------
    // 8. Resize
    // ------------------------------------------------------------------
    function onResize() {
        const container = document.getElementById('livererank-viewport');
        if (!container || !renderer) return;
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w === 0 || h === 0) return;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    // ------------------------------------------------------------------
    // 9. Filter panel wiring
    // ------------------------------------------------------------------
    function readFilters() {
        const val = (id, fallback) => {
            const el = document.getElementById(id);
            if (!el) return fallback;
            return parseFloat(el.value) || fallback;
        };

        filters.min_iptm = val('lr-min-iptm', 0);
        filters.min_plddt = val('lr-min-plddt', 0);
        filters.max_refolding_rmsd = val('lr-max-refolding-rmsd', 10);
        filters.max_interface_pae = val('lr-max-interface-pae', 30);
        filters.max_site_pae = val('lr-max-site-pae', 30);
        filters.min_sc = val('lr-min-sc', 0);
        filters.max_interface_ke = val('lr-max-interface-ke', 1);
        filters.max_aa_fraction = val('lr-max-aa-fraction', 1);

        filters.min_sif = val('lr-min-sif', 0);
        filters.max_centroid_dist = val('lr-max-centroid-dist', 40);
        filters.min_cos_angle = val('lr-min-cos-angle', -1);
        filters.min_contact_fraction = val('lr-min-contact-fraction', 0);

        filters.w_plddt = val('lr-w-plddt', 0.3);
        filters.w_iptm = val('lr-w-iptm', 0.6);
        filters.w_dg = val('lr-w-dg', 0.1);

        const noCys = document.getElementById('lr-no-cys');
        filters.no_cys = noCys ? noCys.checked : false;

        const ssRadio = document.querySelector('input[name="lr_ss_bias"]:checked');
        filters.ss_bias = ssRadio ? ssRadio.value : 'any';

        // Tool toggles
        filters.active_tools = new Set();
        document.querySelectorAll('.lr-tool-cb:checked').forEach(cb => {
            filters.active_tools.add(cb.dataset.tool);
        });
    }

    function onFilterChange() {
        // Batch via rAF to avoid jank during fast slider drag
        if (filterRAF) cancelAnimationFrame(filterRAF);
        filterRAF = requestAnimationFrame(() => {
            readFilters();
            runFilters();
            filterRAF = null;
        });
    }

    function wireFilterControls() {
        // Wire all range/number inputs in the filter panel
        document.querySelectorAll('#livererank-filters input[type="range"], #livererank-filters input[type="number"]').forEach(el => {
            el.addEventListener('input', () => {
                // Sync linked display if this is a range slider
                syncLinkedDisplay(el);
                onFilterChange();
            });
        });

        // Checkbox + radio
        document.querySelectorAll('#livererank-filters input[type="checkbox"], #livererank-filters input[type="radio"]').forEach(el => {
            el.addEventListener('change', onFilterChange);
        });

        // Build tool checkboxes
        const toolDiv = document.getElementById('lr-tool-toggles');
        if (toolDiv) {
            const tools = [...new Set(designs.map(d => d.tool).filter(Boolean))].sort();
            toolDiv.innerHTML = '';
            for (const t of tools) {
                const color = TOOL_COLORS[t] || '#888';
                const label = document.createElement('label');
                label.className = 'flex items-center gap-2 cursor-pointer text-xs';
                label.innerHTML = `<input type="checkbox" checked class="lr-tool-cb border-slate-600" data-tool="${t}">` +
                    `<span style="color:${color}">${t}</span>`;
                toolDiv.appendChild(label);
                filters.active_tools.add(t);
            }
            // Wire tool checkboxes
            toolDiv.querySelectorAll('.lr-tool-cb').forEach(cb => {
                cb.addEventListener('change', onFilterChange);
            });
        }

        // All / None buttons
        document.getElementById('lr-tools-all')?.addEventListener('click', () => {
            document.querySelectorAll('.lr-tool-cb').forEach(cb => { cb.checked = true; });
            onFilterChange();
        });
        document.getElementById('lr-tools-none')?.addEventListener('click', () => {
            document.querySelectorAll('.lr-tool-cb').forEach(cb => { cb.checked = false; });
            onFilterChange();
        });

        // Reset button
        document.getElementById('lr-reset')?.addEventListener('click', resetFilters);

        // Export CSV
        document.getElementById('lr-export-csv')?.addEventListener('click', exportCSV);

        // Apply to Rerank tab
        document.getElementById('lr-apply-rerank')?.addEventListener('click', applyToRerank);
    }

    function syncLinkedDisplay(el) {
        // If element has a linked display span (id = el.id + '-val')
        const linked = document.getElementById(el.id + '-val');
        if (linked) linked.textContent = el.value;
        // If it is a number input linked to a range slider
        const rangeId = el.dataset.range;
        if (rangeId) {
            const range = document.getElementById(rangeId);
            if (range) range.value = el.value;
        }
        const numberId = el.dataset.number;
        if (numberId) {
            const num = document.getElementById(numberId);
            if (num) num.value = el.value;
        }
    }

    function resetFilters() {
        const defaults = {
            'lr-min-iptm': 0, 'lr-min-plddt': 0, 'lr-max-refolding-rmsd': 10,
            'lr-max-interface-pae': 30, 'lr-max-site-pae': 30, 'lr-min-sc': 0,
            'lr-max-interface-ke': 1, 'lr-max-aa-fraction': 1,
            'lr-min-sif': 0, 'lr-max-centroid-dist': 40, 'lr-min-cos-angle': -1,
            'lr-min-contact-fraction': 0,
            'lr-w-plddt': 0.3, 'lr-w-iptm': 0.6, 'lr-w-dg': 0.1
        };
        for (const [id, val] of Object.entries(defaults)) {
            const el = document.getElementById(id);
            if (el) {
                el.value = val;
                syncLinkedDisplay(el);
            }
        }
        const noCys = document.getElementById('lr-no-cys');
        if (noCys) noCys.checked = false;

        const anyRadio = document.querySelector('input[name="lr_ss_bias"][value="any"]');
        if (anyRadio) anyRadio.checked = true;

        document.querySelectorAll('.lr-tool-cb').forEach(cb => { cb.checked = true; });

        onFilterChange();
    }

    // ------------------------------------------------------------------
    // 10. Export + Apply
    // ------------------------------------------------------------------
    function exportCSV() {
        const passingDesigns = [];
        for (let i = 0; i < designs.length; i++) {
            if (!designPass[i]) continue;
            passingDesigns.push({ ...designs[i], combined_score_live: designScores[i] });
        }
        if (passingDesigns.length === 0) {
            alert('No designs pass the current filters.');
            return;
        }

        // Determine columns from the first design's metrics
        const metricKeys = Object.keys(passingDesigns[0].metrics || {});
        const header = ['rank', 'design_id', 'tool', 'combined_score_live', ...metricKeys];
        // Sort by score descending
        passingDesigns.sort((a, b) => (b.combined_score_live || 0) - (a.combined_score_live || 0));

        let csv = header.join(',') + '\n';
        passingDesigns.forEach((d, i) => {
            const m = d.metrics || {};
            const row = [
                i + 1,
                d.design_id || '',
                d.tool || '',
                d.combined_score_live != null ? d.combined_score_live.toFixed(4) : '',
                ...metricKeys.map(k => {
                    const v = m[k];
                    if (v == null) return '';
                    if (typeof v === 'number') return v.toFixed(6);
                    // Escape strings with commas/quotes
                    const s = String(v);
                    if (s.includes(',') || s.includes('"') || s.includes('\n')) {
                        return '"' + s.replace(/"/g, '""') + '"';
                    }
                    return s;
                })
            ];
            csv += row.join(',') + '\n';
        });

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `livererank_${JOB_ID}_filtered.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function applyToRerank() {
        // Switch to rerank tab and populate fields from current filter values
        const tabBtn = document.querySelector('.tab-btn[data-tab="rerank"]');
        if (tabBtn) tabBtn.click();

        // Map filter values to rerank form fields
        const mappings = {
            'score_weights': () => `${filters.w_plddt},${filters.w_iptm},${filters.w_dg}`,
            'max_aa_fraction': () => filters.max_aa_fraction < 1 ? filters.max_aa_fraction : '',
            'min_sc': () => filters.min_sc > 0 ? filters.min_sc : '',
            'max_refolding_rmsd': () => filters.max_refolding_rmsd < 10 ? filters.max_refolding_rmsd : '',
            'max_interface_ke': () => filters.max_interface_ke < 1 ? filters.max_interface_ke : '',
            'min_site_interface_fraction': () => filters.min_sif > 0 ? filters.min_sif : '',
        };
        for (const [name, fn] of Object.entries(mappings)) {
            const el = document.querySelector(`#rerank-form [name="${name}"]`);
            if (el) el.value = fn();
        }

        // No-cys checkbox
        const noCysField = document.querySelector('#rerank-form [name="no_cys"]');
        if (noCysField) noCysField.checked = filters.no_cys;

        // SS bias select
        const ssField = document.querySelector('#rerank-form [name="ss_bias"]');
        if (ssField) ssField.value = filters.ss_bias;
    }

    // ------------------------------------------------------------------
    // 11. UI element creation
    // ------------------------------------------------------------------
    function createUIElements() {
        // Tooltip (fixed-position floating div)
        tooltipEl = document.createElement('div');
        tooltipEl.id = 'lr-tooltip';
        tooltipEl.className = 'hidden';
        tooltipEl.style.cssText =
            'position:fixed;z-index:9999;background:#1e293b;border:1px solid #334155;' +
            'border-radius:6px;padding:6px 10px;font-size:11px;color:#e2e8f0;' +
            'pointer-events:none;max-width:300px;line-height:1.5;box-shadow:0 4px 12px rgba(0,0,0,0.4);';
        document.body.appendChild(tooltipEl);

        // Stats bar
        statsEl = document.getElementById('livererank-stats');

        // Histogram canvas
        histCanvas = document.getElementById('livererank-histogram');
        if (histCanvas) {
            histCanvas.width = 200;
            histCanvas.height = 60;
            histCtx = histCanvas.getContext('2d');
        }

        // Detail card
        detailEl = document.getElementById('livererank-detail');
    }

    // ------------------------------------------------------------------
    // 12. Public init
    // ------------------------------------------------------------------
    async function init() {
        if (initialized) {
            // Already loaded — just resume animation
            startAnimation();
            onResize();
            return;
        }

        // Check that THREE is available
        if (typeof THREE === 'undefined') {
            console.error('LiveRerank: Three.js not loaded');
            const vp = document.getElementById('livererank-viewport');
            if (vp) vp.innerHTML = '<p class="text-red-400 p-4">Three.js library not loaded. Check script tags.</p>';
            return;
        }

        createUIElements();

        const ok = await fetchGeometry();
        if (!ok || designs.length === 0) {
            const vp = document.getElementById('livererank-viewport');
            if (vp && !vp.querySelector('.text-red-400')) {
                vp.innerHTML = '<p class="text-slate-400 p-8">No geometry data available for this job. Run the pipeline with site specification to enable 3D visualization.</p>';
            }
            return;
        }

        // Build bright color palette for white background
        for (const [tool, hex] of Object.entries(TOOL_COLORS)) {
            BRIGHT_COLORS[tool] = brightenColor(hex);
        }

        initScene();
        buildScene();

        // Initialize filters — all tools active by default
        filters.active_tools = new Set(designs.map(d => d.tool).filter(Boolean));
        wireFilterControls();
        wireViewButtons();

        // Initial filter pass
        readFilters();
        runFilters();

        initialized = true;
        startAnimation();
    }

    function onTabHide() {
        stopAnimation();
        hideTooltip();
    }

    function dispose() {
        stopAnimation();
        hideTooltip();
        if (tooltipEl && tooltipEl.parentNode) {
            tooltipEl.parentNode.removeChild(tooltipEl);
        }
        if (renderer) {
            renderer.domElement.removeEventListener('mousemove', onMouseMove);
            renderer.domElement.removeEventListener('click', onMouseClick);
            renderer.dispose();
        }
        window.removeEventListener('resize', onResize);

        // Dispose Three.js resources
        for (const im of Object.values(instancedMeshes)) {
            im.geometry.dispose();
            im.material.dispose();
            scene.remove(im);
        }
        if (targetMesh) { targetMesh.geometry.dispose(); targetMesh.material.dispose(); }
        if (targetWire) { targetWire.geometry.dispose(); targetWire.material.dispose(); }
        for (const s of siteSpheres) { s.geometry.dispose(); s.material.dispose(); }
        if (siteCentroidSphere) { siteCentroidSphere.geometry.dispose(); siteCentroidSphere.material.dispose(); }
        if (selectedMarker) { selectedMarker.geometry.dispose(); selectedMarker.material.dispose(); }

        instancedMeshes = {};
        designIndexMap = {};
        initialized = false;
    }

    // ------------------------------------------------------------------
    // Expose public API
    // ------------------------------------------------------------------
    window.LiveRerank = {
        init: init,
        onTabHide: onTabHide,
        dispose: dispose,
        isInitialized: () => initialized
    };

})();
