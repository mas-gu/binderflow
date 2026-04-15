# ProteaFlow Design System

## 1. Visual Theme & Atmosphere

ProteaFlow is a scientific data platform for protein binder design and structure-based drug design. The UI is dark-mode-native — a deep slate canvas where data-dense tables, scatter plots, and 3D molecular structures are the primary content. The design prioritizes information density and readability over decoration: every pixel serves either data or navigation.

The aesthetic is clinical precision meets developer tooling — closest to Linear's ultra-minimal dark UI, but adapted for scientific workflows with color-coded tool identification, data gradient tables, and embedded 3D viewers. The overall impression should be: "this is where serious computational biology happens."

**Key Characteristics:**
- Dark-mode-native: deep slate backgrounds, no light theme
- Data-dense: AG Grid tables, Plotly charts, 3Dmol.js viewers are the primary content
- Tool colors as identity: each computational tool has a distinctive color used consistently across all views
- Muted chrome, loud data: the UI frame is quiet (slate-700 borders, slate-400 text), while data points and scores use vivid colors
- Scientific typography: monospace for sequences and SMILES, compact sans-serif for everything else
- No shadows, minimal depth: flat cards with thin borders — depth comes from background color steps, not box-shadow

## 2. Color Palette

### Background Surfaces (darkest → lightest)
| Token | Hex | Tailwind | Usage |
|-------|-----|----------|-------|
| `bg-body` | `#0a0f1a` | `bg-slate-950` | Page body, main canvas |
| `bg-card` | `#0f172a` | `bg-slate-900` | Cards, panels, nav bar, table cells |
| `bg-input` | `#1e293b` | `bg-slate-800` | Input fields, hover states, AG Grid header |
| `bg-elevated` | `#334155` | `bg-slate-700` | Buttons (secondary), active tabs, tooltips |

### Text (brightest → dimmest)
| Token | Hex | Tailwind | Usage |
|-------|-----|----------|-------|
| `text-primary` | `#f1f5f9` | `text-slate-100` | Body text, default content |
| `text-secondary` | `#cbd5e1` | `text-slate-300` | Section headings, emphasis |
| `text-tertiary` | `#94a3b8` | `text-slate-400` | Labels, metadata, axis titles |
| `text-muted` | `#64748b` | `text-slate-500` | Hints, placeholders, timestamps |
| `text-faint` | `#475569` | `text-slate-600` | Disabled text, de-emphasized |

### Accent & Status
| Token | Hex | Tailwind | Usage |
|-------|-----|----------|-------|
| `accent-primary` | `#3b82f6` | `text-blue-500` / `bg-blue-600` | Primary actions, active tabs, links |
| `accent-hover` | `#60a5fa` | `text-blue-400` | Hover state on primary |
| `success` | `#10b981` | `text-emerald-400` | Completed, pass, available |
| `warning` | `#f59e0b` | `text-amber-400` | In-progress, caution, molecule rerank |
| `error` | `#ef4444` | `text-red-400` | Failed, critical, high utilization |

### Status Badges
| State | Background | Text | Border |
|-------|-----------|------|--------|
| Running | `bg-blue-900` | `text-blue-300` | — |
| Completed | `bg-emerald-900/40` | `text-emerald-300` | `border-emerald-700` |
| Failed | `bg-red-900/40` | `text-red-300` | `border-red-700` |
| Busy | `bg-amber-900` | `text-amber-300` | — |
| Free | `bg-green-900` | `text-green-300` | — |

### Border & Dividers
| Token | Hex | Tailwind | Usage |
|-------|-----|----------|-------|
| `border-default` | `#334155` | `border-slate-700` | Cards, sections, tables |
| `border-input` | `#475569` | `border-slate-600` | Input fields, dropzone |
| `border-active` | `#3b82f6` | `border-blue-500` | Active tab underline |

### Tool Identity Colors
Each generative tool has a unique color used consistently in scatter plots, table dots, dashboard charts, and PyMOL scripts.

**Binder tools:**
| Tool | Hex | Visual |
|------|-----|--------|
| RFdiffusion | `#E53935` | Red |
| RFdiffusion3 | `#00BCD4` | Cyan |
| BoltzGen | `#43A047` | Green |
| BindCraft | `#1E88E5` | Blue |
| PXDesign | `#FB8C00` | Orange |
| Proteina | `#8E24AA` | Purple |
| Proteina Complexa | `#00897B` | Teal |

**Molecule tools:**
| Tool | Hex | Visual |
|------|-----|--------|
| PocketFlow | `#E53935` | Red |
| MolCRAFT | `#7C3AED` | Violet |
| PocketXMol | `#0EA5E9` | Sky |
| Library | `#FF9800` | Amber |

## 3. Typography

### Font Stack
- **Primary:** System sans-serif via Tailwind defaults (`ui-sans-serif, system-ui, -apple-system, sans-serif`)
- **Monospace:** `JetBrains Mono, Fira Code, Cascadia Code, ui-monospace, monospace` — for SMILES, sequences, log output, code

### Scale
| Role | Tailwind | Size | Weight | Usage |
|------|----------|------|--------|-------|
| Page title | `text-2xl font-bold` | 24px | 700 | H1 headers |
| Section header | `text-lg font-semibold` | 18px | 600 | H2, card titles |
| Body | `text-sm` | 14px | 400 | Default text, table cells |
| Label | `text-xs` | 12px | 400 | Badges, metadata, filter labels |
| Score value | `text-lg font-semibold` | 18px | 600 | Score cards (colored) |
| Monospace | `text-sm font-mono` | 14px | 400 | SMILES, sequences |

### Principles
- No custom fonts loaded — system fonts only (fast, no FOUT)
- Monospace is mandatory for scientific data: SMILES strings, amino acid sequences, file paths
- Labels are always `text-xs text-slate-400/500` — small, muted, never competing with data
- Score values use accent colors (`text-blue-400`, `text-emerald-400`, `text-amber-400`) at `text-lg`

## 4. Component Styles

### Cards / Panels
```
bg-slate-900 border border-slate-700 rounded-lg p-4
```
- The universal container: GPU cards, job cards, filter panels, score card groups
- Never use shadows — depth from background color step only
- `p-4` for standard content, `p-6` for form sections
- Always `rounded-lg` (8px)

### Buttons

**Primary action:**
```
bg-blue-600 hover:bg-blue-500 text-white px-4 py-1 rounded font-medium
```
Usage: "New Job", "Submit", "Run Rerank"

**Secondary action:**
```
bg-slate-700 hover:bg-slate-600 text-sm px-3 py-1 rounded
```
Usage: "Filters", "Download CSV", "Copy"

**Toggle / Tab selector:**
```
px-6 py-2 rounded-lg font-semibold
Active:   bg-blue-600 text-white     (binder)
Active:   bg-amber-600 text-white    (molecule)
Inactive: bg-slate-700 text-slate-400
```

**Danger / Cancel:**
```
bg-red-600 hover:bg-red-500 text-white px-3 py-1 rounded
```

### Input Fields
```
w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm
```
- All inputs: text, number, select, file
- Focus: browser default (no custom focus ring needed in dark theme)
- Placeholder: `text-slate-500`

### Checkboxes & Radios
```
<label class="flex items-center gap-2 cursor-pointer">
    <input type="checkbox" class="border-slate-600">
    <span class="text-sm">Label text</span>
</label>
```
- Tool checkboxes include a colored dot: `w-3 h-3 rounded-full inline-block` with tool color

### Tabs
```
<div class="border-b border-slate-700 mb-4">
    <div class="flex gap-1">
        <!-- Active -->
        <button class="px-4 py-2 text-sm rounded-t border-b-2 border-blue-500 text-blue-400">
        <!-- Inactive -->
        <button class="px-4 py-2 text-sm rounded-t border-b-2 border-transparent text-slate-400 hover:text-white">
    </div>
</div>
```

### AG Grid Table
```css
--ag-background-color: #0f172a;
--ag-header-background-color: #1e293b;
--ag-odd-row-background-color: #0f172a;
--ag-row-hover-color: #1e293b;
--ag-border-color: #334155;
--ag-header-foreground-color: #94a3b8;
--ag-foreground-color: #e2e8f0;
--ag-font-size: 13px;
--ag-row-height: 32px;
--ag-header-height: 36px;
```

### Score Cards (Design Detail)
```html
<div class="grid grid-cols-4 gap-2">
    <div class="bg-slate-900 border border-slate-700 rounded p-2">
        <p class="text-xs text-slate-500">Label</p>
        <p class="text-lg font-semibold text-blue-400">0.750</p>
    </div>
</div>
```
- 4-column grid for top metrics (Score, QED, SA, Vina)
- Each card: muted label above, colored value below
- Color by meaning: blue=score, emerald=QED, amber=SA, red=Vina

### Property Tables (Design Detail)
```html
<table>
    <tr class="border-b border-slate-800">
        <td class="py-1.5 text-xs text-slate-400">Property name</td>
        <td class="py-1.5 text-sm font-medium text-slate-200">Value</td>
    </tr>
</table>
```
- Two-column layout: physicochemical left, medicinal chemistry right
- Section headers: `text-xs font-semibold text-slate-500 uppercase tracking-wider`
- Pass/Fail values: `text-emerald-400` / `text-red-400`

### VRAM Progress Bar
```html
<div class="bg-slate-800 rounded-full h-2.5">
    <div class="h-2.5 rounded-full" style="width: 65%">
        <!-- <= 40%: bg-green-500, 40-80%: bg-amber-500, >80%: bg-red-500 -->
    </div>
</div>
```

### Status Badges
```
text-xs px-2 py-0.5 rounded
```
Color mapping per state (see Status Badges in palette section).

## 5. Spacing System

Uses Tailwind's default 4px base unit.

| Scale | Value | Usage |
|-------|-------|-------|
| `gap-1` / `p-1` | 4px | Tight spacing (badge padding) |
| `gap-2` / `p-2` | 8px | Score cards, compact layouts |
| `gap-3` / `p-3` | 12px | Between form fields |
| `gap-4` / `p-4` | 16px | Standard card padding, grid gaps |
| `p-6` | 24px | Form section padding |
| `mb-4` | 16px | Between sections |
| `mb-6` | 24px | Between major page sections |

### Grid Patterns
- GPU cards: `grid grid-cols-1 md:grid-cols-3 gap-4`
- Score cards: `grid grid-cols-4 gap-2`
- Form fields: `grid grid-cols-1 md:grid-cols-2 gap-4`
- Filter grid: `grid grid-cols-2 md:grid-cols-3 gap-4`
- Job cards: `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3`

## 6. Charts & Data Visualization

### Plotly Charts
```javascript
layout = {
    plot_bgcolor: '#0f172a',      // slate-900
    paper_bgcolor: '#0f172a',
    font: { color: '#94a3b8' },   // slate-400
    xaxis: { gridcolor: '#1e293b', color: '#94a3b8' },
    yaxis: { gridcolor: '#1e293b', color: '#94a3b8' },
    legend: { font: { color: '#cbd5e1' } },
    margin: { t: 30 },
}
```
- Background matches card bg (seamless)
- Grid lines: one step lighter (`slate-800`)
- Data points colored by tool identity
- Marker size: 8px, opacity: 0.8

### 3Dmol.js Viewer
```javascript
backgroundColor: '#0f172a'    // matches card bg
```
- Protein: grey cartoon, opacity 0.7
- Molecule: colored sticks (tool color), translucent VDW surface
- Binding site: red sticks
- Contacts: dashed lines (blue=H-bond, yellow=hydrophobic, red=salt bridge, green=pi-stack)

### Dashboard (matplotlib)
- Generated server-side as PNG
- 6-panel layout: boxplots, histograms, scatter, bar chart, table
- Tool colors match web UI colors exactly

## 7. Navigation

```html
<nav class="bg-slate-900 border-b border-slate-700 px-6 py-3">
    <a class="text-xl font-bold text-blue-400">ProteaFlow</a>
    <a class="text-slate-300 hover:text-white px-3 py-1 rounded hover:bg-slate-800">Link</a>
    <a class="bg-blue-600 hover:bg-blue-500 text-white px-4 py-1 rounded font-medium">New Job</a>
</nav>
```

## 8. CDN Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Tailwind CSS | CDN (latest) | Utility-first styling |
| AG Grid Community | 33.1.1 | Data tables with sorting/filtering |
| Plotly.js | 2.35.2 | Scatter plots, box plots, histograms |
| 3Dmol.js | latest | Protein/molecule 3D visualization |
| Three.js | 0.147.0 | 3D rendering (live rerank) |

No build step. No npm. All via CDN script tags.

## 9. Do's and Don'ts

### Do
- Use tool identity colors consistently — same color for RFdiffusion everywhere
- Keep labels small and muted (`text-xs text-slate-400`)
- Let data be loud — score values, plot points, 3D structures are the content
- Use the card pattern (`bg-slate-900 border border-slate-700 rounded-lg`) for any container
- Maintain the background hierarchy: body → card → input (950 → 900 → 800)

### Don't
- Don't use shadows — this is a flat, borderline-brutalist design
- Don't use light backgrounds — everything is dark, always
- Don't add decorative elements (gradients, icons, illustrations)
- Don't use custom colors outside the defined palette — especially not for data
- Don't use font sizes larger than `text-2xl` — this is a tool, not a marketing page
- Don't load custom web fonts — system fonts keep it fast and scientific

## 10. Potential Improvements (Inspired by Linear)

### Currently missing, worth adding:
1. **Subtle border opacity** — Linear uses `rgba(255,255,255,0.05)` borders instead of solid colors. Would make the UI feel more refined. Replace `border-slate-700` with a custom `border-white/5`.
2. **Transition animations** — Tab switches, hover states, and panel reveals could benefit from `transition-colors duration-150`. Currently instant.
3. **Better focus states** — Custom focus ring (`ring-2 ring-blue-500/50 ring-offset-2 ring-offset-slate-900`) for keyboard navigation.
4. **Loading skeletons** — Rankings table shows blank while loading. Pulsing skeleton rows (`animate-pulse bg-slate-800`) would feel more polished.
5. **Toast notifications** — Job completed/failed notifications currently require manual page check. A subtle toast (bottom-right, auto-dismiss) would improve UX.
6. **Micro-interactions** — Score cards could animate when values change (subtle scale or color flash).
