# S_Harko: Technical Reference

> **Program Name:** S_Harko (derived from <em>Swarm Harker</em>) <br>
> **Core Architecture:** Asynchronous FFT Patterson Engine + Symmetry Minimum Function (SMF) / Buerger Superposition + WebGPU Parallel PSO <br> 
> **Symmetry Database:** 230 Space Groups (cctbx_space_groups_all_settings_v6.json)<br>
> **Version:** Initial Harko version 12 oct 2025, last updated August 7, 2026.<br>

## 1. Theoretical Background & The Phase Problem

In X-ray crystallographic analyses, experimentally measured diffraction intensities $I(h,k,l)$ are directly proportional to the squared structure factor magnitudes $|F(\mathbf{h})|^2$. However, phase information $\phi(\mathbf{h})$ is inherently lost during measurement. S_Harko bypasses the direct phase requirement by synthesizing the 3D Patterson function, which represents the real-space autocorrelation of crystal electron density:

$$
P(\mathbf{u}) = \frac{1}{V} \sum_{\mathbf{h}} |F_{\mathbf{h}}|^2 e^{2\pi i \mathbf{h}\cdot\mathbf{u}} = \frac{1}{V} \sum_{\mathbf{h}} |F_{\mathbf{h}}|^2 \cos(2\pi \mathbf{h}\cdot\mathbf{u})
$$

where $\mathbf{u} = (u,v,w)$ are fractional Patterson coordinates and $V$ represents the real-space unit cell volume.

### Physical Properties of the Patterson Map

- **Interatomic Vector Representation:** Patterson map peaks do not directly correspond to atomic locations $\mathbf{x}_i$. Instead, each peak located at $\mathbf{u}$ represents an interatomic displacement vector $\mathbf{u}_{ij} = \mathbf{x}_i - \mathbf{x}_j$ between atoms $i$ and $j$.
- **Weighting Factor:** Peak heights scale proportionally with the product of the atomic numbers of the interacting pair ($Z_i Z_j$). As a consequence, heavy atoms generate dominant peaks that stand out clearly against lighter organic matrices.
- **Origin Peak:** The point at $(0,0,0)$ corresponds to all self-vectors ($\mathbf{x}_i - \mathbf{x}_i = \mathbf{0}$) with total integrated power $\sum Z_k^2$. S_Harko automatically masks a spherical volume around the origin to prevent scalar distortion in peak finding and colour map rendering.
- **Centrosymmetry:** Regardless of whether the space group is non-centrosymmetric, the Patterson function is strictly centrosymmetric ($P(\mathbf{u}) = P(-\mathbf{u})$).

### Harker Sections

Symmetry elements within space groups generate interatomic vectors constrained to specialized planes or lines termed <strong>Harker sections</strong>. For a general space group operation $\mathbf{S}\mathbf{x} = \mathbf{R}\mathbf{x} + \mathbf{t}$, symmetry-equivalent atoms generate Harker vectors defined by:

$$
\mathbf{u}_H = \mathbf{x} - (\mathbf{R}\mathbf{x} + \mathbf{t}) = (\mathbf{I} - \mathbf{R})\mathbf{x} - \mathbf{t}
$$

By evaluating peak maxima on these constrained planes, S_Harko extracts discrete fractional atomic coordinates directly from one-dimensional and two-dimensional searches.

## 2. Program Architecture

S_Harko isolates computational workloads across thread boundaries to guarantee smooth UI interaction at high animation frame rates.

### Multi-Threaded System Map

| Execution Layer | Primary Responsibility | Source Components |
|-----------------|------------------------|-------------------|
| **Main Thread (UI)** | File parsing, UI state management, Three.js 3D structure rendering, 2D slice visualizer, PDF report generation. | `Harko.html`, `patterson3d.js`, `style.css`, `sg_engine.js` |
| **Web Worker Engine** | Asynchronous 3D Radix-2 FFT synthesis, Lorch filtering, peak finding, SMF & Buerger minimum superposition map generation, site consolidation. | `sharko_worker.js`, `symmetry_utils.js` |
| **WebGPU Compute Shader** | Hardware-accelerated global Particle Swarm Optimization, GPU symmetry expansion, minimum contact evaluations. | `swarm_compute.wgsl` |

### 2.2 Web Worker Pipeline

When data files are uploaded or parameters change, S_Harko triggers a background worker thread via a single `CALCULATE` execution message:

```
STAGE 1: ASYNCHRONOUS FFT PATTERSON SYNTHESIS
  ├─ Expand unique reflections across full Friedel sphere using space group symmetry operators
  ├─ Apply Lorch modification envelope: I *= (1-s) + s * sinc(π * d* * d_min)
  ├─ Grid determination: N = next_power_of_two( max(requested_grid, 2*h_max + 1) )
  ├─ Populate complex grid array with intensity values I(h,k,l)
  ├─ Execute 3D Radix-2 Cooley-Tukey Inverse FFT
  ├─ Map normalization: P(u,v,w) = Real(FFT) / Volume
  └─ Extract metadata: d_min resolution limit, calculated peak width σ = max(0.26 * d_min, 0.7 * dx)

STAGE 2: PEAK DETECTION, SMF & BUERGER SUPERPOSITION
  ├─ Generate origin mask: exclude all grid points within 1.1 Å radius of (0,0,0)
  ├─ Calculate map statistics (mean, sigma) over non-origin voxels
  ├─ Compute Symmetry Minimum Function (SMF) across all non-identity space group operators
  ├─ Fallback (if P1): Compute Buerger Minimum Superposition Map M(u) = min(P(u), P(u - u_top))
  └─ Extract peaks from SMF/Supermap (unmasked) as consolidated absolute candidate sites
```

### 2.3 WebGPU Swarm Optimization

The third piece of the pipeline, and the only one that runs on the GPU rather than the main thread or the worker, is the particle swarm that searches for atomic coordinates once the map and its peaks exist. Rather than duplicate that material here, this subsection is a pointer: the full treatment &ndash; the update equations, the fitness function, and importantly the two separate hardware limits that govern how large a search you can run &ndash; lives in section 6, with the hardware-limits half specifically in 6.3. The short version, for the architectural picture: the main thread builds the WGSL compute shader from `swarm_compute.wgsl`, sizing a couple of its constants to your actual GPU before compiling it, then dispatches one workgroup per particle every generation and reads back only the best-found fitness and position, keeping the heavy per-particle arithmetic entirely off the JavaScript thread.

### 2.4 Symmetry & Operator Resolution

Every stage of the pipeline &ndash; reflection expansion, Harker section geometry, the SMF, and the swarm's own symmetry expansion of trial atoms &ndash; needs the same thing first: the actual list of symmetry operators for whichever space group you picked. That sounds like a simple lookup, but a single space-group <em>number</em> can correspond to more than one standard <em>setting</em> in the database (different origin choices or axis conventions produce different operator lists for what is nominally "the same" space group), so S_Harko has to decide which setting you actually mean. It does this by trying to match the symbol your data file supplies against the settings on record for that number; if there is only one setting on file it uses that regardless, and if there are several and none matches the file's symbol, it falls back to the first on record and says so explicitly &ndash; because picking the wrong setting here does not cause an obvious failure downstream, it just quietly makes every Harker section, every SMF-derived site, and every swarm-generated symmetry copy wrong in a way that looks like a normal failed structure solution rather than a mismatched space group. If a structure refuses to solve and you're not certain the setting is right, this resolution step, not the swarm, is the first thing worth checking.

## 3. Data Input & Intensity Correction

### 3.1 File Formats

S_Harko reads ASCII data exports generated by powder diffraction profile refinement tools (such as Powder 5). It also robustly supports generic peak lists, such as the standard SHELX HKLF 4 format (e.g., lines containing <code>h, k, l, intensity, sigma</code>). If metadata like the Space Group or Cell parameters are found in the header, S_Harko will auto-populate the UI. Otherwise, they can be set manually.

### 3.2 Intensity Classification Hierarchy

Synthesis of a physically accurate Patterson map requires pure structure factor magnitudes squared, $|F(\mathbf{h})|^2$. However, raw measured powder diffraction peak intensities ($I_{\text{obs}}$ or $I_{hkl}$) carry geometric and instrumental distortions:

- **Lorentz–Polarization Factor ($Lp$):** Instrument geometry and beam polarization artificially inflate measured intensities at low and high scattering angles.
- **Multiplicity ($m$):** Overlapping symmetry-equivalent reflections in powder diffraction scale the integrated peak area proportionally to the orbit multiplicity.

If $Lp$ and $m$ are not removed, low-angle reflections artificially dominate the Patterson synthesis, distorting vector peak heights and masking true interatomic vectors.

Because profile refinement software exports vary, S_Harko inspects the uploaded file columns and automatically applies the highest available priority tier:

| Priority Tier | Detected Columns | Intensity Calculation ($|F|^2$) | Processing Route & Description |
|---------------|------------------|---------------------------------|--------------------------------|
| **Tier 1 (Highest)** | `|Fo|` | $|F|^2 = |F_o|^2$ | Direct $|F_o|$ values supplied by the file. The refinement software has already removed $m$ and $Lp$ at high floating-point precision using refined $2\theta$ positions. No further corrections are applied. |
| **Tier 2** | `I_hkl, m, Lp` | $|F|^2 = \frac{I_{hkl}}{m \cdot Lp}$ | Explicit $m$ and $Lp$ columns present in the table header are used directly to isolate $|F|^2$ per reflection. |
| **Tier 3** | `I_hkl, 2th` | $|F|^2 = \frac{I_{hkl}}{Lp(2\theta)}$ | $Lp(2\theta)$ is calculated by S_Harko using the beam polarisation constant $K$ read from the file header. Multiplicity $m$ is handled during symmetry orbit expansion. |
| **Tier 4 (Fallback)** | `I_hkl only` | $|F|^2 = I_{hkl}$ | Neither $|F_o|$ nor $2\theta$/$Lp$ data are available. Uncorrected raw peak areas are used. Low-angle reflections will dominate the map, and a warning is displayed. |

## 4. Patterson Map Synthesis

### 4.1 FFT Synthesis Engine

S_Harko uses a 3D Radix-2 Fast Fourier Transform to calculate maps ($O(N^3 \log N)$), accelerating calculations by over 300 times compared to direct summations.

### 4.2 Lorch Series Termination Filter

Truncation of the reciprocal space data at resolution limit $d_{\min}$ introduces Fourier series termination ripples (ghost oscillations around heavy atom vectors). S_Harko incorporates a tunable Lorch modification strength slider ($s \in [0, 1]$) applied per-reflection during accumulation:

$$
I(\mathbf{h})_{\text{modified}} = I(\mathbf{h}) \cdot \left[ (1 - s) + s \cdot \text{sinc}(\pi \cdot d^* \cdot d_{\min}) \right]
$$

where $d^* = \frac{1}{d(\mathbf{h})}$ is the reciprocal interplanar distance in $\text{\AA}^{-1}$, $d_{\min}$ is the dataset's minimum $d$-spacing, and $\text{sinc}(x) = \frac{\sin(x)}{x}$ (with $\text{sinc}(0) = 1$).

- **$s = 0.00$ (Off):** Preserves original measured intensities without filtering. Peak heights remain maximal, though termination ripples may persist.
- **$0 < s < 1.00$ (Tunable Blend):** Blends raw intensities with the Lorch envelope, allowing users to dial in just enough ripple suppression without causing excessive peak flattening.
- **$s = 1.00$ (Full Lorch):** Applies complete Lorch modification, effectively taming truncation ripples at the cost of slight peak broadening.

### 4.3 Grid Resolution & Anti-Aliasing

The FFT grid automatically expands to a power of two satisfying $N \ge 2 h_{\max} + 1$ to prevent high-index reflections from aliasing onto low-order data.

### 4.4 Model Gaussian Smearing

Model Patterson maps generated from trial atomic coordinates are convoluted with a Gaussian broadening kernel matching the resolution limit $d_{\min}$ and Debye-Waller temperature factor $B$.

### 4.5 Least-Squares Map Scaling

Trial calculated maps are scaled to observed data using least-squares scale fitting over non-origin voxels.

## 5. Peak Finding & Harker Analysis

### 5.1 3D Local Maxima Search

S_Harko identifies peak vectors by scanning non-masked voxels using a 26-neighbor 3D local maximum search with periodic boundary wrapping.

### 5.2 Harker Section Deconvolution

Peaks located on Harker sections are mapped to candidate fractional atomic coordinates $(x,y,z)$ using space-group-specific analytical expressions.

### 5.3 Multi-Section Site Combination

Each Harker section is solved independently, so the same real atom typically produces a slightly different-looking candidate position from every section it appears in &ndash; one partial solution per section, each only as good as the peak that section happened to find. The worker still solves every section this way, and every one of those partial, per-section positions is kept and shown to you (the "Harker Solutions" column in the Peaks tab, described below), because a partial solution that agrees with the final structure is a useful sanity check even when it wasn't itself used to build that structure.

What those per-section solutions do <em>not</em> currently do is get cross-referenced against each other to build the "Consolidated Sites" list. There is a tolerance-based combination routine in the worker that does exactly that &ndash; averaging together whichever per-section positions land within a chosen distance of one another &ndash; and the code path for it (a `COMBINE_ONLY` worker message, and the underlying `combineSites()` function) is still there, but the slider that used to drive it is hidden in the current interface and nothing in the UI calls it any more. Instead, every entry in Consolidated Sites today comes directly from peak-finding on the SMF or Buerger superposition map described next (5.4) &ndash; a single, more reliable source, rather than an agreement vote across several partial, section-by-section solutions. If you are comparing the Harker Solutions column against the Consolidated Sites column and wondering why a position in the first doesn't obviously map onto one in the second, this is why: the second column isn't built from the first.

### 5.4 Symmetry Minimum Function (SMF) & Buerger Superposition

Extracting atomic positions directly from a raw Patterson map is complicated by overlapping vectors and floating relative origins. S_Harko implements a dual-mode map reduction architecture: <strong>Symmetry Minimum Function (SMF)</strong> for absolute crystallographic positioning, and <strong>Buerger Minimum Superposition</strong> as a single-shift fallback.

#### A. Symmetry Minimum Function (SMF)

The SMF queries the Patterson map at vector displacements predicted by <em>all</em> non-identity symmetry operators $\mathbf{S}_i = (\mathbf{R}_i, \mathbf{t}_i)$ of the space group simultaneously. For any trial atomic position $\mathbf{x} = (x,y,z)$, the expected interatomic Harker vector generated by operator $i$ is $\mathbf{u}_i = \mathbf{x} - (\mathbf{R}_i \mathbf{x} + \mathbf{t}_i)$. The SMF value at $\mathbf{x}$ is the minimum Patterson intensity across all active symmetry operations:

$$
\text{SMF}(\mathbf{x}) = \min_{i = 1 \dots N_{\text{ops}}} P\Big( \mathbf{x} - (\mathbf{R}_i \mathbf{x} + \mathbf{t}_i) \Big)
$$

Because all symmetry operators are bound directly to the unit cell's origin, <strong>the SMF produces peaks directly in the absolute crystallographic frame</strong> (e.g., placing heavy atoms accurately on special positions such as $y=0.75$ in space group <em>Pnma</em>). Taking the multi-way minimum acts as a powerful logical "AND" filter, obliterating background noise and random vector overlaps while preserving genuine atomic sites.

#### B. Buerger Minimum Superposition Function

In space group $P1$ or when symmetry operators are unavailable, S_Harko falls back to the classical Buerger Minimum Function. The worker selects the strongest non-origin vector peak $\mathbf{u}_{\text{top}}$ clearing the origin exclusion radius ($d_{\min}$ or $1.1\text{ \AA}$) and shifts a copy of the Patterson map:

$$
M(\mathbf{u}) = \min\left( P(\mathbf{u}), \; P(\mathbf{u} - \mathbf{u}_{\text{top}}) \right)
$$

<em>Note:</em> Buerger superposition fixes the origin relative to the shifted atom (placing that atom at $(0,0,0)$), whereas SMF resolves positions within the true space group frame.

#### C. Unmasked Consolidated Peak Detection

Unlike the raw Patterson map (where the $(0,0,0)$ origin peak is masked to prevent scalar distortion), peak finding on the resulting SMF or Superposition map is performed <strong>without an origin mask</strong>. Because SMF/Superposition shifts valid atomic sites directly onto grid positions (which may include $(0,0,0)$ or special positions on mirror planes), unmasked peak detection ensures valid structural candidates are never accidentally deleted.

## 6. WebGPU Swarm Optimization

### 6.1 Particle Swarm Dynamics

S_Harko optimizes candidate atomic positions within the asymmetric unit using Particle Swarm Optimization (PSO) on the GPU. The position vector $\mathbf{X}_i$ and velocity vector $\mathbf{V}_i$ for particle $i$ update each generation according to:

$$
\mathbf{V}_i^{(t+1)} = w \mathbf{V}_i^{(t)} + c_1 r_1 \left( \mathbf{P}_{best, i} - \mathbf{X}_i^{(t)} \right) + c_2 r_2 \left( \mathbf{G}_{best} - \mathbf{X}_i^{(t)} \right)
$$

$$
\mathbf{X}_i^{(t+1)} = \text{frac}\left( \mathbf{X}_i^{(t)} + \mathbf{V}_i^{(t+1)} \right)
$$

### 6.2 Vector Fitness & Sigma Normalization

S_Harko evaluates trial structural candidate configurations by measuring how strongly the observed Patterson map supports the predicted interatomic vectors.

#### A. Raw Vector Fitness Calculation

For $N$ independent atoms expanded across all space group symmetry operations, the raw fitness $\text{Fitness}_{\text{raw}}$ computes the pair-weight-normalized mean Patterson intensity at all predicted vector positions $\mathbf{u}_{ij} = \mathbf{x}_i - \mathbf{x}_j$, minus steric collision penalties:

$$
\text{Fitness}_{\text{raw}} = \frac{1}{\sum_{i < j} 2 Z_i Z_j} \left( \sum_{i < j} 2 Z_i Z_j \cdot P(\mathbf{u}_{ij}) \;-\; \text{Penalty}_{\text{bump}} \right)
$$

where $P(\mathbf{u}_{ij})$ is trilinearly interpolated from the observed map. Normalizing by the constant sum of all pair weights ensures the search is not artificially biased toward structures sitting on special positions.

#### B. Sigma ($\sigma$) Scale Conversion

Raw map values depend entirely on the scale of the source file intensities. To provide a standard metric, S_Harko converts the raw fitness into standard deviation units ($\sigma$):

$$
\text{Fitness}_{\sigma} = \frac{\text{Fitness}_{\text{raw}}}{\sigma_{\text{map}}}
$$

where $\sigma_{\text{map}}$ is the standard deviation of non-origin voxels in the observed map. Quoting the fitness in $\sigma$ places the swarm convergence score on the exact same physical scale used by the 2D color bars, peak table height values, and PDF reports.

### 6.3 Multi-Restart PSO & GPU Hardware Limits

Multi-restart searches run fresh randomized swarms to escape local minima. Underneath that, S_Harko has to fit two very different things inside the fixed budget every GPU gives a compute shader, and it now measures your actual hardware for both rather than assuming a number that might be too small for a good GPU or too large for a modest one.

#### A. How many particles can run at once

Every particle (trial structure) needs its own slice of three large storage buffers holding the fractional coordinates of every atom in the asymmetric unit. S_Harko reads the browser's `maxStorageBufferBindingSize` and `maxComputeWorkgroupsPerDimension` limits the moment WebGPU becomes available, works out how many particles' worth of buffer space that leaves after reserving room for the map textures, and sets the <strong>Configurations</strong> slider's maximum accordingly &ndash; rounded down to a multiple of 64, never below 64, never above 8192. Because the buffers scale with the number of atoms in your asymmetric unit, adding another atom to the unit shrinks the particle ceiling; you can see the current number, and the raw binding-size and workgroup limits it was computed from, by hovering the <strong>?</strong> next to "WebGPU Global Optimization" in the Swarm section of the left panel.

#### B. How many symmetry-expanded atoms one particle can hold

Separately from the particle count, evaluating a <em>single</em> particle's fitness means generating every symmetry-equivalent copy of every asymmetric-unit atom &ndash; all of them, for every operator in the space group &ndash; and holding all of those positions in the compute shader's per-workgroup memory at once so every pair of them can be compared. That per-workgroup memory is a small, fixed pool (WebGPU guarantees at least 16&nbsp;KB per workgroup, and most desktop GPUs offer more), so there is a hard ceiling on <code>(atoms in the asymmetric unit) &times; (symmetry operators in the space group)</code> that any one particle can carry.

This used to be a single number wired into the shader by hand. It now measures the connected GPU's `maxComputeWorkgroupStorageSize` at startup, works out the largest atom&times;operator total that fits with headroom to spare, rounds that down to a whole multiple of the space group's operator count (so an atom's symmetry orbit is never split &ndash; carrying five of an atom's six symmetry images and dropping the sixth would silently bias that atom's contribution to the fitness), and compiles that number directly into the shader before the run starts. The floor is 384 combinations, which is generous for most structures &ndash; eight independent atoms in a 48-operator space group, or two atoms in one of the 192-operator cubic groups &ndash; and capable GPUs comfortably clear a thousand or more.

If your asymmetric unit and space group between them ask for more symmetry-expanded positions than even this device-sized limit allows, S_Harko does not silently drop the excess atoms and quietly hand back a fitness computed on an incomplete structure. It refuses to start the run at all and reports the exact numbers involved &ndash; how many atoms, how many operators, the total that implies, and the GPU's actual limit &ndash; so the fix (remove an atom, or accept that this particular structure needs a beefier GPU) is obvious rather than something you'd have to reverse-engineer from a suspiciously bad-looking result.

### 6.4 Minimum Contact Distance Filter & Zero Floor

S_Harko calculates pair contact cutoffs using user-defined distances or van der Waals radius sums. To allow independent atoms of the same element to dynamically merge onto special positions, **the absolute hard limit has been set to a zero floor ($0.00\text{ \AA}$)**. Both the CPU audit and the WGSL shader (`ABSOLUTE_MIN_CONTACT = 0.0`) allow close approaches when evaluating overlapping special positions, preventing artificial collision penalties from blocking structural convergence.

### 6.5 PSO Hyperparameter Tuning ($w, c_1, c_2$)

S_Harko exposes three core PSO control sliders in the UI, letting users adapt search behavior to noisy or rippled maps:

- **Inertia ($w$, default: $0.50$):** Controls how much previous velocity momentum a particle retains. <em>Tuning tip:</em> Increase $w$ (e.g., $0.70 - 0.80$) to help particles roll over high-frequency ripples without getting trapped; lower $w$ (e.g., $0.40$) for friction when fine-tuning smoothed peaks.
- **Cognitive Factor ($c_1$, default: $1.50$):** Dictates the particle's pull toward its own personal best-found position ($\mathbf{P}_{best}$). <em>Tuning tip:</em> Higher values encourage individual exploration of local neighborhoods.
- **Social Factor ($c_2$, default: $1.50$):** Dictates the particle's pull toward the swarm's global best position ($\mathbf{G}_{best}$). <em>Tuning tip:</em> Increase $c_2$ when working with smoothed or SMF maps to rapidly pull all particles toward the consensus solution.

## 7. UI & Controls Guided Tour

### Top-Right Quick Controls

Floating cleanly in the upper right corner, you'll find three subtle, lightweight icons:

- **Info Button (ⓘ):** Hover your mouse over this icon to see a small tooltip naming the program and the date it was built, without taking up any permanent space on the page.
- **Theme Toggle (☀️ / 🌙):** Instantly switches between Light Mode and Dark Mode. Your preference is saved locally, and all 3D cell frames and 2D map canvases dynamically adapt their color palettes.
- **Help Button (?):** Opens this documentation page in a fresh browser tab.

### 7.2 Left Controls Panel

The left panel handles data loading, structural setup, map calculation parameters, and the WebGPU swarm solver's settings. It used to be split into two tabs, but a tab hides everything except the one you're on, and several of these settings are ones you want to glance at together &ndash; the peak width $\sigma$ slider next to the temperature factor $B$ slider, say, since both broaden the calculated map and you're usually adjusting them in response to the same difference-map residual. So instead the panel is four <strong>collapsible sections</strong>, stacked top to bottom and each opened by default: <strong>1. Data Input</strong>, <strong>2. Structure</strong>, <strong>3. Patterson Map Synthesis</strong>, and <strong>4. Swarm Optimization</strong>. Click any section's title bar (the small triangle rotates to show open/closed) to fold it out of the way once you've got it set the way you want &ndash; there's no penalty for leaving all four open all the time, folding is purely to reclaim vertical space, and each section remembers its own open/closed state independently of the others.

#### 1. Data Input

- **Select Data File:** Click here to load your powder export file or generic peak list.
- **Wavelength (&#8491;):** The X-ray wavelength used to convert between $2\theta$ and $d$-spacing when the file needs that conversion. Defaults to Cu-K$\alpha$ ($1.54056$&nbsp;&#8491;); change it if your data were collected on a different source.
- **Divide out Lorentz&ndash;polarisation:** Checked by default. This only actually does anything when the file gives S_Harko no better option &ndash; if it supplies $|F_o|$ directly (Tier 1) or an explicit $Lp$ column (Tier 2), those take priority and this toggle has nothing left to correct; it matters for Tier 3 files that give you $2\theta$ but not $Lp$ itself, where S_Harko has to compute $Lp(2\theta)$ on your behalf. See 3.2 below for the full priority order.

#### 2. Structure

- **Select Space Group &raquo;:** Opens a searchable modal listing all 230 space groups; search by number or by symbol (e.g. <code>14</code> or <code>P 21/c</code>).
- **Unit cell parameters ($a, b, c, \alpha, \beta, \gamma$):** Populated automatically if your file's header carries them, and editable by hand otherwise or if you need to correct them.

#### 3. Patterson Map Synthesis

This is the section with the most controls in it, because it covers everything about turning your loaded reflections into a Patterson map: the FFT settings, the two broadening sliders that shape the <em>calculated</em> map for comparison against the observed one, the peak-search filters, and the button that actually runs the transform.

- **High resolution grid:** Off uses a $64^3$ voxel grid, on uses $128^3$ &ndash; eight times the voxels, memory, and time, for finer detail in the map. Because the FFT needs a power-of-two grid, these are the only two settings exposed; if your reflections reach an index so high that even $128^3$ would alias one reflection onto another, S_Harko silently steps the grid up further on its own and says so in the readout underneath the toggle, so you always know what grid a given map was actually calculated at, whichever of the two you asked for.
- **Lorch filter strength ($s \in [0, 1]$):** $s = 0$ leaves peak heights untouched, higher values (typically $0.10$&ndash;$0.25$) damp Fourier series termination ripples at the cost of slightly broader peaks; see 4.2 for the exact formula.
- **Peak width $\sigma$ ($\text{\AA}$):** The Gaussian broadening applied only to the <em>calculated</em> map, so it can be compared fairly with the resolution-limited observed one. S_Harko seeds it to $0.26 \cdot d_{\min}$ whenever a map is calculated, which is the value your data's own resolution implies; raise it if your sample is heavily thermally smeared, lower it to sharpen the model. It has no effect on the observed map and none on the swarm's fitness score.
- **Overall temperature factor $B$ ($\text{\AA}^2$):** A second, independent broadening on the calculated map representing thermal motion, combined in quadrature with the peak-width broadening above. Set it by hand, or turn on <strong>Fit B</strong> on the Swarm Results tab and let S_Harko scan for the value that best matches the observed map. Like peak width, this affects only the calculated and difference maps, never the observed map or the swarm fitness.
- **Max Peaks** and **Min $\sigma$ Threshold:** The two filters the peak search applies after the transform &ndash; keep at most this many peaks, and ignore anything weaker than this many standard deviations above the map's mean. Defaults are 50 peaks and $3.0\sigma$; raise Max Peaks or lower the threshold if you suspect a genuine site is being filtered out, tighten them if the peak tables are cluttered with noise.
- **Calculate Map:** Runs the FFT synthesis, the Lorch filter, peak finding, and the SMF or Buerger superposition, all in one background pass, using whatever the five settings above are set to at the moment you click it.

<div class="callout callout-warning">
<strong>&#9888; None of the five settings above recalculate anything by themselves.</strong> Nudging the Lorch slider, flipping the resolution toggle, or editing Max Peaks or Min $\sigma$ only records the new value and marks the map you're looking at as out of date &ndash; the status line names which setting changed and tells you to press Calculate Map, and the button itself gets a small amber ring so it's obvious at a glance that something needs re-running. This is deliberate: recalculating on every slider tick used to mean an accidental drag could kick off a full $128^3$ transform, and it made the four controls behave inconsistently with each other besides. The map already on screen is left alone until you actually press the button &ndash; it just no longer matches the controls, which is exactly what the warning is telling you.
</div>

#### 4. Swarm Optimization

- **Asymmetric Unit Contents:** Type an element symbol and a quantity, then click <strong>Add</strong>. Each entry becomes a chip showing the element, its quantity (&times;N), and its atomic number $Z$ &ndash; van der Waals radius isn't shown on the chip itself, but it's looked up automatically from the element symbol and used behind the scenes for the contact filter below. S_Harko expands every chip to its full set of symmetry-equivalent copies internally, so you only ever describe the asymmetric unit here, never the whole cell.
- **Contact Filter (minimum contact distance, $\text{\AA}$):** Penalises configurations where atoms sit closer together than this distance. The slider runs from $0.00$ to $3.00\,\text{\AA}$ and defaults to $0.00$; see 6.4 for why the floor is exactly zero rather than some small nonzero "safety margin".
- **Configurations, Iterations, Restarts:** The particle count (default 512, adjustable up to a device-dependent ceiling &ndash; see 6.3.A), the number of PSO steps per run (default 100), and the number of independent randomized restarts kept and compared (default 4).
- **Inertia ($w$), Cognitive ($c_1$), Social ($c_2$):** The three PSO tuning sliders, default $0.50$, $1.50$, and $1.50$ respectively; see 6.5 for tuning guidance.
- **Seed heaviest atom from Harker sites:** When enabled, the heaviest atom in the asymmetric unit starts on one of the consolidated candidate positions from peak analysis rather than at a random point, and is kept within one peak-width $\sigma$ of that seed rather than roaming the whole cell &ndash; the other atoms still start and search freely. Needs the peak analysis to have already run (i.e. a Consolidated Sites list to seed from).
- **Run Swarm / Stop:** Starts and, if needed, cancels the search. While a swarm is running, a progress bar and a small live readout (current generation, best fitness so far in $\sigma$ units, elapsed time and iterations/second) update in place beneath the buttons, and a line under that reports which GPU is doing the work.

<div class="callout callout-info">
<strong>While anything is calculating.</strong> The map transform and the swarm search both lock down the rest of the interface for their duration: loading a new file, adding an atom, starting a second swarm run, and every export button across both panels are all disabled until the current job finishes. This isn't just tidiness &ndash; loading new data or changing the asymmetric unit out from under a running GPU search would corrupt the buffers it's reading from, so the lock is there to make that impossible rather than merely unlikely. The one exception is the <strong>Stop</strong> button, which stays live throughout a swarm run for the obvious reason that it needs to be; everything else re-enables automatically the instant the run ends or is stopped.
</div>

### 7.3 Right Results Panel

The right main display gives you real-time visual feedback, split into three view modes:

#### 1. Maps View

Displays a synchronized <strong>2&times;2 multi-map grid</strong> comparing all four map modes at once:

- **Observed Map:** The experimental Patterson synthesis computed directly from your diffraction intensities.
- **Calculated Map:** The model Patterson map generated from the trial atomic structure found by the swarm.
- **Difference Map (Observed − Calculated):** Highlights missing scattering density or incorrectly placed atoms.
- **SMF (Symmetry Minimum Function):** A multi-way minimum superposition map that filters out background noise and places heavy atoms on their true crystallographic positions.

Use the <strong>Axis Select dropdown</strong> to switch your slicing plane (a&ndash;b, a&ndash;c, or b&ndash;c) and drag the <strong>Section Slider</strong> to slice through the unit cell volume. The <strong>Auto range</strong> checkbox controls how each pane's colour scale is set: left unchecked (the default), every pane uses one fixed scale spanning that map's minimum and maximum over the <em>whole</em> volume, so any two slices or sections are directly comparable &ndash; a faint slice really does look faint next to a strong one. Checking it instead rescales each pane to the minimum and maximum of the <em>current slice only</em>, which is useful when you're hunting for a weak feature in an otherwise quiet section, but be aware that a genuinely faint slice will then look just as vivid as a strong one, since both are being stretched to fill the same colour range.

#### 2. Peaks View

Organizes the peak extraction pipeline into three columns, left to right, so what's derived from what stays visible as a single left-to-right pipeline rather than three unrelated tables:

1. **Patterson Peaks:** Raw 3D local maxima vectors $(u,v,w)$ found in the Patterson map, ordered by peak height in $\sigma$.
2. **Harker Solutions:** Atomic coordinates $(x,y,z)$ deconvoluted independently from individual Harker sections &ndash; one candidate position per section, each only as reliable as the peak that section happened to contribute.
3. **Consolidated Sites:** The final candidate list, extracted directly from the SMF or Buerger superposition map (5.4) rather than by cross-referencing the Harker Solutions column against itself. A cross-referencing routine does exist in the code for exactly that purpose, but its control is currently hidden and nothing in the interface invokes it &ndash; see 5.3 for the full explanation. Practically, this means the Harker Solutions column is there for you to sanity-check against the final result, not because the final result was built from it.

#### 3. Swarm Results View

When you run the global optimization, this tab fills with three columns side by side, plus an editable table spanning the full width underneath them:

- **Fitness Evolution (left column):** A chart of the best structure score over the run, quoted in $\sigma$ map units.
- **Proposed Structure (middle column):** A real-time Three.js view of all symmetry-generated atoms inside the unit cell box. Left-click and drag to rotate, right-click to pan, and scroll to zoom. If you've hand-edited any coordinate in the table below, a small "edited" tag appears here to remind you that what you're looking at is no longer exactly what the swarm returned.
- **Atoms in Cell (right column):** Three running statistics &ndash; total atom count in the cell, the shortest interatomic contact distance found, and the cell mass. The theoretical density ($\text{g/cm}^3$) is calculated too, but it's shown as a small note beside the Asymmetric Unit table's own heading further down, not grouped in with these three.
- **Fit B (below the three columns):** When turned on, S_Harko scans the overall temperature factor $B$ to minimise the residual between the calculated and observed Patterson maps, and writes the result back to the $B$ slider in the Patterson Map Synthesis section. It runs immediately if a structure is already on screen, and again automatically at the end of every subsequent swarm run.
- **Asymmetric Unit (full width, below Fit B):** A live table of every independent atom. Editing any fractional coordinate by hand immediately regenerates every symmetry-equivalent copy, redraws the 3D structure, rechecks all contact distances, and changes exactly what Save CIF will write &ndash; there's no separate "apply" step. Values you type outside the $0$&ndash;$1$ range are wrapped back into the cell automatically rather than rejected.

Below the table sit three actions, left to right: <strong>Revert edits</strong> (discards any hand edits and restores exactly what the swarm returned), <strong>Save CIF</strong>, and <strong>PDF Report</strong> &ndash; see 7.4 for what each of the last two actually produces.

### 7.4 Exporting CIF, GRD, CSV & Summary Reports

S_Harko makes it easy to export your results for publication or further refinement:

- **Save CIF (Swarm Results tab, bottom action row):** Writes a standard Crystallographic Information File containing your unit cell, space group, and the solved atomic coordinates.
- **PDF Report (Swarm Results tab, same action row, immediately to the right of Save CIF):** Generates a full summary document with timestamped crystal parameters, map slices, the fitness chart, peak tables, and a 3D structure snapshot. It used to sit at the bottom of the left panel and enable as soon as a file loaded, which let you generate a report before anything had actually been solved; it now lives next to Save CIF and needs the same thing Save CIF does &ndash; an actual swarm result &ndash; before it will click.
- **Density (.grd) (Maps tab, bottom of the map controls row):** Exports all four Patterson map modes (observed, calculated, difference, SMF) as full 3D volume grid files ready to open directly in VESTA.
- **Peaks (.csv) (Peaks tab, top of the column header):** Exports the raw Patterson peak vector list.
- **Save CSV (Peaks tab, bottom of the Consolidated column):** A separate export of just the Consolidated Sites &ndash; the SMF/superposition-derived candidate positions, not the raw vector list above.

## 8. Step-by-Step Workflows

The three walkthroughs below cover the same ground at different levels of granularity: 8.1 and 8.2 together are the detailed, one-decision-at-a-time version of exactly what 8.3 compresses into five lines. If this is your first run, follow 8.1 and 8.2; once the workflow is familiar, 8.3 is enough of a reminder.

### 8.1 Deterministic Harker & SMF Analysis

Everything in this half of the workflow is deterministic &ndash; no swarm, no GPU, just the FFT and the peak search.

1. **Load Data:** Click <strong>Select Data File</strong> in the "1. Data Input" section of the left panel to load your <code>.txt</code>, <code>.hkl</code>, or <code>.dat</code> file. S_Harko will attempt to parse reflections and any available metadata headers.
2. **Define Structure:** In "2. Structure", click <strong>Select Space Group &raquo;</strong> to choose your symmetry, and verify the populated cell parameters ($a, b, c, \alpha, \beta, \gamma$).
3. **Calculate Map:** In "3. Patterson Map Synthesis", set the Lorch strength, grid resolution, and peak-search filters the way you want them, then click <strong>Calculate Map</strong> to generate the Patterson synthesis. If you change any of those settings afterward, remember to press Calculate Map again &ndash; they no longer trigger a recalculation on their own (see 7.2).
4. **Inspect the Maps:** Switch to the <strong>Maps</strong> tab on the right. Drag the <strong>Section Slider</strong> to inspect your Patterson map. Notice how the <strong>SMF</strong> pane cleans up random noise and highlights candidate heavy-atom positions compared to the raw Observed pane.
5. **Check the Candidate Peaks:** Click the <strong>Peaks</strong> tab on the right to view the Consolidated Sites column &ndash; the top candidate locations extracted from the SMF (or Buerger) map.

### 8.2 WebGPU Swarm Global Search

Picking up from the Consolidated Sites you just inspected, this half hands the search over to the GPU.

1. **Set Up the Swarm:** Open the "4. Swarm Optimization" section of the left panel. Enter your expected asymmetric unit contents (e.g. <code>Pb</code>&nbsp;&times;1, <code>Cl</code>&nbsp;&times;2). Turn on <strong>Seed heaviest atom from Harker sites</strong> to start the heavy atom on one of the candidate positions from step 8.1 instead of a random point.
2. **Run Global Optimization:** Click <strong>Run Swarm</strong>. Watch the fitness curve rise on the Swarm Results tab as WebGPU updates thousands of configurations per second; the generation counter, best fitness, and elapsed time update live beneath the Run Swarm / Stop buttons on the left while it runs.
3. **Refine &amp; Export:** Turn on <strong>Fit B</strong> to optimise the thermal parameter against the observed map. Inspect the solved model in the 3D viewer, nudge any coordinates by hand in the Asymmetric Unit table if needed, then click <strong>Save CIF</strong> and, if you want a full write-up of the run, <strong>PDF Report</strong> right next to it.

### 8.3 Combined Hybrid Resolution Workflow

1. Load your diffraction data file and click <strong>Calculate Map</strong>. S_Harko computes the FFT Patterson map, applies the optional Lorch filter, and constructs the <strong>Symmetry Minimum Function (SMF)</strong> map.
2. Inspect the SMF pane in the Maps tab.
3. Review the Consolidated Sites generated from the SMF map in the <strong>Peaks</strong> tab.
4. Open "4. Swarm Optimization" on the left, add your asymmetric unit atoms, optionally enable heavy-atom Harker seeding, tune $w$, $c_1$, $c_2$ if the defaults aren't converging well, and click <strong>Run Swarm</strong>.
5. Inspect the final model in the Swarm Results tab, and export your CIF file (and, if you'd like, the PDF report right beside it).

## 9. Troubleshooting & Technical FAQ

| Symptom / Question | Root Cause / Explanation | Resolution |
|--------------------|--------------------------|------------|
| Lorch filter does not change peak list coordinates | True interatomic vectors are robust physical features. Lorch modification tames ripples and broadens peaks, but does not displace true vector centers. | This is normal and expected. Use Lorch strength ($0.1 - 0.3$) if termination ripples create noise peaks in the peak table. |
| SMF map looks sparse/different from the raw Patterson map | SMF evaluates an $N$-way minimum across all space group operators. It acts as a logical "AND" filter, suppressing background noise and leaving only true atomic positions in the absolute crystallographic frame. | This is the expected behavior of SMF. Use the extracted SMF peaks as absolute candidate sites for seeding. |
| "WebGPU is not supported..." | WebGPU disabled or unsupported browser. | Use a Chromium v113+ browser with updated graphics drivers. |
| Atoms refusing to merge on special positions | Minimum contact distance filter set too high. | Lower the minimum contact distance slider down toward $0.00\text{ \AA}$ to permit overlapping symmetry-equivalent atoms. |
| "&hellip; generated positions, above this GPU's limit of &hellip;. Search for fewer atoms at a time." | Evaluating one particle means holding every symmetry-generated copy of every asymmetric-unit atom in the compute shader's per-workgroup memory at once, and that memory is a small, fixed pool on any GPU. The number of copies is (atoms you've entered) &times; (space group operators), and this message means that product is larger than what your specific GPU's workgroup memory can hold &ndash; S_Harko sizes this limit to your actual hardware rather than assuming a fixed number, but there is still a ceiling somewhere. See 6.3.B for the full explanation of why this is a hard refusal rather than a silent, partially-wrong result. | Remove an atom from the asymmetric unit if you can (the message tells you exactly how many positions were requested and what your GPU's actual limit is, so you can see how far over you are), or accept that this particular combination of structure size and space group needs a GPU with more workgroup storage than the one you're currently running on. |

S_Harko Documentation &middot;  NitaD, Univ Paris-Saclay, 07 August 2026.<br>Edited with AI code inspection.