// Patterson-vector fitness for one PSO particle.
//
// LAYOUT: one WORKGROUP per particle, not one invocation per particle.
//
// The previous version gave every invocation six private array<f32, 256>
// scratch arrays - about 5.4 KB per thread, 345 KB per 64-thread workgroup.
// No GPU has anything like that many registers, so all six arrays spilled to
// scratch memory and every access to a generated atom became a round trip to
// device memory. Occupancy collapsed at the same time, because the driver
// could keep almost no workgroups resident.
//
// Moving the generated atoms into workgroup storage fixes both problems at
// once: the arrays are allocated once per particle instead of once per thread,
// they live in fast on-chip memory, and the O(n^2) pair loop that dominates
// the cost gets spread across all 64 threads instead of running serially in
// one. Storage is ~6.7 KB per workgroup, comfortably inside the 16 KB that
// WebGPU guarantees.

struct Params {
    mapRes: f32,
    numAtoms: f32,
    numSymOps: f32,
    numParticles: f32,
    maxMapVal: f32, // largest |value| anywhere in the observed map, for scaling the anti-bump penalty
    normW: f32,     // constant sum of pair weights, see the note at the end of main()
    minContact: f32,// fixed contact distance in Angstrom; <= 0 means use the radius rule
    pad2: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<f32>;
@group(0) @binding(1) var<storage, read> zArray: array<f32>;
@group(0) @binding(2) var<storage, read> rArray: array<f32>;
@group(0) @binding(3) var<storage, read> symR: array<f32>;
@group(0) @binding(4) var<storage, read> symT: array<f32>;
@group(0) @binding(5) var<storage, read> obsMap: array<f32>;
@group(0) @binding(6) var<storage, read_write> fitnesses: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read> orthMat: array<f32>;

const WG: u32 = 64u;

// Separate f32 arrays rather than array<vec3<f32>>: vec3 is padded to a
// 16-byte stride in workgroup storage, which would waste a third of the budget.
//
// This value is INJECTED by the host at shader-compile time (see
// applyGpuParticleLimit / the __MAX_GEN_ATOMS__ replacement in Harko.html): it
// is sized to the device's maxComputeWorkgroupStorageSize so capable GPUs can
// track more symmetry-expanded atoms than the conservative floor. The literal
// below is only the standalone fallback used when the file is compiled without
// injection - it must stay a plain u32 literal so the array declarations that
// follow remain compile-time sized. 384 covers 8 atoms in a 48-operator group,
// or 2 in a 192-operator cubic one, and fits the 16 KB WebGPU guarantees.
const MAX_GEN_ATOMS: u32 = 384u; //__MAX_GEN_ATOMS__

var<workgroup> gx: array<f32, MAX_GEN_ATOMS>;
var<workgroup> gy: array<f32, MAX_GEN_ATOMS>;
var<workgroup> gz: array<f32, MAX_GEN_ATOMS>;
var<workgroup> gActive: array<u32, MAX_GEN_ATOMS>;
var<workgroup> partFit: array<f32, WG>;
var<workgroup> partPen: array<f32, WG>;

const MIN_BUMP_DIST_FACTOR: f32 = 0.6;

// Coincidence tolerance for symmetry images of ONE independent atom sitting on
// a special position. Those images are separated by ~0 by construction, so the
// tolerance only has to absorb f32 rounding; 0.2 was far looser than needed.
const SAME_SITE_TOL: f32 = 0.05;

// AFTER
// Hard limit removed to allow particles to merge dynamically on special positions.
const ABSOLUTE_MIN_CONTACT: f32 = 0.0;

// The largest a single vector's fitness contribution can ever be for a given
// pair is Z_i * Z_j * maxMapVal (the biggest observed peak). A flat penalty
// constant was found to be too weak for heavy atoms (e.g. Pb, Z=82) or
// unnormalized/large-intensity data: a single overlapping heavy-atom pair
// could out-score the fixed penalty by exploiting one large peak, letting
// the swarm "cheat" by collapsing atoms together instead of fitting the
// whole vector set. Scaling the penalty by this pair's own Z_i*Z_j*maxMapVal
// (with a safety margin) guarantees any clash is always a net loss,
// regardless of atomic number or how the input intensities are scaled.
// MIN_PENALTY_FLOOR is a fallback for the degenerate case of maxMapVal ~ 0.
const PENALTY_SAFETY_FACTOR: f32 = 10.0;
const MIN_PENALTY_FLOOR: f32 = 10000.0;

fn getCartesianDist(p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    var d = p1 - p2;
    d = d - round(d);

    let cx = orthMat[0]*d.x + orthMat[1]*d.y + orthMat[2]*d.z;
    let cy = orthMat[3]*d.x + orthMat[4]*d.y + orthMat[5]*d.z;
    let cz = orthMat[6]*d.x + orthMat[7]*d.y + orthMat[8]*d.z;

    return sqrt(cx*cx + cy*cy + cz*cz);
}

// Trilinear interpolation of the observed map.
//
// Nearest-neighbour sampling - u32(round(u * mapRes)) - made the fitness a
// piecewise-constant staircase: moving a particle anywhere inside a voxel
// changed nothing at all, and crossing a boundary changed everything. The
// gradient the swarm needs to follow was therefore zero almost everywhere and
// undefined on the boundaries, so particles could only find peaks by landing
// on them. Interpolating restores a continuous fitness surface.
fn sampleMap(uvw: vec3<f32>) -> f32 {
    let res = u32(params.mapRes);
    let f = uvw * params.mapRes;
    let fl = floor(f);
    let t = f - fl;

    let i0 = vec3<u32>(fl) % vec3<u32>(res);
    let i1 = (i0 + vec3<u32>(1u)) % vec3<u32>(res);

    let rowY0 = i0.y * res;
    let rowY1 = i1.y * res;
    let planeZ0 = i0.z * res * res;
    let planeZ1 = i1.z * res * res;

    let c000 = obsMap[planeZ0 + rowY0 + i0.x];
    let c100 = obsMap[planeZ0 + rowY0 + i1.x];
    let c010 = obsMap[planeZ0 + rowY1 + i0.x];
    let c110 = obsMap[planeZ0 + rowY1 + i1.x];
    let c001 = obsMap[planeZ1 + rowY0 + i0.x];
    let c101 = obsMap[planeZ1 + rowY0 + i1.x];
    let c011 = obsMap[planeZ1 + rowY1 + i0.x];
    let c111 = obsMap[planeZ1 + rowY1 + i1.x];

    let a = mix(mix(c000, c100, t.x), mix(c010, c110, t.x), t.y);
    let b = mix(mix(c001, c101, t.x), mix(c011, c111, t.x), t.y);
    return mix(a, b, t.z);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgId: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {

    let pIdx = wgId.x;

    // pIdx is workgroup-uniform, so the whole workgroup takes this branch
    // together and the barriers below stay in uniform control flow.
    if (pIdx >= u32(params.numParticles)) {
        return;
    }

    let totalAtoms = u32(params.numAtoms);
    let numSymOps  = u32(params.numSymOps);
    let totalGen   = min(totalAtoms * numSymOps, MAX_GEN_ATOMS);

    // --- 1. Generate symmetry equivalents, strided across the workgroup ---
    // Generated atom g corresponds to base atom g/numSymOps under operator
    // g%numSymOps, so Z and radius are always recoverable from the index.
    // Storing copies of them, as the old version did, cost two more 256-entry
    // scratch arrays for information that was already there.
    for (var g = lid; g < totalGen; g = g + WG) {
        let a = g / numSymOps;
        let s = g % numSymOps;

        let base = pIdx * totalAtoms * 3u + a * 3u;
        let p = vec3<f32>(particles[base], particles[base + 1u], particles[base + 2u]);

        var n = vec3<f32>(
            p.x * symR[s*9u + 0u] + p.y * symR[s*9u + 1u] + p.z * symR[s*9u + 2u] + symT[s*3u + 0u],
            p.x * symR[s*9u + 3u] + p.y * symR[s*9u + 4u] + p.z * symR[s*9u + 5u] + symT[s*3u + 1u],
            p.x * symR[s*9u + 6u] + p.y * symR[s*9u + 7u] + p.z * symR[s*9u + 8u] + symT[s*3u + 2u]
        );
        n = n - floor(n);   // normalize to [0, 1)

        gx[g] = n.x; gy[g] = n.y; gz[g] = n.z;
        gActive[g] = 1u;
    }
    workgroupBarrier();

    // --- 2. Special-position collapse ---
    // An atom on a special position is mapped onto itself by part of the
    // group, so several of its "equivalents" are the same atom and must not
    // be counted twice. Each thread owns a set of j and scans every i < j;
    // this reads positions only and writes gActive only, so there is no race.
    //
    // The old serial version skipped i that had themselves been deactivated.
    // Doing so here would make the result depend on evaluation order, which
    // is not defined across threads. Scanning all i < j instead is equivalent
    // whenever coincidence is transitive, which it is for genuine special
    // positions, where the coincident images are separated by ~0 rather than
    // by anything near the 0.2 A tolerance.
    for (var j = lid; j < totalGen; j = j + WG) {
        let aj = j / numSymOps;
        let pj = vec3<f32>(gx[j], gy[j], gz[j]);
        var dup = false;
        for (var i = 0u; i < j; i = i + 1u) {
            let ai = i / numSymOps;
            // Only images of the SAME independent atom may be collapsed.
            //
            // This used to compare atomic numbers instead, which opened a hole:
            // two DIFFERENT independent atoms of the same element that drifted
            // within the tolerance were treated as one atom and deactivated, so
            // the pair was never charged the anti-bump penalty. The swarm could
            // therefore park two independent heavy atoms 0.2 A apart - closer
            // than any real interatomic distance - and pay nothing for it.
            // Distinct atoms occupying the same site is never valid; it is a
            // clash, and it must be penalised rather than explained away.
            if (ai != aj) { continue; }
            if (getCartesianDist(vec3<f32>(gx[i], gy[i], gz[i]), pj) < SAME_SITE_TOL) {
                dup = true;
                break;
            }
        }
        if (dup) { gActive[j] = 0u; }
    }
    workgroupBarrier();

    // --- 3. Patterson vectors and anti-bump, split across the workgroup ---
    // Each unordered pair is visited once. The Patterson function is
    // centrosymmetric, P(u) == P(-u), so the (j,i) vector contributes exactly
    // what the (i,j) vector does and the doubling below reproduces the full
    // i != j sum at half the work. The penalty is counted once per clashing
    // pair, which is what was intended - the old i != j loop charged it twice.
    //
    // Striding i across threads balances the load: thread lid takes
    // i = lid, lid+64, lid+128..., so every thread gets a mix of long and
    // short inner loops.
    var fit: f32 = 0.0;
    var pen: f32 = 0.0;

    for (var i = lid; i < totalGen; i = i + WG) {
        if (gActive[i] == 0u) { continue; }
        let ai = i / numSymOps;
        let zi = zArray[ai];
        let ri = rArray[ai];
        let pi = vec3<f32>(gx[i], gy[i], gz[i]);

        for (var j = i + 1u; j < totalGen; j = j + 1u) {
            if (gActive[j] == 0u) { continue; }
            let aj = j / numSymOps;
            let zj = zArray[aj];
            let rj = rArray[aj];
            let pj = vec3<f32>(gx[j], gy[j], gz[j]);

            // A) Anti-bump penalty.
            //
            // The cut-off is either a fixed distance the user set, or the old
            // radius-sum rule. The radius rule is the better default because it
            // scales the exclusion with the atoms involved, but it is derived
            // from tabulated covalent radii that are wrong or missing for some
            // elements, so a fixed override has to be available.
            //
            // Note this runs over the SYMMETRY-GENERATED atoms and uses the
            // minimum-image convention inside getCartesianDist, so contacts
            // across cell boundaries, and between an atom and its own symmetry
            // mates, are caught as well as contacts within the asymmetric unit.
            let rule = select((ri + rj) * MIN_BUMP_DIST_FACTOR,
                              params.minContact,
                              params.minContact > 0.0);
            // Never let the rule fall below the absolute floor: tabulated radii
            // are missing or wrong for some elements, and a user-entered value
            // can be anything at all.
            let cutoff = max(rule, ABSOLUTE_MIN_CONTACT);
            if (getCartesianDist(pi, pj) < cutoff) {
                pen = pen + max(MIN_PENALTY_FLOOR, PENALTY_SAFETY_FACTOR * zi * zj * params.maxMapVal);
            }

            // B) Patterson vector u = xi - xj, wrapped into [0, 1)
            var uvw = pi - pj;
            uvw = uvw - floor(uvw);

            // C) Sample the observed map. The wrap above plus the modulo
            // inside sampleMap() make an out-of-range index impossible, so
            // the old `if (mapIdx < gridSize)` guard - and the phantom
            // penalty in its else branch - were dead code.
            fit = fit + 2.0 * zi * zj * sampleMap(uvw);
        }
    }

    // --- 4. Reduce the per-thread partial sums ---
    partFit[lid] = fit;
    partPen[lid] = pen;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            partFit[lid] = partFit[lid] + partFit[lid + stride];
            partPen[lid] = partPen[lid] + partPen[lid + stride];
        }
        workgroupBarrier();
    }

    // Normalise by a CONSTANT weight sum computed on the CPU from the full
    // uncollapsed generated set.
    //
    // The raw sum was uninterpretable: its magnitude is the product of the
    // atomic numbers, the number of symmetry-generated pairs and whatever
    // arbitrary scale the input intensities happened to carry, so a "best
    // fitness" of 4.7e8 said nothing about how good the fit was and could not
    // be compared between two datasets, or even between two atom lists on the
    // same data. Dividing by the total pair weight turns it into the mean
    // observed map value at the vectors the trial structure predicts - the
    // same units as the map itself, and directly comparable.
    //
    // The divisor is constant rather than the per-particle weight sum
    // precisely so this stays a pure rescaling: dividing by a quantity that
    // varies with the special-position collapse would have changed the
    // ranking, quietly rewarding structures whose atoms sit on special
    // positions for having fewer vectors to satisfy.
    if (lid == 0u) {
        let denom = max(params.normW, 1e-6);
        fitnesses[pIdx] = (partFit[0] - partPen[0]) / denom;
    }
}
