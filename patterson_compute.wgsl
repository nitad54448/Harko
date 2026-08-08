// Direct Fourier synthesis of a 3D Patterson map:
//   P(u,v,w) = (1/V) * sum_over_reflections( I(h,k,l) * cos(2*pi*(h*u + k*v + l*w)) )
//
// This is embarrassingly parallel over voxels (each voxel's sum is
// independent of every other voxel), so one compute invocation handles one
// voxel and loops over the full reflection list - the same loop structure
// as the original CPU version, just spread across the GPU instead of a
// single JS thread.

struct Params {
    mapRes: f32,
    numReflections: f32,
    volume: f32,
    numVoxels: f32,
};

@group(0) @binding(0) var<storage, read> reflH: array<f32>;
@group(0) @binding(1) var<storage, read> reflK: array<f32>;
@group(0) @binding(2) var<storage, read> reflL: array<f32>;
@group(0) @binding(3) var<storage, read> reflI: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputMap: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const PI2: f32 = 6.283185307179586;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // dispatchWorkgroups rounds up to a multiple of 64, so guard against
    // invocations beyond the actual voxel count.
    if (f32(idx) >= params.numVoxels) {
        return;
    }

    let res = u32(params.mapRes);
    let iu = idx % res;
    let iv = (idx / res) % res;
    let iw = idx / (res * res);

    let u = f32(iu) / params.mapRes;
    let v = f32(iv) / params.mapRes;
    let w = f32(iw) / params.mapRes;

    var p: f32 = 0.0;
    let n = u32(params.numReflections);
    for (var r: u32 = 0u; r < n; r = r + 1u) {
        let h = reflH[r];
        let k = reflK[r];
        let l = reflL[r];
        let inten = reflI[r];
        p = p + inten * cos(PI2 * (h * u + k * v + l * w));
    }

    outputMap[idx] = p / params.volume;
}
