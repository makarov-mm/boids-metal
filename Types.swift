import simd

// Mirror of the Metal `SimParams` struct. Every field is a 4-byte scalar so
// the memory layout matches the shader exactly with no padding surprises.
struct SimParams {
    var count: UInt32 = 12000
    var predatorCount: UInt32 = 3
    var obstacleOn: UInt32 = 0
    var dt: Float = 0.1

    var perception: Float = 9.5
    var sepDist: Float = 4.0
    var maxSpeed: Float = 16.0
    var minSpeed: Float = 5.0
    var maxForce: Float = 0.8

    var wSep: Float = 1.7
    var wAli: Float = 1.4
    var wCoh: Float = 1.15
    var wBounds: Float = 2.5
    var wFlee: Float = 4.0

    var bMinX: Float = 0,   bMinY: Float = 0,  bMinZ: Float = 0
    var bMaxX: Float = 140, bMaxY: Float = 90, bMaxZ: Float = 140
    var margin: Float = 12.0

    var predMaxSpeed: Float = 17.0
    var predMaxForce: Float = 0.7
    var fleeRadius: Float = 14.0

    var obsX: Float = 70, obsY: Float = 45, obsZ: Float = 70
    var obsRadius: Float = 14.0
    var wObstacle: Float = 3.5

    var boundsMin: SIMD3<Float> { SIMD3(bMinX, bMinY, bMinZ) }
    var boundsMax: SIMD3<Float> { SIMD3(bMaxX, bMaxY, bMaxZ) }
}

// Mirror of the Metal `RenderUniforms`. float4x4 + float4s are all 16-aligned,
// matching simd on the Swift side.
struct RenderUniforms {
    var viewProj: float4x4 = matrix_identity_float4x4
    var eye: SIMD4<Float> = .zero
    var misc: SIMD4<Float> = .zero    // x=boidSize y=maxSpeed z=time w=aspect
    var flags: SIMD4<Float> = .zero   // x=isPredator
}

// ---- matrix helpers (Metal-style, depth range 0..1) ----
func perspective(fovyRadians fovy: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
    let ys = 1 / tan(fovy * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return float4x4(columns: (
        SIMD4(xs, 0,  0,        0),
        SIMD4(0,  ys, 0,        0),
        SIMD4(0,  0,  zs,      -1),
        SIMD4(0,  0,  zs * near, 0)
    ))
}

func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
    let z = simd_normalize(eye - center)
    let x = simd_normalize(simd_cross(up, z))
    let y = simd_cross(z, x)
    return float4x4(columns: (
        SIMD4(x.x, y.x, z.x, 0),
        SIMD4(x.y, y.y, z.y, 0),
        SIMD4(x.z, y.z, z.z, 0),
        SIMD4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)
    ))
}
