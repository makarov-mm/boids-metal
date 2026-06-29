import simd

// Spherical orbit camera around a target point.
final class OrbitCamera {
    var target: SIMD3<Float> = .zero
    var distance: Float = 240
    var yaw: Float = 0.7
    var pitch: Float = 0.35
    var aspect: Float = 16.0 / 10.0
    var fovy: Float = 50 * .pi / 180

    func rotate(dx: Float, dy: Float) {
        yaw   -= dx * 0.005
        pitch -= dy * 0.005
        let lim = Float.pi / 2 - 0.05
        pitch = max(-lim, min(lim, pitch))
    }

    func zoom(_ delta: Float) {
        distance *= (1 - delta * 0.1)
        distance = max(20, min(900, distance))
    }

    func eye() -> SIMD3<Float> {
        let cp = cos(pitch), sp = sin(pitch)
        let cy = cos(yaw),   sy = sin(yaw)
        let dir = SIMD3<Float>(cp * sy, sp, cp * cy)
        return target + dir * distance
    }

    func viewProj() -> float4x4 {
        let e = eye()
        let proj = perspective(fovyRadians: fovy, aspect: aspect, near: 1, far: 1600)
        let view = lookAt(eye: e, center: target, up: SIMD3(0, 1, 0))
        return proj * view
    }
}
