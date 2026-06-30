import SwiftUI

// Shared, observable state. The renderer reads it each frame; the SwiftUI HUD
// and the controls panel observe / mutate it; the Metal view's input handlers
// mutate it on the main thread.
final class SimState: ObservableObject {
    @Published var paused = false
    @Published var substeps = 1
    @Published var predatorsOn = true
    @Published var obstacleOn = true        // solid spheres visible & draggable by default
    @Published var showHUD = true
    @Published var showControls = true

    @Published var fps: Double = 0
    @Published var boidCount: Int = 16000

    // ---- live-tunable simulation parameters (read every frame) ----
    @Published var sepWeight:  Float = 1.7
    @Published var aliWeight:  Float = 1.4
    @Published var cohWeight:  Float = 1.15
    @Published var perception: Float = 9.5
    @Published var maxSpeed:   Float = 16.0
    @Published var obstacleRadius: Float = 16.0

    // ---- live-tunable visual parameters ----
    @Published var trailDecay:    Float = 0.90   // 0 = no trails, →1 = long trails
    @Published var bloomStrength: Float = 0.95

    // ---- applied on reset (buffer count is fixed per run) ----
    @Published var desiredCount: Double = 16000

    // one-shot reset flag consumed by the renderer
    var resetRequested = false
}
