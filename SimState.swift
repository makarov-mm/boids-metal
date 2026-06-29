import SwiftUI

// Shared, observable state. The renderer reads it each frame; the SwiftUI HUD
// observes it; the Metal view's key handler mutates it on the main thread.
final class SimState: ObservableObject {
    @Published var paused = false
    @Published var substeps = 1
    @Published var predatorsOn = true
    @Published var obstacleOn = false
    @Published var showHUD = true

    @Published var fps: Double = 0
    @Published var boidCount: Int = 12000

    // one-shot reset flag consumed by the renderer
    var resetRequested = false
}
