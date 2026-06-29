import SwiftUI

@main
struct BoidsMetalApp: App {
    @StateObject private var state = SimState()

    var body: some Scene {
        WindowGroup("Boids 3D / Swarm Intelligence") {
            ZStack(alignment: .topLeading) {
                MetalView(state: state)
                    .ignoresSafeArea()
                if state.showHUD { HUD(state: state) }
            }
            .frame(minWidth: 900, minHeight: 600)
            .background(Color.black)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

struct HUD: View {
    @ObservedObject var state: SimState

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Boids 3D / Swarm Intelligence")
                .font(.system(size: 15, weight: .semibold, design: .monospaced))
            Text(String(format: "fps %.0f", state.fps))
            Text("\(state.boidCount) boids   \(state.predatorsOn ? 3 : 0) predators\(state.obstacleOn ? "   obstacle" : "")")
            Text("speed x\(state.substeps)\(state.paused ? "   [PAUSED]" : "")")
            Spacer().frame(height: 8)
            Group {
                Text("drag / wheel  orbit / zoom")
                Text("space    pause")
                Text("up / dn  sim speed")
                Text("O   obstacle on/off")
                Text("P   predators on/off")
                Text("R   reset    H  hide HUD")
            }
            .foregroundStyle(.white.opacity(0.7))
        }
        .font(.system(size: 13, design: .monospaced))
        .foregroundStyle(.white.opacity(0.92))
        .padding(12)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
        .padding(14)
        .allowsHitTesting(false)
    }
}
