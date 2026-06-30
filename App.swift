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
                if state.showControls {
                    Controls(state: state)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
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
            Text("\(state.boidCount) boids   \(state.predatorsOn ? 3 : 0) predators\(state.obstacleOn ? "   3 obstacles" : "")")
            Text("speed x\(state.substeps)\(state.paused ? "   [PAUSED]" : "")")
            Spacer().frame(height: 8)
            Group {
                Text("drag / wheel  orbit / zoom")
                Text("drag a sphere   move obstacle")
                Text("space    pause")
                Text("up / dn  sim speed")
                Text("O   obstacles on/off")
                Text("P   predators on/off")
                Text("C   controls    H  hide HUD")
                Text("R   reset")
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

// Live parameter panel. Hit-testing is ON here so the sliders receive events;
// everything outside this panel's frame falls through to the Metal view.
struct Controls: View {
    @ObservedObject var state: SimState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("parameters")
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.92))

            Group {
                Text("flocking").sectionLabel()
                LabeledSlider("separation", $state.sepWeight, 0...4)
                LabeledSlider("alignment",  $state.aliWeight, 0...4)
                LabeledSlider("cohesion",   $state.cohWeight, 0...4)
                LabeledSlider("perception", $state.perception, 3...20)
                LabeledSlider("max speed",  $state.maxSpeed, 4...30)
            }
            Divider().overlay(Color.white.opacity(0.15))
            Group {
                Text("obstacles").sectionLabel()
                LabeledSlider("radius", $state.obstacleRadius, 6...40)
            }
            Divider().overlay(Color.white.opacity(0.15))
            Group {
                Text("visual").sectionLabel()
                LabeledSlider("trail decay",    $state.trailDecay, 0...0.985)
                LabeledSlider("bloom strength", $state.bloomStrength, 0...2)
            }
            Divider().overlay(Color.white.opacity(0.15))
            Group {
                Text("count  (R to apply)").sectionLabel()
                Slider(value: $state.desiredCount, in: 1000...40000, step: 1000)
                Text("\(Int(state.desiredCount)) boids")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
        .frame(width: 240)
        .padding(12)
        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        .padding(14)
    }
}

private struct LabeledSlider: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>

    init(_ label: String, _ value: Binding<Float>, _ range: ClosedRange<Float>) {
        self.label = label; self._value = value; self.range = range
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: "%.2f", value))
            }
            .font(.system(size: 11, design: .monospaced))
            .foregroundStyle(.white.opacity(0.82))
            Slider(value: $value, in: range)
                .controlSize(.mini)
        }
    }
}

private extension Text {
    func sectionLabel() -> some View {
        self.font(.system(size: 11, weight: .semibold, design: .monospaced))
            .foregroundStyle(.white.opacity(0.55))
    }
}
