import SwiftUI
import MetalKit
import simd

// MTKView subclass that turns mouse/keyboard into camera + sim-state changes.
// Left-drag orbits the camera, unless the press landed on an obstacle sphere —
// then the drag moves that sphere instead, in real time.
final class BoidsMTKView: MTKView {
    weak var renderer: Renderer?
    var state: SimState?
    private var draggingObstacle: Int? = nil

    override var acceptsFirstResponder: Bool { true }

    private func ndc(_ event: NSEvent) -> SIMD2<Float> {
        let p = convert(event.locationInWindow, from: nil)   // view coords, origin bottom-left
        let w = max(1, bounds.width), h = max(1, bounds.height)
        return SIMD2<Float>(Float(2 * p.x / w - 1), Float(2 * p.y / h - 1))
    }

    override func mouseDown(with event: NSEvent) {
        draggingObstacle = renderer?.pickObstacle(ndc: ndc(event))
    }

    override func mouseDragged(with event: NSEvent) {
        if let k = draggingObstacle {
            renderer?.dragObstacle(k, ndc: ndc(event))
        } else {
            renderer?.camera.rotate(dx: Float(event.deltaX), dy: Float(event.deltaY))
        }
    }

    override func mouseUp(with event: NSEvent) {
        draggingObstacle = nil
    }

    override func scrollWheel(with event: NSEvent) {
        renderer?.camera.zoom(Float(event.scrollingDeltaY) * 0.03)
    }

    override func keyDown(with event: NSEvent) {
        guard let s = state else { return }
        switch event.keyCode {
        case 49:  s.paused.toggle()                                   // space
        case 126: s.substeps = min(s.substeps + 1, 8)                 // up
        case 125: s.substeps = max(s.substeps - 1, 1)                 // down
        default:
            switch event.charactersIgnoringModifiers?.lowercased() {
            case "o": s.obstacleOn.toggle()
            case "p": s.predatorsOn.toggle()
            case "r": s.resetRequested = true
            case "h": s.showHUD.toggle()
            case "c": s.showControls.toggle()
            default: break
            }
        }
    }
}

struct MetalView: NSViewRepresentable {
    @ObservedObject var state: SimState

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> BoidsMTKView {
        let view = BoidsMTKView()
        view.device = MTLCreateSystemDefaultDevice()
        let renderer = Renderer(mtkView: view, state: state)
        view.delegate = renderer
        view.renderer = renderer
        view.state = state
        context.coordinator.renderer = renderer
        // give it focus so key events arrive
        DispatchQueue.main.async { view.window?.makeFirstResponder(view) }
        return view
    }

    func updateNSView(_ nsView: BoidsMTKView, context: Context) {}

    final class Coordinator {
        var renderer: Renderer?
    }
}
