import SwiftUI
import MetalKit

// MTKView subclass that turns mouse/keyboard into camera + sim-state changes.
final class BoidsMTKView: MTKView {
    weak var renderer: Renderer?
    var state: SimState?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDragged(with event: NSEvent) {
        renderer?.camera.rotate(dx: Float(event.deltaX), dy: Float(event.deltaY))
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
