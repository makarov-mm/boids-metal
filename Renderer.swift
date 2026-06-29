import MetalKit
import simd

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let camera = OrbitCamera()
    let state: SimState

    // pipelines
    private var boidsPipeline: MTLComputePipelineState!
    private var predatorsPipeline: MTLComputePipelineState!
    private var boidRender: MTLRenderPipelineState!
    private var lineRender: MTLRenderPipelineState!
    private var brightPipeline: MTLRenderPipelineState!
    private var blurPipeline: MTLRenderPipelineState!
    private var compositePipeline: MTLRenderPipelineState!
    private var depthState: MTLDepthStencilState!
    private var sampler: MTLSamplerState!

    // ping-pong simulation buffers
    private var posBuf: [MTLBuffer] = []
    private var velBuf: [MTLBuffer] = []
    private var predPosBuf: [MTLBuffer] = []
    private var predVelBuf: [MTLBuffer] = []
    private var cur = 0
    private let predatorCapacity = 8

    private var lineBuf: MTLBuffer!
    private var lineCount = 0
    private var paramsBuf: MTLBuffer!
    private var uniformsBuf: MTLBuffer!

    // offscreen targets
    private var sceneColor: MTLTexture!
    private var sceneDepth: MTLTexture!
    private var bloomA: MTLTexture!
    private var bloomB: MTLTexture!

    private var params = SimParams()

    // timing
    private var lastTime = CFAbsoluteTimeGetCurrent()
    private var frameAccum = 0
    private var timeAccum: Double = 0
    private var simTime: Float = 0

    init(mtkView: MTKView, state: SimState) {
        self.device = mtkView.device!
        self.queue = device.makeCommandQueue()!
        self.state = state
        super.init()

        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .invalid   // we use our own offscreen depth
        mtkView.preferredFramesPerSecond = 120

        buildPipelines(view: mtkView)
        buildBuffers()
        seedSimulation()

        let c = (params.boundsMin + params.boundsMax) * 0.5
        let ext = params.boundsMax - params.boundsMin
        camera.target = c
        camera.distance = max(ext.x, max(ext.y, ext.z)) * 1.7
    }

    // MARK: setup

    private func buildPipelines(view: MTKView) {
        let lib = device.makeDefaultLibrary()!

        boidsPipeline     = try! device.makeComputePipelineState(function: lib.makeFunction(name: "boids_update")!)
        predatorsPipeline = try! device.makeComputePipelineState(function: lib.makeFunction(name: "predators_update")!)

        let hdr: MTLPixelFormat = .rgba16Float

        func makeRender(_ vs: String, _ fs: String, color: MTLPixelFormat, depth: Bool) -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: vs)
            d.fragmentFunction = lib.makeFunction(name: fs)
            d.colorAttachments[0].pixelFormat = color
            if depth { d.depthAttachmentPixelFormat = .depth32Float }
            return try! device.makeRenderPipelineState(descriptor: d)
        }

        boidRender        = makeRender("boid_vertex", "boid_fragment", color: hdr, depth: true)
        lineRender        = makeRender("line_vertex", "line_fragment", color: hdr, depth: true)
        brightPipeline    = makeRender("fs_vertex", "bright_pass", color: hdr, depth: false)
        blurPipeline      = makeRender("fs_vertex", "blur_pass", color: hdr, depth: false)
        compositePipeline = makeRender("fs_vertex", "composite", color: view.colorPixelFormat, depth: false)

        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .less
        ds.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: ds)

        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear; sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
        sampler = device.makeSamplerState(descriptor: sd)
    }

    private func buildBuffers() {
        let n = Int(params.count)
        let stride = MemoryLayout<SIMD3<Float>>.stride
        let opt: MTLResourceOptions = .storageModeShared

        for _ in 0..<2 {
            posBuf.append(device.makeBuffer(length: n * stride, options: opt)!)
            velBuf.append(device.makeBuffer(length: n * stride, options: opt)!)
            predPosBuf.append(device.makeBuffer(length: predatorCapacity * stride, options: opt)!)
            predVelBuf.append(device.makeBuffer(length: predatorCapacity * stride, options: opt)!)
        }

        // bounding-box edges (12 edges -> 24 vertices)
        let a = params.boundsMin, b = params.boundsMax
        let corner: [SIMD3<Float>] = [
            SIMD3(a.x,a.y,a.z), SIMD3(b.x,a.y,a.z), SIMD3(b.x,b.y,a.z), SIMD3(a.x,b.y,a.z),
            SIMD3(a.x,a.y,b.z), SIMD3(b.x,a.y,b.z), SIMD3(b.x,b.y,b.z), SIMD3(a.x,b.y,b.z)
        ]
        let edges = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
        var lines = edges.map { corner[$0] }
        lineCount = lines.count
        lineBuf = device.makeBuffer(bytes: &lines, length: lines.count * stride, options: opt)

        paramsBuf   = device.makeBuffer(length: MemoryLayout<SimParams>.stride, options: opt)
        uniformsBuf = device.makeBuffer(length: MemoryLayout<RenderUniforms>.stride, options: opt)
    }

    private func seedSimulation() {
        let n = Int(params.count)
        let c = (params.boundsMin + params.boundsMax) * 0.5
        let ext = params.boundsMax - params.boundsMin

        var rng = SystemRandomNumberGenerator()
        func rand(_ lo: Float, _ hi: Float) -> Float { Float.random(in: lo...hi, using: &rng) }

        let pos = posBuf[cur].contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        let vel = velBuf[cur].contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        for i in 0..<n {
            pos[i] = SIMD3(c.x + rand(-0.3, 0.3) * ext.x,
                           c.y + rand(-0.3, 0.3) * ext.y,
                           c.z + rand(-0.3, 0.3) * ext.z)
            var d = SIMD3<Float>(rand(-1,1), rand(-1,1), rand(-1,1))
            let l = simd_length(d); d = l > 1e-5 ? d / l : SIMD3(1,0,0)
            vel[i] = d * rand(params.minSpeed, params.maxSpeed)
        }

        let pp = predPosBuf[cur].contents().bindMemory(to: SIMD3<Float>.self, capacity: predatorCapacity)
        let pv = predVelBuf[cur].contents().bindMemory(to: SIMD3<Float>.self, capacity: predatorCapacity)
        for k in 0..<predatorCapacity {
            pp[k] = SIMD3(rand(params.bMinX, params.bMaxX),
                          rand(params.bMinY, params.bMaxY),
                          rand(params.bMinZ, params.bMaxZ))
            var d = SIMD3<Float>(rand(-1,1), rand(-1,1), rand(-1,1))
            let l = simd_length(d); d = l > 1e-5 ? d / l : SIMD3(1,0,0)
            pv[k] = d * (params.predMaxSpeed * 0.6)
        }
        simTime = 0
    }

    // MARK: resize

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspect = Float(size.width / max(1, size.height))
        let w = max(1, Int(size.width)), h = max(1, Int(size.height))

        sceneColor = makeTarget(.rgba16Float, w, h, read: true)
        sceneDepth = makeTarget(.depth32Float, w, h, read: false)
        bloomA = makeTarget(.rgba16Float, max(1, w/2), max(1, h/2), read: true)
        bloomB = makeTarget(.rgba16Float, max(1, w/2), max(1, h/2), read: true)
    }

    private func makeTarget(_ fmt: MTLPixelFormat, _ w: Int, _ h: Int, read: Bool) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
        d.usage = read ? [.renderTarget, .shaderRead] : [.renderTarget]
        d.storageMode = .private
        return device.makeTexture(descriptor: d)!
    }

    // MARK: frame

    func draw(in view: MTKView) {
        guard let sceneColor, let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }

        if state.resetRequested { state.resetRequested = false; cur = 0; seedSimulation() }

        // sync params from UI
        params.predatorCount = state.predatorsOn ? 3 : 0
        params.obstacleOn = state.obstacleOn ? 1 : 0
        memcpy(paramsBuf.contents(), &params, MemoryLayout<SimParams>.stride)

        let substeps = state.paused ? 0 : max(1, state.substeps)

        // ---- simulation ----
        if substeps > 0, let enc = cmd.makeComputeCommandEncoder() {
            let tile = 256
            let groups = MTLSize(width: (Int(params.count) + tile - 1) / tile, height: 1, depth: 1)
            let tpg = MTLSize(width: tile, height: 1, depth: 1)

            for _ in 0..<substeps {
                let src = cur, dst = 1 - cur

                enc.setComputePipelineState(boidsPipeline)
                enc.setBuffer(posBuf[src], offset: 0, index: 0)
                enc.setBuffer(velBuf[src], offset: 0, index: 1)
                enc.setBuffer(posBuf[dst], offset: 0, index: 2)
                enc.setBuffer(velBuf[dst], offset: 0, index: 3)
                enc.setBuffer(predPosBuf[src], offset: 0, index: 4)
                enc.setBuffer(paramsBuf, offset: 0, index: 8)
                enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)

                if params.predatorCount > 0 {
                    enc.setComputePipelineState(predatorsPipeline)
                    enc.setBuffer(posBuf[src], offset: 0, index: 0)
                    enc.setBuffer(predPosBuf[src], offset: 0, index: 4)
                    enc.setBuffer(predVelBuf[src], offset: 0, index: 5)
                    enc.setBuffer(predPosBuf[dst], offset: 0, index: 6)
                    enc.setBuffer(predVelBuf[dst], offset: 0, index: 7)
                    enc.setBuffer(paramsBuf, offset: 0, index: 8)
                    let pc = Int(params.predatorCount)
                    enc.dispatchThreads(MTLSize(width: pc, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: min(pc, 8), height: 1, depth: 1))
                }
                enc.memoryBarrier(scope: .buffers)
                cur = dst
            }
            enc.endEncoding()
        }

        // ---- render uniforms ----
        var u = RenderUniforms()
        u.viewProj = camera.viewProj()
        let e = camera.eye()
        u.eye = SIMD4(e, 1)
        u.misc = SIMD4(1.4, params.maxSpeed, simTime, camera.aspect)  // x=boidSize
        memcpy(uniformsBuf.contents(), &u, MemoryLayout<RenderUniforms>.stride)

        // ---- scene pass (HDR) ----
        let scenePass = MTLRenderPassDescriptor()
        scenePass.colorAttachments[0].texture = sceneColor
        scenePass.colorAttachments[0].loadAction = .clear
        scenePass.colorAttachments[0].storeAction = .store
        scenePass.colorAttachments[0].clearColor = MTLClearColor(red: 0.012, green: 0.016, blue: 0.03, alpha: 1)
        scenePass.depthAttachment.texture = sceneDepth
        scenePass.depthAttachment.loadAction = .clear
        scenePass.depthAttachment.clearDepth = 1.0
        scenePass.depthAttachment.storeAction = .dontCare

        if let r = cmd.makeRenderCommandEncoder(descriptor: scenePass) {
            r.setDepthStencilState(depthState)

            // bounding box
            r.setRenderPipelineState(lineRender)
            r.setVertexBuffer(lineBuf, offset: 0, index: 0)
            r.setVertexBuffer(uniformsBuf, offset: 0, index: 1)
            r.drawPrimitives(type: .line, vertexStart: 0, vertexCount: lineCount)

            // boids
            r.setRenderPipelineState(boidRender)
            r.setVertexBuffer(posBuf[cur], offset: 0, index: 0)
            r.setVertexBuffer(velBuf[cur], offset: 0, index: 1)
            r.setVertexBuffer(uniformsBuf, offset: 0, index: 2)
            r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3,
                             instanceCount: Int(params.count))

            // predators (separate uniforms with predator flag)
            if params.predatorCount > 0 {
                var up = u; up.flags = SIMD4(1, 0, 0, 0)
                let predU = device.makeBuffer(bytes: &up, length: MemoryLayout<RenderUniforms>.stride, options: .storageModeShared)!
                r.setVertexBuffer(predPosBuf[cur], offset: 0, index: 0)
                r.setVertexBuffer(predVelBuf[cur], offset: 0, index: 1)
                r.setVertexBuffer(predU, offset: 0, index: 2)
                r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3,
                                 instanceCount: Int(params.predatorCount))
            }
            r.endEncoding()
        }

        // ---- bloom: bright pass -> H blur -> V blur ----
        fullscreen(cmd, pipeline: brightPipeline, target: bloomA) { enc in
            enc.setFragmentTexture(self.sceneColor, index: 0)
            enc.setFragmentSamplerState(self.sampler, index: 0)
        }
        var offH = SIMD2<Float>(1.0 / Float(bloomA.width), 0)
        fullscreen(cmd, pipeline: blurPipeline, target: bloomB) { enc in
            enc.setFragmentTexture(self.bloomA, index: 0)
            enc.setFragmentSamplerState(self.sampler, index: 0)
            enc.setFragmentBytes(&offH, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }
        var offV = SIMD2<Float>(0, 1.0 / Float(bloomB.height))
        fullscreen(cmd, pipeline: blurPipeline, target: bloomA) { enc in
            enc.setFragmentTexture(self.bloomB, index: 0)
            enc.setFragmentSamplerState(self.sampler, index: 0)
            enc.setFragmentBytes(&offV, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }

        // ---- composite to drawable ----
        if let rp = view.currentRenderPassDescriptor,
           let r = cmd.makeRenderCommandEncoder(descriptor: rp) {
            r.setRenderPipelineState(compositePipeline)
            r.setFragmentTexture(sceneColor, index: 0)
            r.setFragmentTexture(bloomA, index: 1)
            r.setFragmentSamplerState(sampler, index: 0)
            r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            r.endEncoding()
        }

        cmd.present(drawable)
        cmd.commit()

        if substeps > 0 { simTime += params.dt * Float(substeps) }
        updateFPS()
    }

    private func fullscreen(_ cmd: MTLCommandBuffer, pipeline: MTLRenderPipelineState,
                            target: MTLTexture, _ setup: (MTLRenderCommandEncoder) -> Void) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setRenderPipelineState(pipeline)
        setup(enc)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    private func updateFPS() {
        let now = CFAbsoluteTimeGetCurrent()
        timeAccum += now - lastTime
        lastTime = now
        frameAccum += 1
        if timeAccum >= 0.5 {
            let fps = Double(frameAccum) / timeAccum
            frameAccum = 0; timeAccum = 0
            DispatchQueue.main.async { self.state.fps = fps }
        }
    }
}
