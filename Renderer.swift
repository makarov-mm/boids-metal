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
    private var obstacleRender: MTLRenderPipelineState!
    private var lineRender: MTLRenderPipelineState!
    private var brightPipeline: MTLRenderPipelineState!
    private var trailPipeline: MTLRenderPipelineState!
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
    private let maxBoids = 40000          // buffer capacity; live count ≤ this

    private var lineBuf: MTLBuffer!
    private var lineCount = 0
    private var paramsBuf: MTLBuffer!
    private var uniformsBuf: MTLBuffer!

    // obstacles (xyz = centre, w = radius). Mutated on the main thread by the
    // drag handler; uploaded to the GPU each frame.
    private let obstacleCapacity = 8
    private var obstacles: [SIMD4<Float>] = []
    private var obstacleBuf: MTLBuffer!
    private var obstacleActive = false

    // obstacle sphere mesh
    private var sphereVtxBuf: MTLBuffer!
    private var sphereIdxBuf: MTLBuffer!
    private var sphereIdxCount = 0

    // offscreen targets
    private var sceneColor: MTLTexture!
    private var sceneDepth: MTLTexture!
    private var brightTex: MTLTexture!
    private var trailA: MTLTexture!
    private var trailB: MTLTexture!
    private var blurTmp: MTLTexture!
    private var bloomTex: MTLTexture!
    private var trailCur = 0
    private var needsTrailClear = true

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

        seedObstacles()
        buildPipelines(view: mtkView)
        buildBuffers()
        seedSimulation()

        let c = (params.boundsMin + params.boundsMax) * 0.5
        let ext = params.boundsMax - params.boundsMin
        camera.target = c
        camera.distance = max(ext.x, max(ext.y, ext.z)) * 1.7
    }

    // MARK: setup

    private func seedObstacles() {
        let r = params.obsRadius
        obstacles = [
            SIMD4( 70,  72, 110, r),
            SIMD4(152,  52,  92, r),
            SIMD4(110,  98, 156, r),
        ]
    }

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
        obstacleRender    = makeRender("obstacle_vertex", "obstacle_fragment", color: hdr, depth: true)
        lineRender        = makeRender("line_vertex", "line_fragment", color: hdr, depth: true)
        brightPipeline    = makeRender("fs_vertex", "bright_pass", color: hdr, depth: false)
        trailPipeline     = makeRender("fs_vertex", "trail_pass", color: hdr, depth: false)
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
        let stride = MemoryLayout<SIMD3<Float>>.stride
        let opt: MTLResourceOptions = .storageModeShared

        for _ in 0..<2 {
            posBuf.append(device.makeBuffer(length: maxBoids * stride, options: opt)!)
            velBuf.append(device.makeBuffer(length: maxBoids * stride, options: opt)!)
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

        // obstacle instance buffer + sphere mesh
        obstacleBuf = device.makeBuffer(length: obstacleCapacity * MemoryLayout<SIMD4<Float>>.stride, options: opt)
        buildSphereMesh(stacks: 18, slices: 28)

        paramsBuf   = device.makeBuffer(length: MemoryLayout<SimParams>.stride, options: opt)
        uniformsBuf = device.makeBuffer(length: MemoryLayout<RenderUniforms>.stride, options: opt)
    }

    // UV sphere; unit positions double as normals in the shader.
    private func buildSphereMesh(stacks: Int, slices: Int) {
        var verts: [SIMD3<Float>] = []
        verts.reserveCapacity((stacks + 1) * (slices + 1))
        for i in 0...stacks {
            let phi = Float.pi * Float(i) / Float(stacks)        // 0..π
            let y = cos(phi), r = sin(phi)
            for j in 0...slices {
                let th = 2 * Float.pi * Float(j) / Float(slices)
                verts.append(SIMD3(r * cos(th), y, r * sin(th)))
            }
        }
        var idx: [UInt16] = []
        let row = slices + 1
        for i in 0..<stacks {
            for j in 0..<slices {
                let a = UInt16(i * row + j)
                let b = UInt16((i + 1) * row + j)
                let c = UInt16((i + 1) * row + j + 1)
                let d = UInt16(i * row + j + 1)
                idx += [a, b, d, b, c, d]
            }
        }
        sphereIdxCount = idx.count
        sphereVtxBuf = device.makeBuffer(bytes: &verts, length: verts.count * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        sphereIdxBuf = device.makeBuffer(bytes: &idx, length: idx.count * MemoryLayout<UInt16>.stride, options: .storageModeShared)
    }

    private func seedSimulation() {
        params.count = UInt32(max(100, min(maxBoids, Int(state.desiredCount))))
        DispatchQueue.main.async { self.state.boidCount = Int(self.params.count) }

        let n = Int(params.count)
        let c = (params.boundsMin + params.boundsMax) * 0.5
        let ext = params.boundsMax - params.boundsMin

        var rng = SystemRandomNumberGenerator()
        func rand(_ lo: Float, _ hi: Float) -> Float { Float.random(in: lo...hi, using: &rng) }

        let pos = posBuf[cur].contents().bindMemory(to: SIMD3<Float>.self, capacity: maxBoids)
        let vel = velBuf[cur].contents().bindMemory(to: SIMD3<Float>.self, capacity: maxBoids)
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
        needsTrailClear = true
    }

    // MARK: resize

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspect = Float(size.width / max(1, size.height))
        let w = max(1, Int(size.width)), h = max(1, Int(size.height))
        let hw = max(1, w/2), hh = max(1, h/2)

        sceneColor = makeTarget(.rgba16Float, w, h, read: true)
        sceneDepth = makeTarget(.depth32Float, w, h, read: false)
        brightTex  = makeTarget(.rgba16Float, hw, hh, read: true)
        trailA     = makeTarget(.rgba16Float, hw, hh, read: true)
        trailB     = makeTarget(.rgba16Float, hw, hh, read: true)
        blurTmp    = makeTarget(.rgba16Float, hw, hh, read: true)
        bloomTex   = makeTarget(.rgba16Float, hw, hh, read: true)
        needsTrailClear = true
    }

    private func makeTarget(_ fmt: MTLPixelFormat, _ w: Int, _ h: Int, read: Bool) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
        d.usage = read ? [.renderTarget, .shaderRead] : [.renderTarget]
        d.storageMode = .private
        return device.makeTexture(descriptor: d)!
    }

    // MARK: mouse picking / dragging of obstacles

    // Build a world-space ray from normalised device coords (-1..1, y up).
    private func worldRay(ndc: SIMD2<Float>) -> (o: SIMD3<Float>, d: SIMD3<Float>) {
        let inv = camera.viewProj().inverse
        let near4 = inv * SIMD4<Float>(ndc.x, ndc.y, 0, 1)
        let far4  = inv * SIMD4<Float>(ndc.x, ndc.y, 1, 1)
        let near = SIMD3(near4.x, near4.y, near4.z) / near4.w
        let far  = SIMD3(far4.x,  far4.y,  far4.z)  / far4.w
        return (near, simd_normalize(far - near))
    }

    func pickObstacle(ndc: SIMD2<Float>) -> Int? {
        guard obstacleActive else { return nil }
        let (o, d) = worldRay(ndc: ndc)
        var best = Float.greatestFiniteMagnitude
        var hit: Int? = nil
        for k in 0..<obstacles.count {
            let c = SIMD3(obstacles[k].x, obstacles[k].y, obstacles[k].z)
            let r = obstacles[k].w
            let m = o - c
            let b = simd_dot(m, d)
            let cc = simd_dot(m, m) - r * r
            let disc = b * b - cc
            if disc < 0 { continue }
            let t = -b - sqrt(disc)
            if t > 0 && t < best { best = t; hit = k }
        }
        return hit
    }

    // Drag in the plane through the obstacle centre facing the camera, so it
    // tracks the cursor at roughly constant screen depth. Clamped to the box.
    func dragObstacle(_ k: Int, ndc: SIMD2<Float>) {
        guard k >= 0, k < obstacles.count else { return }
        let (o, d) = worldRay(ndc: ndc)
        let c = SIMD3(obstacles[k].x, obstacles[k].y, obstacles[k].z)
        let n = simd_normalize(camera.target - camera.eye())
        let denom = simd_dot(d, n)
        if abs(denom) < 1e-5 { return }
        let t = simd_dot(c - o, n) / denom
        if t <= 0 { return }
        var p = o + d * t
        let r = obstacles[k].w
        let lo = params.boundsMin + r
        let hi = params.boundsMax - r
        p = simd_min(simd_max(p, lo), hi)
        obstacles[k].x = p.x; obstacles[k].y = p.y; obstacles[k].z = p.z
    }

    // MARK: frame

    func draw(in view: MTKView) {
        guard let sceneColor, let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer() else { return }

        if state.resetRequested { state.resetRequested = false; cur = 0; seedSimulation() }

        // ---- sync live params from the UI ----
        params.predatorCount = state.predatorsOn ? 3 : 0
        obstacleActive       = state.obstacleOn
        params.obstacleCount = obstacleActive ? UInt32(obstacles.count) : 0
        params.wSep = state.sepWeight
        params.wAli = state.aliWeight
        params.wCoh = state.cohWeight
        params.perception = state.perception
        params.maxSpeed = state.maxSpeed
        params.obsRadius = state.obstacleRadius
        memcpy(paramsBuf.contents(), &params, MemoryLayout<SimParams>.stride)

        // push current radius into every obstacle and upload
        for i in 0..<obstacles.count { obstacles[i].w = state.obstacleRadius }
        obstacles.withUnsafeBytes { raw in
            memcpy(obstacleBuf.contents(), raw.baseAddress!, obstacles.count * MemoryLayout<SIMD4<Float>>.stride)
        }

        // clear trail history after a reset / resize
        if needsTrailClear {
            clear(trailA, cmd: cmd); clear(trailB, cmd: cmd)
            trailCur = 0; needsTrailClear = false
        }

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
                enc.setBuffer(obstacleBuf, offset: 0, index: 9)
                enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)

                if params.predatorCount > 0 {
                    enc.setComputePipelineState(predatorsPipeline)
                    enc.setBuffer(posBuf[src], offset: 0, index: 0)
                    enc.setBuffer(predPosBuf[src], offset: 0, index: 4)
                    enc.setBuffer(predVelBuf[src], offset: 0, index: 5)
                    enc.setBuffer(predPosBuf[dst], offset: 0, index: 6)
                    enc.setBuffer(predVelBuf[dst], offset: 0, index: 7)
                    enc.setBuffer(paramsBuf, offset: 0, index: 8)
                    enc.setBuffer(obstacleBuf, offset: 0, index: 9)
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

            // solid obstacle spheres (opaque, depth-writing)
            if params.obstacleCount > 0 {
                r.setRenderPipelineState(obstacleRender)
                r.setVertexBuffer(sphereVtxBuf, offset: 0, index: 0)
                r.setVertexBuffer(obstacleBuf, offset: 0, index: 1)
                r.setVertexBuffer(uniformsBuf, offset: 0, index: 2)
                r.setFragmentBuffer(uniformsBuf, offset: 0, index: 0)
                r.drawIndexedPrimitives(type: .triangle, indexCount: sphereIdxCount,
                                        indexType: .uint16, indexBuffer: sphereIdxBuf,
                                        indexBufferOffset: 0, instanceCount: Int(params.obstacleCount))
            }

            // boids
            r.setRenderPipelineState(boidRender)
            r.setVertexBuffer(posBuf[cur], offset: 0, index: 0)
            r.setVertexBuffer(velBuf[cur], offset: 0, index: 1)
            r.setVertexBuffer(uniformsBuf, offset: 0, index: 2)
            r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3,
                             instanceCount: Int(params.count))

            // predators (same pipeline, predator flag via inline bytes)
            if params.predatorCount > 0 {
                var up = u; up.flags = SIMD4(1, 0, 0, 0)
                r.setVertexBuffer(predPosBuf[cur], offset: 0, index: 0)
                r.setVertexBuffer(predVelBuf[cur], offset: 0, index: 1)
                r.setVertexBytes(&up, length: MemoryLayout<RenderUniforms>.stride, index: 2)
                r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3,
                                 instanceCount: Int(params.predatorCount))
            }
            r.endEncoding()
        }

        // ---- bloom + trails: bright -> trail accumulate -> H blur -> V blur ----
        fullscreen(cmd, pipeline: brightPipeline, target: brightTex) { enc in
            enc.setFragmentTexture(self.sceneColor, index: 0)
            enc.setFragmentSamplerState(self.sampler, index: 0)
        }

        let trailSrc = trailCur == 0 ? trailA! : trailB!
        let trailDst = trailCur == 0 ? trailB! : trailA!
        var decay = state.trailDecay
        fullscreen(cmd, pipeline: trailPipeline, target: trailDst) { enc in
            enc.setFragmentTexture(trailSrc, index: 0)
            enc.setFragmentTexture(self.brightTex, index: 1)
            enc.setFragmentSamplerState(self.sampler, index: 0)
            enc.setFragmentBytes(&decay, length: MemoryLayout<Float>.stride, index: 0)
        }
        trailCur = 1 - trailCur

        var offH = SIMD2<Float>(1.0 / Float(brightTex.width), 0)
        fullscreen(cmd, pipeline: blurPipeline, target: blurTmp) { enc in
            enc.setFragmentTexture(trailDst, index: 0)
            enc.setFragmentSamplerState(self.sampler, index: 0)
            enc.setFragmentBytes(&offH, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }
        var offV = SIMD2<Float>(0, 1.0 / Float(blurTmp.height))
        fullscreen(cmd, pipeline: blurPipeline, target: bloomTex) { enc in
            enc.setFragmentTexture(self.blurTmp, index: 0)
            enc.setFragmentSamplerState(self.sampler, index: 0)
            enc.setFragmentBytes(&offV, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }

        // ---- composite to drawable ----
        if let rp = view.currentRenderPassDescriptor,
           let r = cmd.makeRenderCommandEncoder(descriptor: rp) {
            var strength = state.bloomStrength
            r.setRenderPipelineState(compositePipeline)
            r.setFragmentTexture(sceneColor, index: 0)
            r.setFragmentTexture(bloomTex, index: 1)
            r.setFragmentSamplerState(sampler, index: 0)
            r.setFragmentBytes(&strength, length: MemoryLayout<Float>.stride, index: 0)
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

    private func clear(_ tex: MTLTexture, cmd: MTLCommandBuffer) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = tex
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        cmd.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
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
