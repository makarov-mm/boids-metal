// Boids 3D / Swarm Intelligence — Metal shaders.
//
// Simulation runs entirely on the GPU as an N-body-style brute-force kernel
// with threadgroup tiling (each thread cooperatively loads a tile of boids
// into threadgroup memory, then every thread reads the tile from there).
// For ~16k boids this is far cheaper than a CPU hash grid on Apple Silicon and
// keeps the code dependency-free.
//
// Obstacles are real, opaque, lit spheres now: boids both steer away from them
// (soft force) and are physically pushed out of them (hard, non-penetrating
// constraint), so nothing is ever seen inside the solid geometry. They can be
// dragged at runtime, so their positions/radii arrive in a small float4 buffer.
//
// Rendering is instanced camera-facing arrowheads into an HDR target, followed
// by a bright-pass, a temporal trail-accumulation feedback, a separable
// Gaussian blur and an ACES tone-map composite.

#include <metal_stdlib>
using namespace metal;

// ---- shared parameter block (mirror of SimParams in Swift) ----
// All fields are 4-byte scalars so the layout is identical on both sides
// without any alignment guesswork.
struct SimParams {
    uint  count;
    uint  predatorCount;
    uint  obstacleCount;
    float dt;
    float perception;
    float sepDist;
    float maxSpeed;
    float minSpeed;
    float maxForce;
    float wSep;
    float wAli;
    float wCoh;
    float wBounds;
    float wFlee;
    float bMinX, bMinY, bMinZ;
    float bMaxX, bMaxY, bMaxZ;
    float margin;
    float predMaxSpeed;
    float predMaxForce;
    float fleeRadius;
    float obsRadius;
    float wObstacle;
};

struct RenderUniforms {
    float4x4 viewProj;
    float4   eye;    // xyz = camera position
    float4   misc;   // x=boidSize  y=maxSpeed  z=time  w=aspect
    float4   flags;  // x=isPredator (0/1)
};

static inline float3 limitVec(float3 v, float m) {
    float l2 = dot(v, v);
    if (l2 > m * m && l2 > 1e-12) return v * (m / sqrt(l2));
    return v;
}
static inline float3 setMag(float3 v, float m) {
    float l = length(v);
    return l > 1e-6 ? v * (m / l) : float3(0.0);
}

constant uint TILE = 256;

// ===================== SIMULATION =====================

kernel void boids_update(device const float3* posIn     [[buffer(0)]],
                         device const float3* velIn     [[buffer(1)]],
                         device float3*       posOut    [[buffer(2)]],
                         device float3*       velOut    [[buffer(3)]],
                         device const float3* predPos   [[buffer(4)]],
                         constant SimParams&  P         [[buffer(8)]],
                         device const float4* obstacles [[buffer(9)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]])
{
    threadgroup float3 sPos[TILE];
    threadgroup float3 sVel[TILE];

    // Padding threads (gid >= count) must still hit every barrier, so they
    // run the loop but never read their own boid or write a result.
    bool active = gid < P.count;
    float3 pi = active ? posIn[gid] : float3(0.0);
    float3 vi = active ? velIn[gid] : float3(0.0);

    const float per2 = P.perception * P.perception;
    const float sep2 = P.sepDist * P.sepDist;

    float3 sepSum = 0, aliSum = 0, cohSum = 0;
    int    cohCount = 0;

    uint tiles = (P.count + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; ++t) {
        uint idx = t * TILE + lid;
        sPos[lid] = idx < P.count ? posIn[idx] : float3(1e9);
        sVel[lid] = idx < P.count ? velIn[idx] : float3(0.0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint base = t * TILE;
            uint lim  = min(TILE, P.count - base);
            for (uint j = 0; j < lim; ++j) {
                if (base + j == gid) continue;
                float3 d = pi - sPos[j];
                float dist2 = dot(d, d);
                if (dist2 > per2 || dist2 < 1e-8) continue;
                cohSum += sPos[j];
                aliSum += sVel[j];
                cohCount++;
                if (dist2 < sep2) sepSum += d * (1.0 / dist2);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float3 acc = 0;
    if (cohCount > 0) {
        float inv = 1.0 / float(cohCount);
        float3 desiredAli = setMag(aliSum * inv, P.maxSpeed);
        acc += limitVec(desiredAli - vi, P.maxForce) * P.wAli;
        float3 center = cohSum * inv;
        float3 desiredCoh = setMag(center - pi, P.maxSpeed);
        acc += limitVec(desiredCoh - vi, P.maxForce) * P.wCoh;
    }
    if (dot(sepSum, sepSum) > 0) {
        float3 desired = setMag(sepSum, P.maxSpeed);
        acc += limitVec(desired - vi, P.maxForce) * P.wSep;
    }

    // flee predators
    float flee2 = P.fleeRadius * P.fleeRadius;
    float3 flee = 0;
    for (uint k = 0; k < P.predatorCount; ++k) {
        float3 d = pi - predPos[k];
        float d2 = dot(d, d);
        if (d2 < flee2 && d2 > 1e-6) flee += d * (1.0 / d2);
    }
    if (dot(flee, flee) > 0) {
        float3 desired = setMag(flee, P.maxSpeed);
        acc += limitVec(desired - vi, P.maxForce) * P.wFlee;
    }

    // soft bounds
    float3 bmin = float3(P.bMinX, P.bMinY, P.bMinZ);
    float3 bmax = float3(P.bMaxX, P.bMaxY, P.bMaxZ);
    float3 b = 0;
    if (pi.x < bmin.x + P.margin) b.x += 1; else if (pi.x > bmax.x - P.margin) b.x -= 1;
    if (pi.y < bmin.y + P.margin) b.y += 1; else if (pi.y > bmax.y - P.margin) b.y -= 1;
    if (pi.z < bmin.z + P.margin) b.z += 1; else if (pi.z > bmax.z - P.margin) b.z -= 1;
    if (dot(b, b) > 0) acc += setMag(b, P.maxForce) * P.wBounds;

    // soft avoidance of every solid obstacle
    for (uint k = 0; k < P.obstacleCount; ++k) {
        float4 ob = obstacles[k];
        float3 d = pi - ob.xyz;
        float dist = length(d);
        float safe = ob.w + P.margin;
        if (dist < safe && dist > 1e-4) {
            float s = (safe - dist) / safe;
            acc += setMag(d, P.maxForce) * (P.wObstacle * s);
        }
    }

    // symplectic (semi-implicit) Euler: integrate position with the *new*
    // velocity — cheaper than RK and stays stable at these timesteps.
    float3 v = vi + acc;
    float sp = length(v);
    if (sp > P.maxSpeed)                    v *= P.maxSpeed / sp;
    else if (sp < P.minSpeed && sp > 1e-5)  v *= P.minSpeed / sp;

    float3 newPos = pi + v * P.dt;

    // hard, non-penetrating collision: if a boid ends up inside a solid
    // sphere, snap it to the surface and remove the inward velocity so it
    // slides along instead of tunnelling through.
    for (uint k = 0; k < P.obstacleCount; ++k) {
        float4 ob = obstacles[k];
        float3 d = newPos - ob.xyz;
        float dist = length(d);
        if (dist < ob.w && dist > 1e-5) {
            float3 n = d / dist;
            newPos = ob.xyz + n * ob.w;
            float vn = dot(v, n);
            if (vn < 0) v -= n * vn;
        }
    }

    if (active) {
        velOut[gid] = v;
        posOut[gid] = newPos;
    }
}

kernel void predators_update(device const float3* posIn      [[buffer(0)]],
                             device const float3* predPosIn  [[buffer(4)]],
                             device const float3* predVelIn  [[buffer(5)]],
                             device float3*       predPosOut [[buffer(6)]],
                             device float3*       predVelOut [[buffer(7)]],
                             constant SimParams&  P          [[buffer(8)]],
                             device const float4* obstacles  [[buffer(9)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= P.predatorCount) return;
    float3 pk = predPosIn[gid];

    // nearest boid by brute force (cheap: a handful of predators)
    float best = 1e30; int bi = -1;
    for (uint j = 0; j < P.count; ++j) {
        float3 d = posIn[j] - pk;
        float d2 = dot(d, d);
        if (d2 < best) { best = d2; bi = int(j); }
    }

    float3 acc = 0;
    if (bi >= 0) {
        float3 desired = setMag(posIn[bi] - pk, P.predMaxSpeed);
        acc += limitVec(desired - predVelIn[gid], P.predMaxForce);
    }

    float3 bmin = float3(P.bMinX, P.bMinY, P.bMinZ);
    float3 bmax = float3(P.bMaxX, P.bMaxY, P.bMaxZ);
    float3 b = 0;
    if (pk.x < bmin.x + P.margin) b.x += 1; else if (pk.x > bmax.x - P.margin) b.x -= 1;
    if (pk.y < bmin.y + P.margin) b.y += 1; else if (pk.y > bmax.y - P.margin) b.y -= 1;
    if (pk.z < bmin.z + P.margin) b.z += 1; else if (pk.z > bmax.z - P.margin) b.z -= 1;
    if (dot(b, b) > 0) acc += setMag(b, P.predMaxForce) * 2.5;

    // predators also steer around the solid spheres
    for (uint k = 0; k < P.obstacleCount; ++k) {
        float4 ob = obstacles[k];
        float3 d = pk - ob.xyz;
        float dist = length(d);
        float safe = ob.w + P.margin;
        if (dist < safe && dist > 1e-4) {
            float s = (safe - dist) / safe;
            acc += setMag(d, P.predMaxForce) * (3.0 * s);
        }
    }

    float3 v = predVelIn[gid] + acc;
    float sp = length(v);
    if (sp > P.predMaxSpeed) v *= P.predMaxSpeed / sp;

    float3 newPos = pk + v * P.dt;
    for (uint k = 0; k < P.obstacleCount; ++k) {
        float4 ob = obstacles[k];
        float3 d = newPos - ob.xyz;
        float dist = length(d);
        if (dist < ob.w && dist > 1e-5) {
            float3 n = d / dist;
            newPos = ob.xyz + n * ob.w;
            float vn = dot(v, n);
            if (vn < 0) v -= n * vn;
        }
    }

    predVelOut[gid] = v;
    predPosOut[gid] = newPos;
}

// ===================== SCENE RENDER =====================

struct VSOut {
    float4 pos [[position]];
    float3 color;
};

// Three vertices per instance form a camera-facing arrowhead aligned to
// velocity. Fast boids get an HDR (>1) emissive boost so bloom catches them.
vertex VSOut boid_vertex(uint vid [[vertex_id]],
                         uint iid [[instance_id]],
                         device const float3* pos [[buffer(0)]],
                         device const float3* vel [[buffer(1)]],
                         constant RenderUniforms& U [[buffer(2)]])
{
    float3 p  = pos[iid];
    float3 vv = vel[iid];
    float  sp = length(vv);
    float3 f  = sp > 1e-5 ? vv / sp : float3(1, 0, 0);

    float3 toCam = normalize(U.eye.xyz - p);
    float3 side  = cross(f, toCam);
    if (dot(side, side) < 1e-6) side = cross(f, float3(0, 1, 0));
    side = normalize(side);

    bool  isPred = U.flags.x > 0.5;
    float size   = U.misc.x * (isPred ? 2.4 : 1.0);

    float3 tip = p + f * size;
    float3 bl  = p - f * (size * 0.5) + side * (size * 0.45);
    float3 br  = p - f * (size * 0.5) - side * (size * 0.45);
    float3 wp  = (vid == 0) ? tip : (vid == 1 ? bl : br);

    float maxSpeed = U.misc.y;
    float t = clamp(sp / maxSpeed, 0.0, 1.0);

    float3 col;
    float  emis;
    if (isPred) {
        col  = float3(1.0, 0.16, 0.10);
        emis = 3.4;
    } else {
        float shade = 0.55 + 0.45 * t;
        col  = float3((0.5 + 0.5 * f.x) * shade,
                      (0.5 + 0.5 * f.z) * shade,
                      (0.62 + 0.38 * f.y) * shade);
        emis = 0.7 + 1.6 * t * t;        // HDR tail on the fastest boids
    }
    float tipBoost = (vid == 0) ? 1.35 : 0.75;   // bright nose, dim tail

    VSOut o;
    o.pos   = U.viewProj * float4(wp, 1.0);
    o.color = col * emis * tipBoost;
    return o;
}

fragment float4 boid_fragment(VSOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}

// ---- solid, opaque, lit obstacle spheres ----
struct ObsVSOut {
    float4 pos  [[position]];
    float3 nrm;
    float3 wpos;
};

// Unit-sphere positions double as normals. Each instance reads its centre and
// radius from the obstacle buffer.
vertex ObsVSOut obstacle_vertex(uint vid [[vertex_id]],
                                uint iid [[instance_id]],
                                device const float3* unitVtx  [[buffer(0)]],
                                device const float4* obstacles [[buffer(1)]],
                                constant RenderUniforms& U     [[buffer(2)]])
{
    float3 n  = unitVtx[vid];
    float4 ob = obstacles[iid];
    float3 wp = ob.xyz + n * ob.w;

    ObsVSOut o;
    o.pos  = U.viewProj * float4(wp, 1.0);
    o.nrm  = n;
    o.wpos = wp;
    return o;
}

// Matte diffuse + ambient + a cool fresnel rim. Values stay below 1.0 so the
// spheres read as solid surfaces and do not feed the bloom.
fragment float4 obstacle_fragment(ObsVSOut in [[stage_in]],
                                  constant RenderUniforms& U [[buffer(0)]])
{
    float3 N = normalize(in.nrm);
    float3 L = normalize(float3(0.4, 0.92, 0.35));
    float3 V = normalize(U.eye.xyz - in.wpos);
    float  diff = max(dot(N, L), 0.0);
    float  amb  = 0.18;
    float  fres = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.45;

    float3 base = float3(0.60, 0.30, 0.17);                  // warm matte
    float3 col  = base * (amb + 0.90 * diff)
                + fres * float3(0.35, 0.50, 0.85);           // cool rim
    return float4(col, 1.0);
}

// bounding-box wireframe
vertex float4 line_vertex(uint vid [[vertex_id]],
                          device const float3* p [[buffer(0)]],
                          constant RenderUniforms& U [[buffer(1)]])
{
    return U.viewProj * float4(p[vid], 1.0);
}
fragment float4 line_fragment() {
    return float4(0.10, 0.18, 0.32, 1.0);
}

// ===================== POST =====================

struct FSOut {
    float4 pos [[position]];
    float2 uv;
};

// fullscreen triangle, no vertex buffer
vertex FSOut fs_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    FSOut o;
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv  = float2(p.x, 1.0 - p.y);
    return o;
}

fragment float4 bright_pass(FSOut in [[stage_in]],
                            texture2d<float> scene [[texture(0)]],
                            sampler smp [[sampler(0)]])
{
    float3 c = scene.sample(smp, in.uv).rgb;
    float  l = dot(c, float3(0.2126, 0.7152, 0.0722));
    float  k = smoothstep(1.0, 2.2, l);
    return float4(c * k, 1.0);
}

// Temporal trail: previous accumulation faded by `decay`, plus this frame's
// bright pass. Long-lived bright/fast boids leave glowing streaks; the dim
// box and the opaque spheres never enter the bright pass, so they don't smear.
fragment float4 trail_pass(FSOut in [[stage_in]],
                           texture2d<float> prev [[texture(0)]],
                           texture2d<float> add  [[texture(1)]],
                           sampler smp [[sampler(0)]],
                           constant float& decay [[buffer(0)]])
{
    float3 p = prev.sample(smp, in.uv).rgb;
    float3 a = add.sample(smp, in.uv).rgb;
    return float4(p * decay + a, 1.0);
}

fragment float4 blur_pass(FSOut in [[stage_in]],
                          texture2d<float> tex [[texture(0)]],
                          sampler smp [[sampler(0)]],
                          constant float2& offset [[buffer(0)]])
{
    const float w[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
    float3 c = tex.sample(smp, in.uv).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        c += tex.sample(smp, in.uv + offset * float(i)).rgb * w[i];
        c += tex.sample(smp, in.uv - offset * float(i)).rgb * w[i];
    }
    return float4(c, 1.0);
}

static float3 acesTonemap(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

fragment float4 composite(FSOut in [[stage_in]],
                          texture2d<float> scene [[texture(0)]],
                          texture2d<float> bloom [[texture(1)]],
                          sampler smp [[sampler(0)]],
                          constant float& bloomStrength [[buffer(0)]])
{
    float3 c  = scene.sample(smp, in.uv).rgb;
    float3 bl = bloom.sample(smp, in.uv).rgb;
    c += bl * bloomStrength;                          // additive glow + trails

    float2 q   = in.uv - 0.5;
    float  vig = smoothstep(0.95, 0.30, length(q));   // soft vignette
    c *= mix(0.70, 1.0, vig);

    c = acesTonemap(c * 1.12);
    c = pow(c, float3(1.0 / 2.2));                    // gamma
    return float4(c, 1.0);
}
