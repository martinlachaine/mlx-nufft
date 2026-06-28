// MPSGraph 2D complex64 FFT bench — candidate vs MLX four-step (WS1 screen).
// Batched 2D inverse FFT over axes [1,2] of a (nu3,nu1,nu2) ComplexFloat32
// tensor, the same work the slab pipeline's FFT stage does. Args: nu nu3.
import Foundation
import Metal
import MetalPerformanceShadersGraph

let args = CommandLine.arguments
let nu = args.count > 1 ? Int(args[1])! : 7200
let nu3 = args.count > 2 ? Int(args[2])! : 24
let reps = 5

guard let dev = MTLCreateSystemDefaultDevice(),
      let q = dev.makeCommandQueue() else { fatalError("no Metal device") }

let nElem = nu3 * nu * nu          // complex elements
let bytes = nElem * 8              // ComplexFloat32 = 2 x float32
guard let buf = dev.makeBuffer(length: bytes, options: .storageModeShared) else {
    fatalError("alloc failed (\(Double(bytes)/1e9) GB)")
}
// fill with deterministic pseudo-random floats
let fp = buf.contents().bindMemory(to: Float.self, capacity: nElem * 2)
var seed: UInt64 = 88172645463325252
for i in 0..<(nElem * 2) {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
    fp[i] = Float(Int32(truncatingIfNeeded: seed)) / Float(Int32.max)
}

let graph = MPSGraph()
let shape = [NSNumber(value: nu3), NSNumber(value: nu), NSNumber(value: nu)]
let inp = graph.placeholder(shape: shape, dataType: .complexFloat32, name: "x")
let desc = MPSGraphFFTDescriptor()
desc.inverse = true
desc.scalingMode = .none
let out = graph.fastFourierTransform(inp, axes: [1, 2], descriptor: desc, name: "ifft2")
let td = MPSGraphTensorData(buf, shape: shape, dataType: .complexFloat32)

func run() {
    let r = graph.run(with: q, feeds: [inp: td], targetTensors: [out],
                      targetOperations: nil)
    // force materialization
    _ = r[out]
}

// warm
do { run() }
var best = Double.greatestFiniteMagnitude
for _ in 0..<reps {
    let t0 = DispatchTime.now().uptimeNanoseconds
    run()
    let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    best = min(best, dt)
}
let gib = Double(bytes) / 1073741824.0
print(String(format: "MPSGraph ifft2 (%dx%dx%d): %.1f ms (%.2f ms/slab, grid %.1f GiB c64)",
             nu3, nu, nu, best, best / Double(nu3), gib))
