import java.lang.Double.max
import kotlin.math.pow
import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default
import java.io.File

data class NetworkParams(
    val activationFunctions: List<(Double) -> Double>,
    val layerSizes: List<Int>,
    val biasLimits: Pair<Double, Double>,
    val numberOfLayers: Int = layerSizes.size
)

fun reLU(x: Double): Double = max(0.0, x)

fun sigmoid(x: Double): Double = 1 / (1 + Math.E.pow(x))

fun calculateDenseLayer(nodes: Matrix, weights: Matrix, bias: Matrix, function: (Double) -> Double): Matrix {
    val weighted = weights * nodes
    return (weighted + bias).applyFunction(function)
}

fun inferenceProcess(inputNodes: Matrix, params: NetworkParams): Matrix {
    var currentLayer = inputNodes
    for (i in 0 until params.numberOfLayers) {
        val relationWeights = Matrix(params.layerSizes[i], currentLayer.rows).fillWithRandomNumbers()
        val bias =
            Matrix(params.layerSizes[i], 1).fillWithRandomNumbers(params.biasLimits.first, params.biasLimits.second)

        currentLayer = calculateDenseLayer(currentLayer, relationWeights, bias, params.activationFunctions[i])
    }
    return currentLayer
}

// Input matrix is interpreted as a single vector
fun readInput(inFile: File): Matrix {
    val list = mutableListOf<Double>()
    for (line in inFile.readLines()) {
        list += line.split(" ").map { it.toDouble() }
    }
    return Matrix(arrayOf(list.toTypedArray())).transpose()
}

class Args(parser: ArgParser) {
    val source by parser.positional("source filename") { File(this) }
    val dest by parser.storing("-o", "--out", help = "destination filename") { File(this) }.default<File?>(null)
    val generate by parser.storing(
        "-g",
        "--gen",
        help = "ignore input file and generate random input vector of given size"
    ) { toInt() }
        .default<Int?>(null)
    val sizes by parser.adding(
        "-s",
        "--size",
        help = "specify sizes of dense layers (by default 512 and 512)"
    ) { toInt() }.default(
        mutableListOf(512, 512)
    )
}

fun main(args: Array<String>) {

    ArgParser(args).parseInto(::Args).run {
        val params = NetworkParams(listOf(::reLU, ::sigmoid), sizes, Pair(0.0, 10.0))
        val input: Matrix = if (generate == null) {
            readInput(source)
        } else {
            Matrix(generate!!, 1).fillWithRandomNumbers()
        }
        val processed = inferenceProcess(input, params)
        if (dest == null) {
            println(processed)
        } else {
            dest!!.writeText(processed.toString())
        }
    }
}