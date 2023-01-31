import java.lang.Double.max
import kotlin.math.pow
import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default
import java.io.File

data class NetworkParams(
    val layerToFunction: Map<Int, (Double) -> Double>,
    val biasLimits: Pair<Double, Double>
)

object NetworkParamsFactory {
    fun create(processors: List<(Double) -> Double>, lo: Double, hi: Double): NetworkParams {
        val layerToFunctionMap: Map<Int, (Double) -> Double> = ((0..processors.size).zip(processors).toMap())
        return NetworkParams(layerToFunctionMap, Pair(lo, hi))
    }
}

fun reLU(x: Double): Double = max(0.0, x)

fun sigmoid(x: Double): Double = 1 / (1 + Math.E.pow(x))

fun calculateSingleLayer(nodes: Matrix, weights: Matrix, bias: Matrix, function: (Double) -> Double): Matrix {
    val weighted = weights * nodes
    return (weighted + bias).applyFunction(function)
}

fun inferenceProcess(inputNodes: Matrix, params: NetworkParams): Matrix {
    var currentLayer = inputNodes
    for ((_, processorFunction) in params.layerToFunction) {
        // TODO: may be wrong dimensions
        val relationWeights = Matrix(currentLayer.rows, currentLayer.rows).fillWithRandomNumbers()
        val bias = Matrix(currentLayer.rows, 1).fillWithRandomNumbers(params.biasLimits.first, params.biasLimits.second)

        currentLayer = calculateSingleLayer(currentLayer, relationWeights, bias, processorFunction)
    }
    return currentLayer
}
/*
2048x2048 * 2048x1 = 2048x1 + 2048x1 = 2048x1
2048x2048 * 2048x1
 */

// Input matrix interpreted as a single vector of values
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
}

fun main(args: Array<String>) {
    val params: NetworkParams = NetworkParamsFactory.create(listOf(::reLU, ::sigmoid), 0.0, 10.0)

    ArgParser(args).parseInto(::Args).run {
        val input = readInput(source)
        val processed = inferenceProcess(input, params)
        if (dest == null) {
            println(processed)
        } else {
            dest!!.writeText(processed.toString())
        }
    }
}