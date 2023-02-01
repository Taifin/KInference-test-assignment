import kotlin.math.abs
import kotlin.math.max
import kotlin.random.Random

class Matrix(private val values: Array<Array<Double>>) {
    val rows: Int = values.size
    val cols: Int = values.first().size
    companion object StaticParams {
        val simpleMultiplicationThreshold: Int = 2048
    }

    /**
     * Constructor that creates a matrix with the given number of rows and columns,
     * filled with zeros.
     *
     * @param rows_ number of rows in the matrix
     * @param cols_ number of columns in the matrix
     */
    constructor(rows_: Int, cols_: Int) : this(Array<Array<Double>>(rows_) { Array<Double>(cols_) { 0.0 } })

    /**
     * Overloaded `times` operator for matrix multiplication.
     * For small matrices performs a naive matrix multiplication. Size of 'small' matrix is controlled by simpleMultiplicationThreshold property.
     * Otherwise, performs Strassen's algorithm for fast matrix multiplication.
     *
     * @param other the matrix to multiply with
     * @return the result of the matrix multiplication
     * @throws IllegalArgumentException if the number of columns in `this` matrix does not match the number of rows in `other`
     */
    operator fun times(other: Matrix): Matrix {
        require(cols == other.rows) { "Invalid matrix multiplication: ${this.cols} != ${other.rows}" }

        if (rows <= simpleMultiplicationThreshold || cols <= simpleMultiplicationThreshold) {
            return simpleMultiplication(other)
        }

        val origRows = rows
        val origCols = other.cols

        val aParts = this.squarify().split()
        val bParts = other.squarify().split()

        val c = Array(2) { arrayOfNulls<Matrix>(2) }
        var mult = (aParts[0][0] + aParts[1][1]) * (bParts[0][0] + bParts[1][1])
        c[0][0] = mult
        c[1][1] = mult

        mult = (aParts[1][0] + aParts[1][1]) * bParts[0][0]
        c[1][0] = mult
        c[1][1] = c[1][1]!!.minus(mult)

        mult = aParts[0][0] * (bParts[0][1] - bParts[1][1])
        c[0][1] = mult
        c[1][1] = c[1][1]!!.plus(mult)

        mult = aParts[1][1] * (bParts[1][0] - bParts[0][0])
        c[0][0] = c[0][0]!!.plus(mult)
        c[1][0] = c[1][0]!!.plus(mult)

        mult = (aParts[0][0] + aParts[0][1]) * bParts[1][1]
        c[0][0] = c[0][0]?.minus(mult)
        c[0][1] = c[0][1]?.plus(mult)

        mult = (aParts[1][0] - aParts[0][0]) * (bParts[0][0] + bParts[0][1])
        c[1][1] = c[1][1]?.plus(mult)

        mult = (aParts[0][1] - aParts[1][1]) * (bParts[1][0] + bParts[1][1])
        c[0][0] = c[0][0]?.plus(mult)

        @Suppress("UNCHECKED_CAST")
        return merge(c as Array<Array<Matrix>>).unsqarify(origRows, origCols)
    }

    private fun elementwiseBinaryOp(other: Matrix, func: (Double, Double) -> Double): Matrix {
        require(cols == other.cols && rows == other.rows) {
            "Invalid matrix elementwise operation: (${this.rows}, ${this.cols}) != (${other.rows}, ${other.cols})"
        }

        val result = Matrix(rows, cols)

        for (i in 0 until rows) {
            result.values[i] = values[i].zip(other.values[i], func).toTypedArray()
        }

        return result
    }

    /**
     * Method that splits the matrix into 4 sub-matrices,
     * 2 for the top half and 2 for the bottom half. Matrix must be squared.
     *
     * @return a 2D array of sub-matrices, each containing 2 sub-matrices
     */
    private fun split(): Array<Array<Matrix>> {
        val n = rows
        val m = n / 2
        val a11 = subMatrix(0, m, 0, m)
        val a12 = subMatrix(0, m, m, n)
        val a21 = subMatrix(m, n, 0, m)
        val a22 = subMatrix(m, n, m, n)
        return arrayOf(arrayOf(a11, a12), arrayOf(a21, a22))
    }

    /**
     * Merges four matrices into a single matrix by combining them into a 2x2 matrix.
     *
     * @param a An array of two arrays of matrices, each containing two matrices.
     * @return The merged matrix which is the combination of the four matrices.
     */
    private fun merge(a: Array<Array<Matrix>>): Matrix {
        val n = a[0][0].rows
        val result = Matrix(n * 2, n * 2)

        for (i in 0 until n) {
            for (j in 0 until n) {
                result.values[i][j] = a[0][0].values[i][j]
                result.values[i + n][j] = a[1][0].values[i][j]
                result.values[i][j + n] = a[0][1].values[i][j]
                result.values[i + n][j + n] = a[1][1].values[i][j]
            }
        }

        return result
    }

    /**
     * Returns squarified matrix to its original shape.
     * @param realRows The number of rows of the original matrix
     * @param realCols The number of columns of the original matrix
     * @return The unsquared matrix
     */
    private fun unsqarify(realRows: Int, realCols: Int): Matrix {
        val result = Matrix(realRows, realCols)
        for (i in 0 until realRows) {
            for (j in 0 until realCols) {
                result.values[i][j] = values[i][j]
            }
        }

        return result
    }

    /**
     * Adjusts the rows and columns of a given matrix
     * to be a squared matrix with rows and columns being a power of 2.
     *
     * @return A new matrix with number of rows and columns adjusted to be a power of 2.
     */
    private fun squarify(): Matrix {
        var newRows = 1
        var newCols = 1

        while (newRows < max(rows, cols)) {
            newRows *= 2
        }

        while (newCols < max(rows, cols)) {
            newCols *= 2
        }

        if (newCols == cols && newRows == rows) {
            return this
        }

        val newMatrix = Matrix(newRows, newCols)

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                newMatrix.values[i][j] = values[i][j]
            }
        }

        return newMatrix
    }

    /**
     * Performs naive multiplication of two matrices. 'Private' is not set for testing purposes.
     * Please, instead use 'times' operator.
     *
     * @param other: The matrix to be multiplied with the calling matrix.
     * @return A new matrix that is the result of the matrix multiplication.
     * @throws IllegalArgumentException if the multiplication cannot be implemented due to non-compatible dimensions.
     */
    fun simpleMultiplication(other: Matrix): Matrix {
        require(cols == other.rows) { "Invalid matrix multiplication: ${this.cols} != ${other.rows}" }

        val result = Matrix(rows, other.cols)

        for (i in 0 until rows) {
            for (j in 0 until other.cols) {
                for (k in 0 until cols) {
                    result.values[i][j] += this.values[i][k] * other.values[k][j]
                }
            }
        }

        return result
    }

    /**
     * Extracts a sub-matrix from the calling matrix.
     *
     * @param startRow: The starting row index (inclusive).
     * @param endRow: The ending row index (exclusive).
     * @param startCol: The starting column index (inclusive).
     * @param endCol: The ending column index (exclusive).
     * @return Extracted sub-matrix.
     */
    private fun subMatrix(startRow: Int, endRow: Int, startCol: Int, endCol: Int): Matrix {
        val subMatrix = Matrix(endRow - startRow, endCol - startCol)
        for (i in startRow until endRow) {
            for (j in startCol until endCol) {
                subMatrix.values[i - startRow][j - startCol] = values[i][j]
            }
        }
        return subMatrix
    }

    operator fun plus(other: Matrix): Matrix {
        return elementwiseBinaryOp(other, Double::plus)
    }

    operator fun minus(other: Matrix): Matrix {
        return elementwiseBinaryOp(other, Double::minus)
    }

    override fun equals(other: Any?): Boolean {
        infix fun Double.equalsDelta(other: Double) = abs(this - other) < 1e-9

        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Matrix

        if (rows != other.rows) return false
        if (cols != other.cols) return false

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (!(values[i][j] equalsDelta other.values[i][j])) return false
            }
        }

        return true
    }

    /**
     * Fills the matrix with random numbers of type Double. By default, all generated numbers are in [0.0, 1.0) segment.
     *
     * @param lo: lower bound (inclusive)
     * @param hi: upper bound (exclusive)
     */
    fun fillWithRandomNumbers(lo: Double = 0.0, hi: Double = 1.0): Matrix {
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                values[i][j] = Random.nextDouble(lo, hi)
            }
        }

        return this
    }

    fun transpose(): Matrix {
        val result = Matrix(cols, rows)
        for (i in 0 until cols) {
            for (j in 0 until rows) {
                result.values[i][j] = values[j][i]
            }
        }

        return result
    }

    /**
     * Applies a function to each element of the matrix.
     * @param func: A function from Double to Double.
     * @return: The matrix after applying the function to each of its elements.
     */
    fun applyFunction(func: (Double) -> Double): Matrix {
        val result = Matrix(rows, cols)
        for (i in 0 until rows) {
            result.values[i] = values[i].map { func(it) }.toTypedArray()
        }

        return result
    }

    override fun toString(): String {
        var str = ""
        for (i in 0 until rows) {
            str += '\n'
            for (j in 0 until cols) {
                str += values[i][j].toString()
            }
        }
        return str
    }

    /**
     * Presence of this method is advised by Kotlin guidelines when 'equals' method is present.
     */
    override fun hashCode(): Int {
        var result = values.contentDeepHashCode()
        result = 31 * result + rows
        result = 31 * result + cols
        return result
    }

}