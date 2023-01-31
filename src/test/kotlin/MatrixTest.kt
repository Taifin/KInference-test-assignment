import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.assertThrows
import java.lang.IllegalArgumentException

internal class MatrixTest {

    @Test
    fun timesCorrect() {
        val matrix1 = Matrix(arrayOf(arrayOf(1.0, 2.0, 3.0), arrayOf(4.0, 5.0, 6.0)))
        val matrix2 = Matrix(arrayOf(arrayOf(6.0, 5.0), arrayOf(4.0, 3.0), arrayOf(2.0, 1.0)))

        val result = matrix1 * matrix2
        Assertions.assertEquals(2, result.rows)
        Assertions.assertEquals(2, result.cols)
        Assertions.assertEquals(Matrix(arrayOf(arrayOf(20.0, 14.0), arrayOf(56.0, 41.0))), result)
    }

    @Test
    fun timesInvalid() {
        val matrix1 = Matrix(2, 3)
        val matrix2 = Matrix(2, 2)
        assertThrows<IllegalArgumentException> { matrix1 * matrix2 }
    }

    @Test
    fun timesFast() {
        val matrix1 = Matrix(250, 512).fillWithRandomNumbers()
        val matrix2 = Matrix(512, 512).fillWithRandomNumbers()

        val simple = matrix1.simpleMultiplication(matrix2)
        val fast = matrix1 * matrix2
        Assertions.assertEquals(simple, fast)
    }

    @Test
    fun plusCorrect() {
        val matrix1 = Matrix(arrayOf(arrayOf(1.0, 2.0, 3.0), arrayOf(4.0, 5.0, 6.0)))
        val matrix2 = Matrix(arrayOf(arrayOf(6.0, 5.0, 4.0), arrayOf(3.0, 2.0, 1.0)))

        val result = matrix1 + matrix2
        Assertions.assertEquals(2, result.rows)
        Assertions.assertEquals(3, result.cols)
        Assertions.assertEquals(Matrix(arrayOf(arrayOf(7.0, 7.0, 7.0), arrayOf(7.0, 7.0, 7.0))), result)
    }

    @Test
    fun testInvalidAddition() {
        val matrix1 = Matrix(2, 3)
        val matrix2 = Matrix(3, 2)
        Assertions.assertThrows(IllegalArgumentException::class.java) { matrix1 + matrix2 }
    }

    @Test
    fun testApplyFunction() {
        val matrix = Matrix(10, 20).fillWithRandomNumbers()

        Assertions.assertEquals(matrix + matrix, matrix.applyFunction { a: Double -> 2 * a })
    }

}