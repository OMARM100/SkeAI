#define PY_SSIZE_T_CLEAN
#include <Python.h>

static int matrix_shape(PyObject *matrix, Py_ssize_t *rows, Py_ssize_t *cols) {
    if (!PyList_Check(matrix)) {
        PyErr_SetString(PyExc_TypeError, "matrix must be a list of rows");
        return 0;
    }

    *rows = PyList_GET_SIZE(matrix);
    if (*rows == 0) {
        *cols = 0;
        return 1;
    }

    PyObject *first_row = PyList_GET_ITEM(matrix, 0);
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_TypeError, "matrix rows must be lists");
        return 0;
    }

    *cols = PyList_GET_SIZE(first_row);

    for (Py_ssize_t row = 0; row < *rows; ++row) {
        PyObject *current = PyList_GET_ITEM(matrix, row);
        if (!PyList_Check(current) || PyList_GET_SIZE(current) != *cols) {
            PyErr_SetString(PyExc_ValueError, "matrix must be rectangular");
            return 0;
        }
    }

    return 1;
}

static int get_double(PyObject *value, double *output) {
    if (!PyFloat_Check(value) && !PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "matrix values must be numeric");
        return 0;
    }

    *output = PyFloat_AsDouble(value);
    return PyErr_Occurred() == NULL;
}

static int set_double(PyObject *row, Py_ssize_t index, double value) {
    PyObject *number = PyFloat_FromDouble(value);
    if (number == NULL) {
        return 0;
    }

    if (PyList_SetItem(row, index, number) < 0) {
        Py_DECREF(number);
        return 0;
    }

    return 1;
}

static PyObject *native_matmul(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *left;
    PyObject *right;

    if (!PyArg_ParseTuple(args, "OO:matmul", &left, &right)) {
        return NULL;
    }

    Py_ssize_t rows;
    Py_ssize_t inner;
    Py_ssize_t right_rows;
    Py_ssize_t cols;

    if (!matrix_shape(left, &rows, &inner) ||
        !matrix_shape(right, &right_rows, &cols)) {
        return NULL;
    }

    if (inner != right_rows) {
        PyErr_SetString(
            PyExc_ValueError,
            "Incompatible shapes for matrix multiplication"
        );
        return NULL;
    }

    PyObject *result = PyList_New(rows);
    if (result == NULL) {
        return NULL;
    }

    if (rows == 0 || cols == 0) {
        for (Py_ssize_t row = 0; row < rows; ++row) {
            PyObject *result_row = PyList_New(cols);
            if (result_row == NULL) {
                Py_DECREF(result);
                return NULL;
            }
            for (Py_ssize_t col = 0; col < cols; ++col) {
                PyObject *number = PyFloat_FromDouble(0.0);
                if (number == NULL) {
                    Py_DECREF(result_row);
                    Py_DECREF(result);
                    return NULL;
                }
                PyList_SET_ITEM(result_row, col, number);
            }
            PyList_SET_ITEM(result, row, result_row);
        }
        return result;
    }

    double *result_data = PyMem_Calloc(
        (size_t)(rows * cols),
        sizeof(double)
    );
    if (result_data == NULL) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t row = 0; row < rows; ++row) {
        PyObject *left_row = PyList_GET_ITEM(left, row);

        for (Py_ssize_t k = 0; k < inner; ++k) {
            double value;
            if (!get_double(PyList_GET_ITEM(left_row, k), &value)) {
                PyMem_Free(result_data);
                Py_DECREF(result);
                return NULL;
            }

            PyObject *right_row = PyList_GET_ITEM(right, k);

            for (Py_ssize_t col = 0; col < cols; ++col) {
                double weight;
                if (!get_double(PyList_GET_ITEM(right_row, col), &weight)) {
                    PyMem_Free(result_data);
                    Py_DECREF(result);
                    return NULL;
                }

                result_data[row * cols + col] += value * weight;
            }
        }
    }

    for (Py_ssize_t row = 0; row < rows; ++row) {
        PyObject *result_row = PyList_New(cols);
        if (result_row == NULL) {
            PyMem_Free(result_data);
            Py_DECREF(result);
            return NULL;
        }

        for (Py_ssize_t col = 0; col < cols; ++col) {
            PyObject *number = PyFloat_FromDouble(
                result_data[row * cols + col]
            );
            if (number == NULL) {
                Py_DECREF(result_row);
                PyMem_Free(result_data);
                Py_DECREF(result);
                return NULL;
            }
            PyList_SET_ITEM(result_row, col, number);
        }

        PyList_SET_ITEM(result, row, result_row);
    }

    PyMem_Free(result_data);
    return result;
}

static PyObject *native_dense_forward(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *inputs;
    PyObject *weights;
    PyObject *bias;
    PyObject *output;

    if (!PyArg_ParseTuple(
        args,
        "OOOO:dense_forward",
        &inputs,
        &weights,
        &bias,
        &output
    )) {
        return NULL;
    }

    Py_ssize_t batch_size;
    Py_ssize_t input_size;
    Py_ssize_t weight_rows;
    Py_ssize_t output_size;
    Py_ssize_t output_rows;
    Py_ssize_t output_cols;

    if (!matrix_shape(inputs, &batch_size, &input_size) ||
        !matrix_shape(weights, &weight_rows, &output_size) ||
        !matrix_shape(output, &output_rows, &output_cols)) {
        return NULL;
    }

    if (!PyList_Check(bias) ||
        PyList_GET_SIZE(bias) != output_size ||
        weight_rows != input_size ||
        output_rows != batch_size ||
        output_cols != output_size) {
        PyErr_SetString(PyExc_ValueError, "Dense forward shapes are incompatible");
        return NULL;
    }

    double *result_data = PyMem_Calloc(
        (size_t)(batch_size * output_size),
        sizeof(double)
    );
    if (result_data == NULL) {
        return PyErr_NoMemory();
    }

    for (Py_ssize_t batch = 0; batch < batch_size; ++batch) {
        PyObject *x_row = PyList_GET_ITEM(inputs, batch);

        for (Py_ssize_t input_index = 0; input_index < input_size; ++input_index) {
            double x_value;
            if (!get_double(
                PyList_GET_ITEM(x_row, input_index),
                &x_value
            )) {
                PyMem_Free(result_data);
                return NULL;
            }

            PyObject *weight_row = PyList_GET_ITEM(weights, input_index);

            for (Py_ssize_t col = 0; col < output_size; ++col) {
                double weight_value;
                if (!get_double(
                    PyList_GET_ITEM(weight_row, col),
                    &weight_value
                )) {
                    PyMem_Free(result_data);
                    return NULL;
                }

                result_data[batch * output_size + col] +=
                    x_value * weight_value;
            }
        }
    }

    for (Py_ssize_t batch = 0; batch < batch_size; ++batch) {
        PyObject *result_row = PyList_GET_ITEM(output, batch);

        for (Py_ssize_t col = 0; col < output_size; ++col) {
            double bias_value;
            if (!get_double(PyList_GET_ITEM(bias, col), &bias_value)) {
                PyMem_Free(result_data);
                return NULL;
            }

            if (!set_double(
                result_row,
                col,
                result_data[batch * output_size + col] + bias_value
            )) {
                PyMem_Free(result_data);
                return NULL;
            }
        }
    }

    PyMem_Free(result_data);
    Py_RETURN_NONE;
}

static PyObject *native_dense_backward(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *inputs;
    PyObject *grad_output;
    PyObject *weights;
    PyObject *grad_weights;
    PyObject *grad_bias;
    PyObject *grad_input;
    int compute_input_gradient;

    if (!PyArg_ParseTuple(
        args,
        "OOOOOOp:dense_backward",
        &inputs,
        &grad_output,
        &weights,
        &grad_weights,
        &grad_bias,
        &grad_input,
        &compute_input_gradient
    )) {
        return NULL;
    }

    Py_ssize_t batch_size;
    Py_ssize_t input_size;
    Py_ssize_t weight_rows;
    Py_ssize_t output_size;
    Py_ssize_t grad_weight_rows;
    Py_ssize_t grad_weight_cols;
    Py_ssize_t grad_input_rows;
    Py_ssize_t grad_input_cols;
    Py_ssize_t grad_output_rows;
    Py_ssize_t grad_output_cols;

    if (!matrix_shape(inputs, &batch_size, &input_size) ||
        !matrix_shape(weights, &weight_rows, &output_size) ||
        !matrix_shape(grad_output, &grad_output_rows, &grad_output_cols) ||
        !matrix_shape(grad_weights, &grad_weight_rows, &grad_weight_cols) ||
        !matrix_shape(grad_input, &grad_input_rows, &grad_input_cols)) {
        return NULL;
    }

    if (!PyList_Check(grad_bias) ||
        weight_rows != input_size ||
        grad_output_rows != batch_size ||
        grad_output_cols != output_size ||
        grad_weight_rows != input_size ||
        grad_weight_cols != output_size ||
        grad_input_rows != batch_size ||
        grad_input_cols != input_size ||
        PyList_GET_SIZE(grad_bias) != output_size) {
        PyErr_SetString(PyExc_ValueError, "Dense backward shapes are incompatible");
        return NULL;
    }

    size_t weight_count = (size_t)(input_size * output_size);
    size_t bias_count = (size_t)output_size;
    size_t input_count = (size_t)(batch_size * input_size);

    double *grad_w = PyMem_Calloc(weight_count, sizeof(double));
    double *grad_b = PyMem_Calloc(bias_count, sizeof(double));
    double *grad_x = compute_input_gradient
        ? PyMem_Calloc(input_count, sizeof(double))
        : NULL;

    if (grad_w == NULL || grad_b == NULL ||
        (compute_input_gradient && grad_x == NULL)) {
        PyMem_Free(grad_w);
        PyMem_Free(grad_b);
        PyMem_Free(grad_x);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t batch = 0; batch < batch_size; ++batch) {
        PyObject *go_row = PyList_GET_ITEM(grad_output, batch);

        for (Py_ssize_t col = 0; col < output_size; ++col) {
            double go_value;
            if (!get_double(PyList_GET_ITEM(go_row, col), &go_value)) {
                PyMem_Free(grad_w);
                PyMem_Free(grad_b);
                PyMem_Free(grad_x);
                return NULL;
            }
            grad_b[col] += go_value;
        }
    }

    for (Py_ssize_t batch = 0; batch < batch_size; ++batch) {
        PyObject *x_row = PyList_GET_ITEM(inputs, batch);
        PyObject *go_row = PyList_GET_ITEM(grad_output, batch);

        for (Py_ssize_t input_index = 0; input_index < input_size; ++input_index) {
            double x_value;
            if (!get_double(
                PyList_GET_ITEM(x_row, input_index),
                &x_value
            )) {
                PyMem_Free(grad_w);
                PyMem_Free(grad_b);
                PyMem_Free(grad_x);
                return NULL;
            }

            PyObject *weight_row = PyList_GET_ITEM(weights, input_index);

            for (Py_ssize_t col = 0; col < output_size; ++col) {
                double go_value;
                if (!get_double(
                    PyList_GET_ITEM(go_row, col),
                    &go_value
                )) {
                    PyMem_Free(grad_w);
                    PyMem_Free(grad_b);
                    PyMem_Free(grad_x);
                    return NULL;
                }

                grad_w[input_index * output_size + col] +=
                    x_value * go_value;

                if (compute_input_gradient) {
                    double weight_value;
                    if (!get_double(
                        PyList_GET_ITEM(weight_row, col),
                        &weight_value
                    )) {
                        PyMem_Free(grad_w);
                        PyMem_Free(grad_b);
                        PyMem_Free(grad_x);
                        return NULL;
                    }

                    grad_x[batch * input_size + input_index] +=
                        go_value * weight_value;
                }
            }
        }
    }

    for (Py_ssize_t input_index = 0; input_index < input_size; ++input_index) {
        PyObject *grad_row = PyList_GET_ITEM(grad_weights, input_index);
        for (Py_ssize_t col = 0; col < output_size; ++col) {
            if (!set_double(
                grad_row,
                col,
                grad_w[input_index * output_size + col]
            )) {
                PyMem_Free(grad_w);
                PyMem_Free(grad_b);
                PyMem_Free(grad_x);
                return NULL;
            }
        }
    }

    for (Py_ssize_t col = 0; col < output_size; ++col) {
        if (!set_double(grad_bias, col, grad_b[col])) {
            PyMem_Free(grad_w);
            PyMem_Free(grad_b);
            PyMem_Free(grad_x);
            return NULL;
        }
    }

    if (compute_input_gradient) {
        for (Py_ssize_t batch = 0; batch < batch_size; ++batch) {
            PyObject *grad_row = PyList_GET_ITEM(grad_input, batch);
            for (Py_ssize_t input_index = 0; input_index < input_size; ++input_index) {
                if (!set_double(
                    grad_row,
                    input_index,
                    grad_x[batch * input_size + input_index]
                )) {
                    PyMem_Free(grad_w);
                    PyMem_Free(grad_b);
                    PyMem_Free(grad_x);
                    return NULL;
                }
            }
        }
    }

    PyMem_Free(grad_w);
    PyMem_Free(grad_b);
    PyMem_Free(grad_x);

    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"matmul", native_matmul, METH_VARARGS, "Matrix multiplication."},
    {"dense_forward", native_dense_forward, METH_VARARGS, "Dense forward kernel."},
    {"dense_backward", native_dense_backward, METH_VARARGS, "Dense backward kernel."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "SkeAI native numerical kernels.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
