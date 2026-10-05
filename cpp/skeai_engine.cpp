#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace {

struct StorageObject {
    PyObject_HEAD
    std::vector<double> values;
};

PyTypeObject StorageType = { PyVarObject_HEAD_INIT(nullptr, 0) };

bool is_storage(PyObject* object) {
    return PyObject_TypeCheck(object, &StorageType) != 0;
}

StorageObject* as_storage(PyObject* object) {
    if (!is_storage(object)) {
        PyErr_SetString(PyExc_TypeError, "expected a SkeAI C++ Storage object");
        return nullptr;
    }
    return reinterpret_cast<StorageObject*>(object);
}

StorageObject* create_storage(std::size_t size) {
    auto* object = reinterpret_cast<StorageObject*>(
        StorageType.tp_alloc(&StorageType, 0)
    );
    if (object == nullptr) {
        return nullptr;
    }

    try {
        new (&object->values) std::vector<double>(size);
    } catch (const std::bad_alloc&) {
        StorageType.tp_free(reinterpret_cast<PyObject*>(object));
        PyErr_NoMemory();
        return nullptr;
    }

    return object;
}

PyObject* storage_new(
    PyTypeObject* type,
    PyObject*,
    PyObject*
) {
    auto* object = reinterpret_cast<StorageObject*>(
        type->tp_alloc(type, 0)
    );
    if (object == nullptr) {
        return nullptr;
    }

    try {
        new (&object->values) std::vector<double>();
    } catch (const std::bad_alloc&) {
        type->tp_free(reinterpret_cast<PyObject*>(object));
        PyErr_NoMemory();
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(object);
}

void storage_dealloc(StorageObject* self) {
    self->values.~vector<double>();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* storage_size(StorageObject* self, void*) {
    return PyLong_FromSize_t(self->values.size());
}

PyObject* storage_to_list(StorageObject* self, PyObject*) {
    PyObject* output = PyList_New(
        static_cast<Py_ssize_t>(self->values.size())
    );
    if (output == nullptr) {
        return nullptr;
    }

    for (std::size_t index = 0; index < self->values.size(); ++index) {
        PyObject* value = PyFloat_FromDouble(self->values[index]);
        if (value == nullptr) {
            Py_DECREF(output);
            return nullptr;
        }
        PyList_SET_ITEM(output, static_cast<Py_ssize_t>(index), value);
    }

    return output;
}

PyObject* storage_copy(StorageObject* self, PyObject*) {
    StorageObject* output = create_storage(self->values.size());
    if (output == nullptr) {
        return nullptr;
    }

    try {
        output->values = self->values;
    } catch (const std::bad_alloc&) {
        Py_DECREF(reinterpret_cast<PyObject*>(output));
        PyErr_NoMemory();
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(output);
}

bool checked_product(
    Py_ssize_t left,
    Py_ssize_t right,
    std::size_t* result
) {
    if (left < 0 || right < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "tensor dimensions cannot be negative"
        );
        return false;
    }

    const auto a = static_cast<std::size_t>(left);
    const auto b = static_cast<std::size_t>(right);

    if (a != 0 &&
        b > std::numeric_limits<std::size_t>::max() / a) {
        PyErr_SetString(
            PyExc_OverflowError,
            "tensor size is too large"
        );
        return false;
    }

    *result = a * b;
    return true;
}

bool same_size(
    StorageObject* left,
    StorageObject* right,
    const char* message
) {
    if (left->values.size() != right->values.size()) {
        PyErr_SetString(PyExc_ValueError, message);
        return false;
    }
    return true;
}

bool same_three_sizes(
    StorageObject* first,
    StorageObject* second,
    StorageObject* third,
    const char* message
) {
    if (first->values.size() != second->values.size() ||
        first->values.size() != third->values.size()) {
        PyErr_SetString(PyExc_ValueError, message);
        return false;
    }
    return true;
}

PyObject* storage_from_flat(PyObject*, PyObject* args) {
    PyObject* values_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "O:storage_from_flat",
        &values_object
    )) {
        return nullptr;
    }

    PyObject* sequence = PySequence_Fast(
        values_object,
        "values must be a finite numeric sequence"
    );
    if (sequence == nullptr) {
        return nullptr;
    }

    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    StorageObject* storage = create_storage(
        static_cast<std::size_t>(size)
    );
    if (storage == nullptr) {
        Py_DECREF(sequence);
        return nullptr;
    }

    PyObject** items = PySequence_Fast_ITEMS(sequence);

    for (Py_ssize_t index = 0; index < size; ++index) {
        const double value = PyFloat_AsDouble(items[index]);

        if (PyErr_Occurred() != nullptr) {
            Py_DECREF(sequence);
            Py_DECREF(reinterpret_cast<PyObject*>(storage));
            return nullptr;
        }

        if (!std::isfinite(value)) {
            Py_DECREF(sequence);
            Py_DECREF(reinterpret_cast<PyObject*>(storage));
            PyErr_SetString(
                PyExc_ValueError,
                "tensor values must be finite"
            );
            return nullptr;
        }

        storage->values[static_cast<std::size_t>(index)] = value;
    }

    Py_DECREF(sequence);
    return reinterpret_cast<PyObject*>(storage);
}

PyObject* storage_zeros(PyObject*, PyObject* args) {
    Py_ssize_t size;

    if (!PyArg_ParseTuple(
        args,
        "n:storage_zeros",
        &size
    )) {
        return nullptr;
    }

    if (size < 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "storage size cannot be negative"
        );
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(
        create_storage(static_cast<std::size_t>(size))
    );
}

#define DEFINE_BINARY_KERNEL(NAME, EXPRESSION) \
PyObject* cpp_##NAME(PyObject*, PyObject* args) { \
    PyObject* left_object = nullptr; \
    PyObject* right_object = nullptr; \
    if (!PyArg_ParseTuple(args, "OO:" #NAME, &left_object, &right_object)) { \
        return nullptr; \
    } \
    StorageObject* left = as_storage(left_object); \
    StorageObject* right = as_storage(right_object); \
    if (left == nullptr || right == nullptr) { \
        return nullptr; \
    } \
    if (!same_size(left, right, "tensor storage sizes must match")) { \
        return nullptr; \
    } \
    StorageObject* output = create_storage(left->values.size()); \
    if (output == nullptr) { \
        return nullptr; \
    } \
    const std::size_t size = left->values.size(); \
    Py_BEGIN_ALLOW_THREADS \
    for (std::size_t index = 0; index < size; ++index) { \
        output->values[index] = (EXPRESSION); \
    } \
    Py_END_ALLOW_THREADS \
    return reinterpret_cast<PyObject*>(output); \
}

DEFINE_BINARY_KERNEL(add, left->values[index] + right->values[index])
DEFINE_BINARY_KERNEL(subtract, left->values[index] - right->values[index])
DEFINE_BINARY_KERNEL(multiply, left->values[index] * right->values[index])

#undef DEFINE_BINARY_KERNEL

PyObject* cpp_scalar_add(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    double scalar;
    if (!PyArg_ParseTuple(args, "Od:scalar_add", &input_object, &scalar)) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    StorageObject* output = create_storage(input->values.size());
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t size = input->values.size();
    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] = input->values[index] + scalar;
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_scalar_subtract(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    double scalar;
    if (!PyArg_ParseTuple(
        args,
        "Od:scalar_subtract",
        &input_object,
        &scalar
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    StorageObject* output = create_storage(input->values.size());
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t size = input->values.size();
    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] = input->values[index] - scalar;
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_scalar_reverse_subtract(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    double scalar;
    if (!PyArg_ParseTuple(
        args,
        "Od:scalar_reverse_subtract",
        &input_object,
        &scalar
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    StorageObject* output = create_storage(input->values.size());
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t size = input->values.size();
    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] = scalar - input->values[index];
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_scalar_multiply(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    double scalar;
    if (!PyArg_ParseTuple(
        args,
        "Od:scalar_multiply",
        &input_object,
        &scalar
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    StorageObject* output = create_storage(input->values.size());
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t size = input->values.size();
    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] = input->values[index] * scalar;
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_scalar_divide(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    double scalar;
    if (!PyArg_ParseTuple(
        args,
        "Od:scalar_divide",
        &input_object,
        &scalar
    )) {
        return nullptr;
    }

    if (scalar == 0.0) {
        PyErr_SetString(
            PyExc_ZeroDivisionError,
            "cannot divide tensor by zero"
        );
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    StorageObject* output = create_storage(input->values.size());
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t size = input->values.size();
    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] = input->values[index] / scalar;
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_matmul(PyObject*, PyObject* args) {
    PyObject* left_object = nullptr;
    PyObject* right_object = nullptr;
    Py_ssize_t rows;
    Py_ssize_t inner;
    Py_ssize_t right_rows;
    Py_ssize_t cols;

    if (!PyArg_ParseTuple(
        args,
        "OOnnnn:matmul",
        &left_object,
        &right_object,
        &rows,
        &inner,
        &right_rows,
        &cols
    )) {
        return nullptr;
    }

    StorageObject* left = as_storage(left_object);
    StorageObject* right = as_storage(right_object);

    if (left == nullptr || right == nullptr) {
        return nullptr;
    }

    if (inner != right_rows) {
        PyErr_SetString(
            PyExc_ValueError,
            "matmul dimensions are incompatible"
        );
        return nullptr;
    }

    std::size_t left_size;
    std::size_t right_size;
    std::size_t result_size;

    if (!checked_product(rows, inner, &left_size) ||
        !checked_product(right_rows, cols, &right_size) ||
        !checked_product(rows, cols, &result_size)) {
        return nullptr;
    }

    if (left->values.size() != left_size ||
        right->values.size() != right_size) {
        PyErr_SetString(
            PyExc_ValueError,
            "matmul storage sizes do not match their shapes"
        );
        return nullptr;
    }

    StorageObject* output = create_storage(result_size);
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t rows_u = static_cast<std::size_t>(rows);
    const std::size_t inner_u = static_cast<std::size_t>(inner);
    const std::size_t cols_u = static_cast<std::size_t>(cols);

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t row = 0; row < rows_u; ++row) {
        const double* left_row =
            left->values.data() + row * inner_u;
        double* output_row =
            output->values.data() + row * cols_u;

        for (std::size_t k = 0; k < inner_u; ++k) {
            const double value = left_row[k];
            const double* right_row =
                right->values.data() + k * cols_u;

            for (std::size_t column = 0; column < cols_u; ++column) {
                output_row[column] +=
                    value * right_row[column];
            }
        }
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_transpose(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    Py_ssize_t rows;
    Py_ssize_t cols;

    if (!PyArg_ParseTuple(
        args,
        "Onn:transpose",
        &input_object,
        &rows,
        &cols
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    std::size_t count;
    if (!checked_product(rows, cols, &count)) {
        return nullptr;
    }

    if (input->values.size() != count) {
        PyErr_SetString(
            PyExc_ValueError,
            "transpose storage size does not match its shape"
        );
        return nullptr;
    }

    StorageObject* output = create_storage(count);
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t rows_u = static_cast<std::size_t>(rows);
    const std::size_t cols_u = static_cast<std::size_t>(cols);

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t row = 0; row < rows_u; ++row) {
        for (std::size_t column = 0; column < cols_u; ++column) {
            output->values[column * rows_u + row] =
                input->values[row * cols_u + column];
        }
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

bool validate_dense_forward(
    StorageObject* inputs,
    StorageObject* weights,
    StorageObject* bias,
    StorageObject* output,
    Py_ssize_t batch,
    Py_ssize_t input_size,
    Py_ssize_t output_size
) {
    std::size_t input_count;
    std::size_t weight_count;
    std::size_t output_count;

    if (!checked_product(batch, input_size, &input_count) ||
        !checked_product(input_size, output_size, &weight_count) ||
        !checked_product(batch, output_size, &output_count)) {
        return false;
    }

    if (inputs->values.size() != input_count ||
        weights->values.size() != weight_count ||
        bias->values.size() != static_cast<std::size_t>(output_size) ||
        output->values.size() != output_count) {
        PyErr_SetString(
            PyExc_ValueError,
            "Dense forward buffers do not match the provided shapes"
        );
        return false;
    }

    return true;
}

PyObject* cpp_dense_forward(PyObject*, PyObject* args) {
    PyObject* inputs_object = nullptr;
    PyObject* weights_object = nullptr;
    PyObject* bias_object = nullptr;
    PyObject* output_object = nullptr;
    Py_ssize_t batch;
    Py_ssize_t input_size;
    Py_ssize_t output_size;

    if (!PyArg_ParseTuple(
        args,
        "OOOOnnn:dense_forward",
        &inputs_object,
        &weights_object,
        &bias_object,
        &output_object,
        &batch,
        &input_size,
        &output_size
    )) {
        return nullptr;
    }

    StorageObject* inputs = as_storage(inputs_object);
    StorageObject* weights = as_storage(weights_object);
    StorageObject* bias = as_storage(bias_object);
    StorageObject* output = as_storage(output_object);

    if (inputs == nullptr || weights == nullptr ||
        bias == nullptr || output == nullptr) {
        return nullptr;
    }

    if (!validate_dense_forward(
        inputs,
        weights,
        bias,
        output,
        batch,
        input_size,
        output_size
    )) {
        return nullptr;
    }

    const std::size_t batch_u = static_cast<std::size_t>(batch);
    const std::size_t input_u = static_cast<std::size_t>(input_size);
    const std::size_t output_u = static_cast<std::size_t>(output_size);

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t row = 0; row < batch_u; ++row) {
        const double* input_row =
            inputs->values.data() + row * input_u;
        double* output_row =
            output->values.data() + row * output_u;

        std::copy(
            bias->values.begin(),
            bias->values.end(),
            output_row
        );

        for (std::size_t input_index = 0;
             input_index < input_u;
             ++input_index) {
            const double value = input_row[input_index];
            const double* weight_row =
                weights->values.data() +
                input_index * output_u;

            for (std::size_t column = 0;
                 column < output_u;
                 ++column) {
                output_row[column] +=
                    value * weight_row[column];
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_dense_backward(PyObject*, PyObject* args) {
    PyObject* inputs_object = nullptr;
    PyObject* grad_output_object = nullptr;
    PyObject* weights_object = nullptr;
    PyObject* grad_weights_object = nullptr;
    PyObject* grad_bias_object = nullptr;
    PyObject* grad_input_object = nullptr;
    int compute_input_gradient;
    Py_ssize_t batch;
    Py_ssize_t input_size;
    Py_ssize_t output_size;

    if (!PyArg_ParseTuple(
        args,
        "OOOOOOpnnn:dense_backward",
        &inputs_object,
        &grad_output_object,
        &weights_object,
        &grad_weights_object,
        &grad_bias_object,
        &grad_input_object,
        &compute_input_gradient,
        &batch,
        &input_size,
        &output_size
    )) {
        return nullptr;
    }

    StorageObject* inputs = as_storage(inputs_object);
    StorageObject* grad_output = as_storage(grad_output_object);
    StorageObject* weights = as_storage(weights_object);
    StorageObject* grad_weights = as_storage(grad_weights_object);
    StorageObject* grad_bias = as_storage(grad_bias_object);
    StorageObject* grad_input = as_storage(grad_input_object);

    if (inputs == nullptr || grad_output == nullptr ||
        weights == nullptr || grad_weights == nullptr ||
        grad_bias == nullptr || grad_input == nullptr) {
        return nullptr;
    }

    std::size_t input_count;
    std::size_t weight_count;
    std::size_t output_count;

    if (!checked_product(batch, input_size, &input_count) ||
        !checked_product(input_size, output_size, &weight_count) ||
        !checked_product(batch, output_size, &output_count)) {
        return nullptr;
    }

    if (inputs->values.size() != input_count ||
        weights->values.size() != weight_count ||
        grad_output->values.size() != output_count ||
        grad_weights->values.size() != weight_count ||
        grad_bias->values.size() != static_cast<std::size_t>(output_size) ||
        grad_input->values.size() != input_count) {
        PyErr_SetString(
            PyExc_ValueError,
            "Dense backward buffers do not match the provided shapes"
        );
        return nullptr;
    }

    const std::size_t batch_u = static_cast<std::size_t>(batch);
    const std::size_t input_u = static_cast<std::size_t>(input_size);
    const std::size_t output_u = static_cast<std::size_t>(output_size);

    Py_BEGIN_ALLOW_THREADS

    std::fill(
        grad_weights->values.begin(),
        grad_weights->values.end(),
        0.0
    );
    std::fill(
        grad_bias->values.begin(),
        grad_bias->values.end(),
        0.0
    );

    if (compute_input_gradient) {
        std::fill(
            grad_input->values.begin(),
            grad_input->values.end(),
            0.0
        );
    }

    for (std::size_t row = 0; row < batch_u; ++row) {
        const double* grad_row =
            grad_output->values.data() +
            row * output_u;

        for (std::size_t column = 0;
             column < output_u;
             ++column) {
            grad_bias->values[column] +=
                grad_row[column];
        }
    }

    for (std::size_t row = 0; row < batch_u; ++row) {
        const double* input_row =
            inputs->values.data() +
            row * input_u;
        const double* grad_row =
            grad_output->values.data() +
            row * output_u;
        double* grad_input_row =
            grad_input->values.data() +
            row * input_u;

        for (std::size_t input_index = 0;
             input_index < input_u;
             ++input_index) {
            const double input_value =
                input_row[input_index];

            double* grad_weight_row =
                grad_weights->values.data() +
                input_index * output_u;

            for (std::size_t column = 0;
                 column < output_u;
                 ++column) {
                const double grad_value =
                    grad_row[column];

                grad_weight_row[column] +=
                    input_value * grad_value;

                if (compute_input_gradient) {
                    grad_input_row[input_index] +=
                        grad_value *
                        weights->values[
                            input_index * output_u +
                            column
                        ];
                }
            }
        }
    }

    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_relu_forward(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    PyObject* output_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "OO:relu_forward",
        &input_object,
        &output_object
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    StorageObject* output = as_storage(output_object);

    if (input == nullptr || output == nullptr) {
        return nullptr;
    }

    if (!same_size(
        input,
        output,
        "ReLU buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size = input->values.size();

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        const double value = input->values[index];
        output->values[index] =
            value > 0.0 ? value : 0.0;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_relu_backward(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    PyObject* grad_object = nullptr;
    PyObject* output_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "OOO:relu_backward",
        &input_object,
        &grad_object,
        &output_object
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    StorageObject* gradient = as_storage(grad_object);
    StorageObject* output = as_storage(output_object);

    if (input == nullptr || gradient == nullptr ||
        output == nullptr) {
        return nullptr;
    }

    if (!same_three_sizes(
        input,
        gradient,
        output,
        "ReLU backward buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size = input->values.size();

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] =
            input->values[index] > 0.0
                ? gradient->values[index]
                : 0.0;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_tanh_forward(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    PyObject* output_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "OO:tanh_forward",
        &input_object,
        &output_object
    )) {
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    StorageObject* output = as_storage(output_object);

    if (input == nullptr || output == nullptr) {
        return nullptr;
    }

    if (!same_size(
        input,
        output,
        "Tanh buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size = input->values.size();

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        output->values[index] =
            std::tanh(input->values[index]);
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_tanh_backward(PyObject*, PyObject* args) {
    PyObject* cached_output_object = nullptr;
    PyObject* grad_object = nullptr;
    PyObject* output_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "OOO:tanh_backward",
        &cached_output_object,
        &grad_object,
        &output_object
    )) {
        return nullptr;
    }

    StorageObject* cached_output =
        as_storage(cached_output_object);
    StorageObject* gradient =
        as_storage(grad_object);
    StorageObject* output =
        as_storage(output_object);

    if (cached_output == nullptr ||
        gradient == nullptr ||
        output == nullptr) {
        return nullptr;
    }

    if (!same_three_sizes(
        cached_output,
        gradient,
        output,
        "Tanh backward buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size =
        cached_output->values.size();

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        const double value =
            cached_output->values[index];

        output->values[index] =
            gradient->values[index] *
            (1.0 - value * value);
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_mse_forward(PyObject*, PyObject* args) {
    PyObject* predictions_object = nullptr;
    PyObject* targets_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "OO:mse_forward",
        &predictions_object,
        &targets_object
    )) {
        return nullptr;
    }

    StorageObject* predictions =
        as_storage(predictions_object);
    StorageObject* targets =
        as_storage(targets_object);

    if (predictions == nullptr ||
        targets == nullptr) {
        return nullptr;
    }

    if (!same_size(
        predictions,
        targets,
        "MSE buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size =
        predictions->values.size();

    if (size == 0) {
        return PyFloat_FromDouble(0.0);
    }

    double total = 0.0;

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        const double delta =
            predictions->values[index] -
            targets->values[index];
        total += delta * delta;
    }
    Py_END_ALLOW_THREADS

    return PyFloat_FromDouble(
        total / static_cast<double>(size)
    );
}

PyObject* cpp_mse_backward(PyObject*, PyObject* args) {
    PyObject* predictions_object = nullptr;
    PyObject* targets_object = nullptr;
    PyObject* gradient_object = nullptr;

    if (!PyArg_ParseTuple(
        args,
        "OOO:mse_backward",
        &predictions_object,
        &targets_object,
        &gradient_object
    )) {
        return nullptr;
    }

    StorageObject* predictions =
        as_storage(predictions_object);
    StorageObject* targets =
        as_storage(targets_object);
    StorageObject* gradient =
        as_storage(gradient_object);

    if (predictions == nullptr ||
        targets == nullptr ||
        gradient == nullptr) {
        return nullptr;
    }

    if (!same_three_sizes(
        predictions,
        targets,
        gradient,
        "MSE backward buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size =
        predictions->values.size();

    if (size == 0) {
        Py_RETURN_NONE;
    }

    const double scale =
        2.0 / static_cast<double>(size);

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0; index < size; ++index) {
        gradient->values[index] =
            (predictions->values[index] -
             targets->values[index]) *
            scale;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_cross_entropy_forward(
    PyObject*,
    PyObject* args
) {
    PyObject* logits_object = nullptr;
    PyObject* targets_object = nullptr;
    PyObject* gradient_object = nullptr;
    Py_ssize_t batch;
    Py_ssize_t classes;

    if (!PyArg_ParseTuple(
        args,
        "OOOnn:cross_entropy_forward",
        &logits_object,
        &targets_object,
        &gradient_object,
        &batch,
        &classes
    )) {
        return nullptr;
    }

    if (batch < 0 || classes <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "cross-entropy dimensions are invalid"
        );
        return nullptr;
    }

    StorageObject* logits =
        as_storage(logits_object);
    StorageObject* gradient =
        as_storage(gradient_object);

    if (logits == nullptr ||
        gradient == nullptr) {
        return nullptr;
    }

    std::size_t total_count;

    if (!checked_product(
        batch,
        classes,
        &total_count
    )) {
        return nullptr;
    }

    if (logits->values.size() != total_count ||
        gradient->values.size() != total_count) {
        PyErr_SetString(
            PyExc_ValueError,
            "Cross-entropy buffers do not match the provided shape"
        );
        return nullptr;
    }

    PyObject* sequence = PySequence_Fast(
        targets_object,
        "targets must be a sequence of integer class indices"
    );
    if (sequence == nullptr) {
        return nullptr;
    }

    if (PySequence_Fast_GET_SIZE(sequence) != batch) {
        Py_DECREF(sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "number of targets must match the batch size"
        );
        return nullptr;
    }

    std::vector<Py_ssize_t> targets;

    try {
        targets.resize(
            static_cast<std::size_t>(batch)
        );
    } catch (const std::bad_alloc&) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return nullptr;
    }

    PyObject** items =
        PySequence_Fast_ITEMS(sequence);

    for (Py_ssize_t index = 0;
         index < batch;
         ++index) {
        if (!PyLong_Check(items[index])) {
            Py_DECREF(sequence);
            PyErr_SetString(
                PyExc_TypeError,
                "class targets must be integers"
            );
            return nullptr;
        }

        const Py_ssize_t target =
            PyLong_AsSsize_t(items[index]);

        if (PyErr_Occurred() != nullptr) {
            Py_DECREF(sequence);
            return nullptr;
        }

        if (target < 0 || target >= classes) {
            Py_DECREF(sequence);
            PyErr_Format(
                PyExc_ValueError,
                "target class out of range: %zd",
                target
            );
            return nullptr;
        }

        targets[
            static_cast<std::size_t>(index)
        ] = target;
    }

    Py_DECREF(sequence);

    if (batch == 0) {
        return PyFloat_FromDouble(0.0);
    }

    double total_loss = 0.0;
    bool invalid_softmax = false;

    const std::size_t batch_u =
        static_cast<std::size_t>(batch);
    const std::size_t classes_u =
        static_cast<std::size_t>(classes);
    const double inverse_batch =
        1.0 / static_cast<double>(batch);

    Py_BEGIN_ALLOW_THREADS

    for (std::size_t row = 0;
         row < batch_u && !invalid_softmax;
         ++row) {
        const double* logits_row =
            logits->values.data() +
            row * classes_u;
        double* gradient_row =
            gradient->values.data() +
            row * classes_u;

        double maximum = logits_row[0];

        for (std::size_t column = 1;
             column < classes_u;
             ++column) {
            maximum = std::max(
                maximum,
                logits_row[column]
            );
        }

        double sum = 0.0;

        for (std::size_t column = 0;
             column < classes_u;
             ++column) {
            const double probability =
                std::exp(
                    logits_row[column] - maximum
                );

            gradient_row[column] =
                probability;
            sum += probability;
        }

        if (!(sum > 0.0) ||
            !std::isfinite(sum)) {
            invalid_softmax = true;
            break;
        }

        const double inverse_sum =
            1.0 / sum;

        for (std::size_t column = 0;
             column < classes_u;
             ++column) {
            gradient_row[column] *= inverse_sum;
        }

        const std::size_t target =
            static_cast<std::size_t>(
                targets[row]
            );

        const double target_probability =
            std::max(
                gradient_row[target],
                1e-12
            );

        total_loss -=
            std::log(target_probability);

        gradient_row[target] -= 1.0;

        for (std::size_t column = 0;
             column < classes_u;
             ++column) {
            gradient_row[column] *=
                inverse_batch;
        }
    }

    Py_END_ALLOW_THREADS

    if (invalid_softmax) {
        PyErr_SetString(
            PyExc_ValueError,
            "Invalid softmax normalization"
        );
        return nullptr;
    }

    return PyFloat_FromDouble(
        total_loss /
        static_cast<double>(batch)
    );
}

PyObject* cpp_sgd_step(PyObject*, PyObject* args) {
    PyObject* parameter_object = nullptr;
    PyObject* gradient_object = nullptr;
    double learning_rate;

    if (!PyArg_ParseTuple(
        args,
        "OOd:sgd_step",
        &parameter_object,
        &gradient_object,
        &learning_rate
    )) {
        return nullptr;
    }

    StorageObject* parameter =
        as_storage(parameter_object);
    StorageObject* gradient =
        as_storage(gradient_object);

    if (parameter == nullptr ||
        gradient == nullptr) {
        return nullptr;
    }

    if (!same_size(
        parameter,
        gradient,
        "parameter and gradient buffers must have the same size"
    )) {
        return nullptr;
    }

    const std::size_t size =
        parameter->values.size();

    Py_BEGIN_ALLOW_THREADS
    for (std::size_t index = 0;
         index < size;
         ++index) {
        parameter->values[index] -=
            learning_rate *
            gradient->values[index];
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyMethodDef module_methods[] = {
    {"storage_from_flat", storage_from_flat, METH_VARARGS,
     "Create contiguous C++ storage."},
    {"storage_zeros", storage_zeros, METH_VARARGS,
     "Create zero-initialized C++ storage."},
    {"add", cpp_add, METH_VARARGS,
     "Elementwise tensor addition."},
    {"subtract", cpp_subtract, METH_VARARGS,
     "Elementwise tensor subtraction."},
    {"multiply", cpp_multiply, METH_VARARGS,
     "Elementwise tensor multiplication."},
    {"scalar_add", cpp_scalar_add, METH_VARARGS,
     "Scalar tensor addition."},
    {"scalar_subtract", cpp_scalar_subtract, METH_VARARGS,
     "Scalar tensor subtraction."},
    {"scalar_reverse_subtract", cpp_scalar_reverse_subtract, METH_VARARGS,
     "Scalar minus tensor."},
    {"scalar_multiply", cpp_scalar_multiply, METH_VARARGS,
     "Scalar tensor multiplication."},
    {"scalar_divide", cpp_scalar_divide, METH_VARARGS,
     "Scalar tensor division."},
    {"matmul", cpp_matmul, METH_VARARGS,
     "Matrix multiplication."},
    {"transpose", cpp_transpose, METH_VARARGS,
     "Rank-2 transpose."},
    {"dense_forward", cpp_dense_forward, METH_VARARGS,
     "Dense forward kernel."},
    {"dense_backward", cpp_dense_backward, METH_VARARGS,
     "Dense backward kernel."},
    {"relu_forward", cpp_relu_forward, METH_VARARGS,
     "ReLU forward kernel."},
    {"relu_backward", cpp_relu_backward, METH_VARARGS,
     "ReLU backward kernel."},
    {"tanh_forward", cpp_tanh_forward, METH_VARARGS,
     "Tanh forward kernel."},
    {"tanh_backward", cpp_tanh_backward, METH_VARARGS,
     "Tanh backward kernel."},
    {"mse_forward", cpp_mse_forward, METH_VARARGS,
     "MSE forward."},
    {"mse_backward", cpp_mse_backward, METH_VARARGS,
     "MSE backward."},
    {"cross_entropy_forward", cpp_cross_entropy_forward, METH_VARARGS,
     "Cross-entropy forward and gradient."},
    {"sgd_step", cpp_sgd_step, METH_VARARGS,
     "In-place SGD step."},
    {nullptr, nullptr, 0, nullptr}
};

PyMethodDef storage_methods[] = {
    {"to_list",
     reinterpret_cast<PyCFunction>(storage_to_list),
     METH_NOARGS,
     "Copy storage to a flat Python list."},
    {"copy",
     reinterpret_cast<PyCFunction>(storage_copy),
     METH_NOARGS,
     "Copy storage."},
    {nullptr, nullptr, 0, nullptr}
};

PyGetSetDef storage_getset[] = {
    {"size",
     reinterpret_cast<getter>(storage_size),
     nullptr,
     "Number of scalar values.",
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_cpp",
    "SkeAI C++ numerical engine.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__cpp() {
    StorageType.tp_name = "skeai._cpp.Storage";
    StorageType.tp_basicsize = sizeof(StorageObject);
    StorageType.tp_itemsize = 0;
    StorageType.tp_dealloc =
        reinterpret_cast<destructor>(storage_dealloc);
    StorageType.tp_flags = Py_TPFLAGS_DEFAULT;
    StorageType.tp_doc =
        "Contiguous C++-owned SkeAI tensor storage.";
    StorageType.tp_methods = storage_methods;
    StorageType.tp_getset = storage_getset;
    StorageType.tp_new = storage_new;

    if (PyType_Ready(&StorageType) < 0) {
        return nullptr;
    }

    PyObject* module =
        PyModule_Create(&module_definition);

    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(&StorageType);

    if (PyModule_AddObject(
        module,
        "Storage",
        reinterpret_cast<PyObject*>(&StorageType)
    ) < 0) {
        Py_DECREF(&StorageType);
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}
