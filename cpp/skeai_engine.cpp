#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory_resource>
#include <new>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace {

thread_local std::pmr::unsynchronized_pool_resource g_train_memory_pool;

inline std::pmr::memory_resource* train_memory_resource() noexcept {
    return &g_train_memory_pool;
}

using TrainBuffer = std::pmr::vector<double>;

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


PyObject* cpp_dense_indexed_forward(PyObject*, PyObject* args) {
    PyObject* indices_object = nullptr;
    PyObject* weights_object = nullptr;
    PyObject* bias_object = nullptr;
    PyObject* output_object = nullptr;
    int batch;
    int input_size;
    int output_size;

    if (!PyArg_ParseTuple(
        args,
        "OOOOiii:dense_indexed_forward",
        &indices_object,
        &weights_object,
        &bias_object,
        &output_object,
        &batch,
        &input_size,
        &output_size
    )) {
        return nullptr;
    }

    if (batch <= 0 || input_size <= 0 || output_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dense indexed dimensions must be positive");
        return nullptr;
    }

    StorageObject* weights = as_storage(weights_object);
    StorageObject* bias = as_storage(bias_object);
    StorageObject* output = as_storage(output_object);

    if (weights == nullptr || bias == nullptr || output == nullptr) {
        return nullptr;
    }

    std::size_t weight_count;
    std::size_t output_count;

    if (!checked_product(input_size, output_size, &weight_count) ||
        !checked_product(batch, output_size, &output_count)) {
        return nullptr;
    }

    if (weights->values.size() != weight_count ||
        bias->values.size() != static_cast<std::size_t>(output_size) ||
        output->values.size() != output_count) {
        PyErr_SetString(
            PyExc_ValueError,
            "Dense indexed buffers do not match the provided shapes"
        );
        return nullptr;
    }

    PyObject* sequence = PySequence_Fast(
        indices_object,
        "indices must be a sequence of integer feature indices"
    );
    if (sequence == nullptr) {
        return nullptr;
    }

    const Py_ssize_t index_count =
        PySequence_Fast_GET_SIZE(sequence);

    if (index_count <= 0 || index_count % batch != 0) {
        Py_DECREF(sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "Indexed input count must be a positive multiple of batch size"
        );
        return nullptr;
    }

    const std::size_t positions_per_row =
        static_cast<std::size_t>(index_count / batch);

    std::vector<std::size_t> indices;
    try {
        indices.resize(static_cast<std::size_t>(index_count));
    } catch (const std::bad_alloc&) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return nullptr;
    }

    PyObject** items = PySequence_Fast_ITEMS(sequence);

    for (Py_ssize_t index = 0; index < index_count; ++index) {
        if (!PyLong_Check(items[index])) {
            Py_DECREF(sequence);
            PyErr_SetString(
                PyExc_TypeError,
                "indexed inputs must contain integers"
            );
            return nullptr;
        }

        const Py_ssize_t feature_index =
            PyLong_AsSsize_t(items[index]);

        if (PyErr_Occurred() != nullptr) {
            Py_DECREF(sequence);
            return nullptr;
        }

        if (feature_index < 0 || feature_index >= input_size) {
            Py_DECREF(sequence);
            PyErr_Format(
                PyExc_ValueError,
                "indexed feature out of range: %zd",
                feature_index
            );
            return nullptr;
        }

        indices[static_cast<std::size_t>(index)] =
            static_cast<std::size_t>(feature_index);
    }

    Py_DECREF(sequence);

    const std::size_t batch_u = static_cast<std::size_t>(batch);
    const std::size_t output_u = static_cast<std::size_t>(output_size);

    Py_BEGIN_ALLOW_THREADS

    for (std::size_t row = 0; row < batch_u; ++row) {
        double* output_row =
            output->values.data() + row * output_u;

        for (std::size_t column = 0; column < output_u; ++column) {
            output_row[column] = bias->values[column];
        }

        const std::size_t index_base =
            row * positions_per_row;

        for (std::size_t position = 0;
             position < positions_per_row;
             ++position) {
            const std::size_t feature_index =
                indices[index_base + position];

            const double* weight_row =
                weights->values.data() +
                feature_index * output_u;

            for (std::size_t column = 0;
                 column < output_u;
                 ++column) {
                output_row[column] += weight_row[column];
            }
        }
    }

    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject* cpp_dense_indexed_backward(PyObject*, PyObject* args) {
    PyObject* indices_object = nullptr;
    PyObject* grad_output_object = nullptr;
    PyObject* grad_weights_object = nullptr;
    PyObject* grad_bias_object = nullptr;
    int batch;
    int input_size;
    int output_size;

    if (!PyArg_ParseTuple(
        args,
        "OOOOiii:dense_indexed_backward",
        &indices_object,
        &grad_output_object,
        &grad_weights_object,
        &grad_bias_object,
        &batch,
        &input_size,
        &output_size
    )) {
        return nullptr;
    }

    if (batch <= 0 || input_size <= 0 || output_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "Dense indexed dimensions must be positive");
        return nullptr;
    }

    StorageObject* grad_output = as_storage(grad_output_object);
    StorageObject* grad_weights = as_storage(grad_weights_object);
    StorageObject* grad_bias = as_storage(grad_bias_object);

    if (grad_output == nullptr ||
        grad_weights == nullptr ||
        grad_bias == nullptr) {
        return nullptr;
    }

    std::size_t weight_count;
    std::size_t output_count;

    if (!checked_product(input_size, output_size, &weight_count) ||
        !checked_product(batch, output_size, &output_count)) {
        return nullptr;
    }

    if (grad_output->values.size() != output_count ||
        grad_weights->values.size() != weight_count ||
        grad_bias->values.size() != static_cast<std::size_t>(output_size)) {
        PyErr_SetString(
            PyExc_ValueError,
            "Dense indexed gradient buffers do not match the provided shapes"
        );
        return nullptr;
    }

    PyObject* sequence = PySequence_Fast(
        indices_object,
        "indices must be a sequence of integer feature indices"
    );
    if (sequence == nullptr) {
        return nullptr;
    }

    const Py_ssize_t index_count =
        PySequence_Fast_GET_SIZE(sequence);

    if (index_count <= 0 || index_count % batch != 0) {
        Py_DECREF(sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "Indexed input count must be a positive multiple of batch size"
        );
        return nullptr;
    }

    const std::size_t positions_per_row =
        static_cast<std::size_t>(index_count / batch);

    std::vector<std::size_t> indices;
    try {
        indices.resize(static_cast<std::size_t>(index_count));
    } catch (const std::bad_alloc&) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return nullptr;
    }

    PyObject** items = PySequence_Fast_ITEMS(sequence);

    for (Py_ssize_t index = 0; index < index_count; ++index) {
        if (!PyLong_Check(items[index])) {
            Py_DECREF(sequence);
            PyErr_SetString(
                PyExc_TypeError,
                "indexed inputs must contain integers"
            );
            return nullptr;
        }

        const Py_ssize_t feature_index =
            PyLong_AsSsize_t(items[index]);

        if (PyErr_Occurred() != nullptr) {
            Py_DECREF(sequence);
            return nullptr;
        }

        if (feature_index < 0 || feature_index >= input_size) {
            Py_DECREF(sequence);
            PyErr_Format(
                PyExc_ValueError,
                "indexed feature out of range: %zd",
                feature_index
            );
            return nullptr;
        }

        indices[static_cast<std::size_t>(index)] =
            static_cast<std::size_t>(feature_index);
    }

    Py_DECREF(sequence);

    const std::size_t batch_u = static_cast<std::size_t>(batch);
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

    for (std::size_t row = 0; row < batch_u; ++row) {
        const double* grad_row =
            grad_output->values.data() + row * output_u;

        for (std::size_t column = 0;
             column < output_u;
             ++column) {
            grad_bias->values[column] += grad_row[column];
        }

        const std::size_t index_base =
            row * positions_per_row;

        for (std::size_t position = 0;
             position < positions_per_row;
             ++position) {
            const std::size_t feature_index =
                indices[index_base + position];

            double* grad_weight_row =
                grad_weights->values.data() +
                feature_index * output_u;

            for (std::size_t column = 0;
                 column < output_u;
                 ++column) {
                grad_weight_row[column] += grad_row[column];
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


PyObject* cpp_causal_softmax(PyObject*, PyObject* args) {
    PyObject* input_object = nullptr;
    Py_ssize_t rows;
    Py_ssize_t cols;

    if (!PyArg_ParseTuple(
        args,
        "Onn:causal_softmax",
        &input_object,
        &rows,
        &cols
    )) {
        return nullptr;
    }

    if (rows < 0 || cols <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "causal softmax dimensions are invalid"
        );
        return nullptr;
    }

    StorageObject* input = as_storage(input_object);
    if (input == nullptr) {
        return nullptr;
    }

    std::size_t expected_size;
    if (!checked_product(rows, cols, &expected_size)) {
        return nullptr;
    }

    if (input->values.size() != expected_size) {
        PyErr_SetString(
            PyExc_ValueError,
            "causal softmax buffer does not match the provided shape"
        );
        return nullptr;
    }

    StorageObject* output = create_storage(expected_size);
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t row_count = static_cast<std::size_t>(rows);
    const std::size_t column_count = static_cast<std::size_t>(cols);

    Py_BEGIN_ALLOW_THREADS

    for (std::size_t row = 0; row < row_count; ++row) {
        const double* input_row =
            input->values.data() + row * column_count;
        double* output_row =
            output->values.data() + row * column_count;

        const std::size_t last_allowed =
            std::min(row, column_count - 1);

        double maximum = input_row[0];
        for (std::size_t column = 1;
             column <= last_allowed;
             ++column) {
            maximum = std::max(maximum, input_row[column]);
        }

        double total = 0.0;
        for (std::size_t column = 0;
             column <= last_allowed;
             ++column) {
            const double value =
                std::exp(input_row[column] - maximum);
            output_row[column] = value;
            total += value;
        }

        const double inverse_total = 1.0 / total;

        for (std::size_t column = 0;
             column <= last_allowed;
             ++column) {
            output_row[column] *= inverse_total;
        }

        for (std::size_t column = last_allowed + 1;
             column < column_count;
             ++column) {
            output_row[column] = 0.0;
        }
    }

    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
}

PyObject* cpp_softmax_backward(PyObject*, PyObject* args) {
    PyObject* probabilities_object = nullptr;
    PyObject* gradient_object = nullptr;
    Py_ssize_t rows;
    Py_ssize_t cols;

    if (!PyArg_ParseTuple(
        args,
        "OOnn:softmax_backward",
        &probabilities_object,
        &gradient_object,
        &rows,
        &cols
    )) {
        return nullptr;
    }

    if (rows < 0 || cols <= 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "softmax backward dimensions are invalid"
        );
        return nullptr;
    }

    StorageObject* probabilities =
        as_storage(probabilities_object);
    StorageObject* gradient =
        as_storage(gradient_object);

    if (probabilities == nullptr || gradient == nullptr) {
        return nullptr;
    }

    std::size_t expected_size;
    if (!checked_product(rows, cols, &expected_size)) {
        return nullptr;
    }

    if (probabilities->values.size() != expected_size ||
        gradient->values.size() != expected_size) {
        PyErr_SetString(
            PyExc_ValueError,
            "softmax backward buffers do not match the provided shape"
        );
        return nullptr;
    }

    StorageObject* output = create_storage(expected_size);
    if (output == nullptr) {
        return nullptr;
    }

    const std::size_t row_count = static_cast<std::size_t>(rows);
    const std::size_t column_count = static_cast<std::size_t>(cols);

    Py_BEGIN_ALLOW_THREADS

    for (std::size_t row = 0; row < row_count; ++row) {
        const double* probability_row =
            probabilities->values.data() + row * column_count;
        const double* gradient_row =
            gradient->values.data() + row * column_count;
        double* output_row =
            output->values.data() + row * column_count;

        double dot = 0.0;
        for (std::size_t column = 0;
             column < column_count;
             ++column) {
            dot += probability_row[column] * gradient_row[column];
        }

        for (std::size_t column = 0;
             column < column_count;
             ++column) {
            output_row[column] =
                probability_row[column] *
                (gradient_row[column] - dot);
        }
    }

    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(output);
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
    double weight_decay = 0.0;

    if (!PyArg_ParseTuple(
        args,
        "OOd|d:sgd_step",
        &parameter_object,
        &gradient_object,
        &learning_rate,
        &weight_decay
    )) {
        return nullptr;
    }

    if (!std::isfinite(learning_rate) ||
        learning_rate <= 0.0) {
        PyErr_SetString(
            PyExc_ValueError,
            "learning rate must be finite and greater than zero"
        );
        return nullptr;
    }

    if (!std::isfinite(weight_decay) ||
        weight_decay < 0.0) {
        PyErr_SetString(
            PyExc_ValueError,
            "weight decay must be finite and non-negative"
        );
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
            (gradient->values[index] +
             weight_decay * parameter->values[index]);
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}



struct TrainMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    TrainBuffer values;

    TrainMatrix()
        : values(train_memory_resource()) {}

    TrainMatrix(std::size_t rows_, std::size_t cols_)
        : rows(rows_),
          cols(cols_),
          values(rows_ * cols_, 0.0, train_memory_resource()) {}

    double* row_ptr(std::size_t row) {
        return values.data() + row * cols;
    }

    const double* row_ptr(std::size_t row) const {
        return values.data() + row * cols;
    }

    double& at(std::size_t row, std::size_t col) {
        return values[row * cols + col];
    }

    const double& at(std::size_t row, std::size_t col) const {
        return values[row * cols + col];
    }
};

struct TrainLayerCache {
    TrainMatrix input;
    TrainMatrix norm_x;
    TrainBuffer mean1;
    TrainBuffer inv1;
    TrainMatrix q;
    TrainMatrix k;
    TrainMatrix v;
    std::vector<TrainMatrix> probabilities;
    TrainMatrix merged;
    TrainMatrix attention_output;
    TrainMatrix residual;
    TrainMatrix norm_residual;
    TrainBuffer mean2;
    TrainBuffer inv2;
    TrainMatrix hidden_pre;
    TrainMatrix hidden;
};

TrainMatrix train_matmul(
    const TrainMatrix& left,
    const TrainMatrix& right
) {
    if (left.cols != right.rows) {
        throw std::runtime_error("incompatible training matmul");
    }

    TrainMatrix output(left.rows, right.cols);

    for (std::size_t row = 0; row < left.rows; ++row) {
        const double* left_row = left.row_ptr(row);
        double* output_row = output.row_ptr(row);

        for (std::size_t inner = 0; inner < left.cols; ++inner) {
            const double value = left_row[inner];
            const double* right_row = right.row_ptr(inner);

            for (std::size_t column = 0;
                 column < right.cols;
                 ++column) {
                output_row[column] +=
                    value * right_row[column];
            }
        }
    }

    return output;
}

TrainMatrix train_transpose(const TrainMatrix& input) {
    TrainMatrix output(input.cols, input.rows);

    for (std::size_t row = 0; row < input.rows; ++row) {
        const double* input_row = input.row_ptr(row);
        for (std::size_t column = 0;
             column < input.cols;
             ++column) {
            output.at(column, row) = input_row[column];
        }
    }

    return output;
}

TrainMatrix train_add(
    const TrainMatrix& left,
    const TrainMatrix& right
) {
    if (left.rows != right.rows || left.cols != right.cols) {
        throw std::runtime_error("incompatible training add");
    }

    TrainMatrix output(left.rows, left.cols);
    for (std::size_t index = 0;
         index < output.values.size();
         ++index) {
        output.values[index] =
            left.values[index] +
            right.values[index];
    }
    return output;
}

void train_layer_norm_forward(
    const TrainMatrix& input,
    TrainMatrix& output,
    TrainBuffer& means,
    TrainBuffer& inv_stds
) {
    output = TrainMatrix(input.rows, input.cols);
    means.assign(input.rows, 0.0);
    inv_stds.assign(input.rows, 0.0);

    const double eps = 1e-5;

    for (std::size_t row = 0; row < input.rows; ++row) {
        const double* input_row = input.row_ptr(row);
        double* output_row = output.row_ptr(row);

        double mean = 0.0;
        for (std::size_t column = 0; column < input.cols; ++column) {
            mean += input_row[column];
        }
        mean /= static_cast<double>(input.cols);

        double variance = 0.0;
        for (std::size_t column = 0; column < input.cols; ++column) {
            const double centered = input_row[column] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(input.cols);

        const double inv_std = 1.0 / std::sqrt(variance + eps);
        means[row] = mean;
        inv_stds[row] = inv_std;

        for (std::size_t column = 0; column < input.cols; ++column) {
            output_row[column] =
                (input_row[column] - mean) * inv_std;
        }
    }
}

TrainMatrix train_layer_norm_backward(
    const TrainMatrix& gradient,
    const TrainMatrix& input,
    const TrainBuffer& means,
    const TrainBuffer& inv_stds
) {
    if (gradient.rows != input.rows ||
        gradient.cols != input.cols ||
        means.size() != input.rows ||
        inv_stds.size() != input.rows) {
        throw std::runtime_error("invalid layer norm backward cache");
    }

    TrainMatrix output(input.rows, input.cols);
    const double width = static_cast<double>(input.cols);

    for (std::size_t row = 0; row < input.rows; ++row) {
        const double* gradient_row = gradient.row_ptr(row);
        const double* input_row = input.row_ptr(row);
        double* output_row = output.row_ptr(row);

        double sum_gradient = 0.0;
        double sum_gradient_xhat = 0.0;

        for (std::size_t column = 0;
             column < input.cols;
             ++column) {
            const double xhat =
                (input_row[column] - means[row]) *
                inv_stds[row];
            sum_gradient += gradient_row[column];
            sum_gradient_xhat +=
                gradient_row[column] * xhat;
        }

        for (std::size_t column = 0;
             column < input.cols;
             ++column) {
            const double xhat =
                (input_row[column] - means[row]) *
                inv_stds[row];

            output_row[column] =
                (inv_stds[row] / width) *
                (
                    width * gradient_row[column]
                    - sum_gradient
                    - xhat * sum_gradient_xhat
                );
        }
    }

    return output;
}

TrainMatrix train_causal_softmax(
    const TrainMatrix& scores
) {
    TrainMatrix output(scores.rows, scores.cols);

    for (std::size_t row = 0; row < scores.rows; ++row) {
        const double* score_row = scores.row_ptr(row);
        double* output_row = output.row_ptr(row);

        const std::size_t last_allowed =
            std::min(row, scores.cols - 1);

        double maximum = score_row[0];
        for (std::size_t column = 1;
             column <= last_allowed;
             ++column) {
            maximum = std::max(maximum, score_row[column]);
        }

        double total = 0.0;
        for (std::size_t column = 0;
             column <= last_allowed;
             ++column) {
            const double value =
                std::exp(score_row[column] - maximum);
            output_row[column] = value;
            total += value;
        }

        const double inverse_total = 1.0 / total;
        for (std::size_t column = 0;
             column <= last_allowed;
             ++column) {
            output_row[column] *= inverse_total;
        }
    }

    return output;
}

TrainMatrix train_softmax_backward(
    const TrainMatrix& probabilities,
    const TrainMatrix& gradient
) {
    if (probabilities.rows != gradient.rows ||
        probabilities.cols != gradient.cols) {
        throw std::runtime_error(
            "incompatible training softmax backward"
        );
    }

    TrainMatrix output(probabilities.rows, probabilities.cols);

    for (std::size_t row = 0;
         row < probabilities.rows;
         ++row) {
        const double* probability_row =
            probabilities.row_ptr(row);
        const double* gradient_row =
            gradient.row_ptr(row);
        double* output_row =
            output.row_ptr(row);

        double dot = 0.0;
        for (std::size_t column = 0;
             column < probabilities.cols;
             ++column) {
            dot +=
                probability_row[column] *
                gradient_row[column];
        }

        for (std::size_t column = 0;
             column < probabilities.cols;
             ++column) {
            output_row[column] =
                probability_row[column] *
                (gradient_row[column] - dot);
        }
    }

    return output;
}

TrainMatrix train_relu(
    const TrainMatrix& input
) {
    TrainMatrix output(input.rows, input.cols);

    for (std::size_t index = 0;
         index < input.values.size();
         ++index) {
        output.values[index] =
            input.values[index] > 0.0
                ? input.values[index]
                : 0.0;
    }

    return output;
}

TrainMatrix train_relu_backward(
    const TrainMatrix& gradient,
    const TrainMatrix& input
) {
    if (gradient.rows != input.rows ||
        gradient.cols != input.cols) {
        throw std::runtime_error(
            "incompatible training relu backward"
        );
    }

    TrainMatrix output(input.rows, input.cols);

    for (std::size_t index = 0;
         index < input.values.size();
         ++index) {
        output.values[index] =
            input.values[index] > 0.0
                ? gradient.values[index]
                : 0.0;
    }

    return output;
}

TrainMatrix train_slice_columns(
    const TrainMatrix& matrix,
    std::size_t start,
    std::size_t width
) {
    if (start + width > matrix.cols) {
        throw std::runtime_error("training slice out of range");
    }

    TrainMatrix output(matrix.rows, width);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        const double* input_row = matrix.row_ptr(row);
        double* output_row = output.row_ptr(row);
        for (std::size_t column = 0; column < width; ++column) {
            output_row[column] = input_row[start + column];
        }
    }
    return output;
}

void train_write_slice(
    TrainMatrix& target,
    const TrainMatrix& source,
    std::size_t start
) {
    if (target.rows != source.rows ||
        start + source.cols > target.cols) {
        throw std::runtime_error("training slice write out of range");
    }

    for (std::size_t row = 0; row < target.rows; ++row) {
        double* target_row = target.row_ptr(row);
        const double* source_row = source.row_ptr(row);
        for (std::size_t column = 0;
             column < source.cols;
             ++column) {
            target_row[start + column] = source_row[column];
        }
    }
}

void train_add_inplace(
    TrainMatrix& target,
    const TrainMatrix& source
) {
    if (target.rows != source.rows ||
        target.cols != source.cols) {
        throw std::runtime_error("training inplace add shape mismatch");
    }

    for (std::size_t index = 0;
         index < target.values.size();
         ++index) {
        target.values[index] += source.values[index];
    }
}

void train_accumulate_parameter_gradient(
    TrainBuffer& gradient,
    const TrainMatrix& value,
    double scale
) {
    if (gradient.size() != value.values.size()) {
        throw std::runtime_error("training gradient shape mismatch");
    }

    for (std::size_t index = 0;
         index < value.values.size();
         ++index) {
        gradient[index] += scale * value.values[index];
    }
}

bool train_is_finite(const TrainMatrix& matrix) {
    for (double value : matrix.values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

PyObject* cpp_transformer_train_step(PyObject*, PyObject* args) {
    PyObject* parameters_object = nullptr;
    PyObject* token_ids_object = nullptr;
    PyObject* target_ids_object = nullptr;
    Py_ssize_t vocab_size;
    Py_ssize_t context_length;
    Py_ssize_t d_model;
    Py_ssize_t n_heads;
    Py_ssize_t feed_forward_size;
    Py_ssize_t n_layers;
    Py_ssize_t sequence_length;
    double learning_rate;

    if (!PyArg_ParseTuple(
        args,
        "OOOnnnnnnnd:transformer_train_step",
        &parameters_object,
        &token_ids_object,
        &target_ids_object,
        &vocab_size,
        &context_length,
        &d_model,
        &n_heads,
        &feed_forward_size,
        &n_layers,
        &sequence_length,
        &learning_rate
    )) {
        return nullptr;
    }

    if (vocab_size <= 0 ||
        context_length <= 0 ||
        d_model <= 0 ||
        n_heads <= 0 ||
        feed_forward_size <= 0 ||
        n_layers <= 0 ||
        sequence_length <= 0 ||
        sequence_length > context_length ||
        d_model % n_heads != 0 ||
        !std::isfinite(learning_rate) ||
        learning_rate <= 0.0) {
        PyErr_SetString(
            PyExc_ValueError,
            "invalid transformer training dimensions or learning rate"
        );
        return nullptr;
    }

    PyObject* parameters_sequence = PySequence_Fast(
        parameters_object,
        "parameters must be a sequence of C++ Storage objects"
    );
    if (parameters_sequence == nullptr) {
        return nullptr;
    }

    const std::size_t layer_count =
        static_cast<std::size_t>(n_layers);
    const std::size_t expected_parameters =
        3 + 6 * layer_count;

    if (PySequence_Fast_GET_SIZE(parameters_sequence) !=
        static_cast<Py_ssize_t>(expected_parameters)) {
        Py_DECREF(parameters_sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "unexpected transformer parameter count"
        );
        return nullptr;
    }

    std::vector<StorageObject*> parameters;
    try {
        parameters.reserve(expected_parameters);
    } catch (const std::bad_alloc&) {
        Py_DECREF(parameters_sequence);
        PyErr_NoMemory();
        return nullptr;
    }

    PyObject** parameter_items =
        PySequence_Fast_ITEMS(parameters_sequence);

    for (std::size_t index = 0;
         index < expected_parameters;
         ++index) {
        StorageObject* storage =
            as_storage(parameter_items[index]);

        if (storage == nullptr) {
            Py_DECREF(parameters_sequence);
            return nullptr;
        }

        parameters.push_back(storage);
    }

    std::vector<int> token_ids;
    std::vector<int> target_ids;

    PyObject* token_sequence = PySequence_Fast(
        token_ids_object,
        "token_ids must be a sequence of integers"
    );
    if (token_sequence == nullptr) {
        Py_DECREF(parameters_sequence);
        return nullptr;
    }

    PyObject* target_sequence = PySequence_Fast(
        target_ids_object,
        "target_ids must be a sequence of integers"
    );
    if (target_sequence == nullptr) {
        Py_DECREF(token_sequence);
        Py_DECREF(parameters_sequence);
        return nullptr;
    }

    const Py_ssize_t token_count =
        PySequence_Fast_GET_SIZE(token_sequence);
    const Py_ssize_t target_count =
        PySequence_Fast_GET_SIZE(target_sequence);

    if (token_count != sequence_length ||
        target_count != sequence_length) {
        Py_DECREF(token_sequence);
        Py_DECREF(target_sequence);
        Py_DECREF(parameters_sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "token and target lengths must match sequence_length"
        );
        return nullptr;
    }

    try {
        token_ids.resize(static_cast<std::size_t>(sequence_length));
        target_ids.resize(static_cast<std::size_t>(sequence_length));
    } catch (const std::bad_alloc&) {
        Py_DECREF(token_sequence);
        Py_DECREF(target_sequence);
        Py_DECREF(parameters_sequence);
        PyErr_NoMemory();
        return nullptr;
    }

    PyObject** token_items =
        PySequence_Fast_ITEMS(token_sequence);
    PyObject** target_items =
        PySequence_Fast_ITEMS(target_sequence);

    for (Py_ssize_t index = 0;
         index < sequence_length;
         ++index) {
        if (!PyLong_Check(token_items[index]) ||
            !PyLong_Check(target_items[index])) {
            Py_DECREF(token_sequence);
            Py_DECREF(target_sequence);
            Py_DECREF(parameters_sequence);
            PyErr_SetString(
                PyExc_TypeError,
                "transformer token and target ids must be integers"
            );
            return nullptr;
        }

        const Py_ssize_t token =
            PyLong_AsSsize_t(token_items[index]);
        const Py_ssize_t target =
            PyLong_AsSsize_t(target_items[index]);

        if (PyErr_Occurred() != nullptr) {
            Py_DECREF(token_sequence);
            Py_DECREF(target_sequence);
            Py_DECREF(parameters_sequence);
            return nullptr;
        }

        if (token < 0 || token >= vocab_size ||
            target < 0 || target >= vocab_size) {
            Py_DECREF(token_sequence);
            Py_DECREF(target_sequence);
            Py_DECREF(parameters_sequence);
            PyErr_SetString(
                PyExc_ValueError,
                "transformer token or target id out of range"
            );
            return nullptr;
        }

        token_ids[static_cast<std::size_t>(index)] =
            static_cast<int>(token);
        target_ids[static_cast<std::size_t>(index)] =
            static_cast<int>(target);
    }

    Py_DECREF(token_sequence);
    Py_DECREF(target_sequence);

    const std::size_t vocab =
        static_cast<std::size_t>(vocab_size);
    const std::size_t context =
        static_cast<std::size_t>(context_length);
    const std::size_t d =
        static_cast<std::size_t>(d_model);
    const std::size_t heads =
        static_cast<std::size_t>(n_heads);
    const std::size_t ff =
        static_cast<std::size_t>(feed_forward_size);
    const std::size_t layers =
        static_cast<std::size_t>(n_layers);
    const std::size_t length =
        static_cast<std::size_t>(sequence_length);
    const std::size_t head_dim = d / heads;

    using TrainClock = std::chrono::steady_clock;
    const bool profile = []() {
        const char* value = std::getenv("SKEAI_CPP_PROFILE");
        return value != nullptr && value[0] == '1';
    }();
    const auto profile_start =
        profile ? TrainClock::now() : TrainClock::time_point{};

    double embedding_ms = 0.0;
    double normalization_ms = 0.0;
    double q_projection_ms = 0.0;
    double k_projection_ms = 0.0;
    double v_projection_ms = 0.0;
    double attention_scores_ms = 0.0;
    double attention_softmax_ms = 0.0;
    double attention_weighted_sum_ms = 0.0;
    double output_projection_ms = 0.0;
    double residual_ms = 0.0;
    double feed_forward_ms = 0.0;
    double activation_ms = 0.0;
    double lm_head_ms = 0.0;
    double loss_ms = 0.0;
    double backward_ms = 0.0;
    double gradient_accumulation_ms = 0.0;
    double optimizer_ms = 0.0;

    const std::size_t token_parameter_count = vocab * d;
    const std::size_t position_parameter_count = context * d;
    const std::size_t qkv_parameter_count = d * d;
    const std::size_t w1_parameter_count = d * ff;
    const std::size_t w2_parameter_count = ff * d;
    const std::size_t lm_parameter_count = d * vocab;

    if (parameters[0]->values.size() != token_parameter_count ||
        parameters[1]->values.size() != position_parameter_count ||
        parameters[2]->values.size() != lm_parameter_count) {
        Py_DECREF(parameters_sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "transformer embedding or LM parameter shape mismatch"
        );
        return nullptr;
    }

    for (std::size_t layer = 0; layer < layers; ++layer) {
        const std::size_t base = 3 + layer * 6;
        if (parameters[base + 0]->values.size() != qkv_parameter_count ||
            parameters[base + 1]->values.size() != qkv_parameter_count ||
            parameters[base + 2]->values.size() != qkv_parameter_count ||
            parameters[base + 3]->values.size() != qkv_parameter_count ||
            parameters[base + 4]->values.size() != w1_parameter_count ||
            parameters[base + 5]->values.size() != w2_parameter_count) {
            Py_DECREF(parameters_sequence);
            PyErr_SetString(
                PyExc_ValueError,
                "transformer block parameter shape mismatch"
            );
            return nullptr;
        }
    }

    std::vector<TrainBuffer> parameter_gradients;
    try {
        parameter_gradients.reserve(expected_parameters);
        for (std::size_t index = 0;
             index < expected_parameters;
             ++index) {
            parameter_gradients.emplace_back(train_memory_resource());
            parameter_gradients.back().assign(
                parameters[index]->values.size(),
                0.0
            );
        }
    } catch (const std::bad_alloc&) {
        Py_DECREF(parameters_sequence);
        PyErr_NoMemory();
        return nullptr;
    }

    Py_DECREF(parameters_sequence);

    try {
        TrainMatrix x(length, d);

        const auto embedding_start =
            profile ? TrainClock::now() : TrainClock::time_point{};
        for (std::size_t row = 0; row < length; ++row) {
            const int token_id = token_ids[row];
            const double* token_row =
                parameters[0]->values.data() +
                static_cast<std::size_t>(token_id) * d;
            const double* position_row =
                parameters[1]->values.data() +
                row * d;
            double* x_row = x.row_ptr(row);

            for (std::size_t column = 0; column < d; ++column) {
                x_row[column] =
                    token_row[column] +
                    position_row[column];
            }
        }
        if (profile) {
            embedding_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - embedding_start
            ).count();
        }

        std::vector<TrainLayerCache> caches;
        caches.resize(layers);

        for (std::size_t layer = 0; layer < layers; ++layer) {
            const std::size_t base = 3 + layer * 6;
            const StorageObject* wq_storage = parameters[base + 0];
            const StorageObject* wk_storage = parameters[base + 1];
            const StorageObject* wv_storage = parameters[base + 2];
            const StorageObject* wo_storage = parameters[base + 3];
            const StorageObject* w1_storage = parameters[base + 4];
            const StorageObject* w2_storage = parameters[base + 5];

            TrainMatrix wq(d, d);
            TrainMatrix wk(d, d);
            TrainMatrix wv(d, d);
            TrainMatrix wo(d, d);
            TrainMatrix w1(d, ff);
            TrainMatrix w2(ff, d);

            wq.values.assign(
                wq_storage->values.begin(),
                wq_storage->values.end()
            );
            wk.values.assign(
                wk_storage->values.begin(),
                wk_storage->values.end()
            );
            wv.values.assign(
                wv_storage->values.begin(),
                wv_storage->values.end()
            );
            wo.values.assign(
                wo_storage->values.begin(),
                wo_storage->values.end()
            );
            w1.values.assign(
                w1_storage->values.begin(),
                w1_storage->values.end()
            );
            w2.values.assign(
                w2_storage->values.begin(),
                w2_storage->values.end()
            );

            TrainLayerCache& cache = caches[layer];
            cache.input = x;

            auto normalization_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            train_layer_norm_forward(
                x,
                cache.norm_x,
                cache.mean1,
                cache.inv1
            );
            if (profile) {
                normalization_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - normalization_start
                ).count();
            }

            auto q_start = profile ? TrainClock::now() : TrainClock::time_point{};
            cache.q = train_matmul(cache.norm_x, wq);
            if (profile) {
                q_projection_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - q_start
                ).count();
            }

            auto k_start = profile ? TrainClock::now() : TrainClock::time_point{};
            cache.k = train_matmul(cache.norm_x, wk);
            if (profile) {
                k_projection_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - k_start
                ).count();
            }

            auto v_start = profile ? TrainClock::now() : TrainClock::time_point{};
            cache.v = train_matmul(cache.norm_x, wv);
            if (profile) {
                v_projection_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - v_start
                ).count();
            }

            cache.probabilities.resize(heads);
            cache.merged = TrainMatrix(length, d);

            const double attention_scale =
                1.0 / std::sqrt(static_cast<double>(head_dim));

            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t start = head * head_dim;
                TrainMatrix qh =
                    train_slice_columns(cache.q, start, head_dim);
                TrainMatrix kh =
                    train_slice_columns(cache.k, start, head_dim);
                TrainMatrix vh =
                    train_slice_columns(cache.v, start, head_dim);

                TrainMatrix kt = train_transpose(kh);
                auto attention_scores_start =
                    profile ? TrainClock::now() : TrainClock::time_point{};
                TrainMatrix scores =
                    train_matmul(qh, kt);

                for (double& value : scores.values) {
                    value *= attention_scale;
                }
                if (profile) {
                    attention_scores_ms += std::chrono::duration<double, std::milli>(
                        TrainClock::now() - attention_scores_start
                    ).count();
                }

                auto attention_softmax_start =
                    profile ? TrainClock::now() : TrainClock::time_point{};
                TrainMatrix probs =
                    train_causal_softmax(scores);
                if (profile) {
                    attention_softmax_ms += std::chrono::duration<double, std::milli>(
                        TrainClock::now() - attention_softmax_start
                    ).count();
                }

                auto attention_weighted_sum_start =
                    profile ? TrainClock::now() : TrainClock::time_point{};
                TrainMatrix attended =
                    train_matmul(probs, vh);
                if (profile) {
                    attention_weighted_sum_ms += std::chrono::duration<double, std::milli>(
                        TrainClock::now() - attention_weighted_sum_start
                    ).count();
                }

                train_write_slice(
                    cache.merged,
                    attended,
                    start
                );
                cache.probabilities[head] =
                    std::move(probs);
            }

            auto output_projection_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            cache.attention_output =
                train_matmul(cache.merged, wo);
            if (profile) {
                output_projection_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - output_projection_start
                ).count();
            }

            auto residual_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            cache.residual =
                train_add(x, cache.attention_output);
            if (profile) {
                residual_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - residual_start
                ).count();
            }

            auto normalization2_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            train_layer_norm_forward(
                cache.residual,
                cache.norm_residual,
                cache.mean2,
                cache.inv2
            );
            if (profile) {
                normalization_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - normalization2_start
                ).count();
            }

            auto feed_forward_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            cache.hidden_pre =
                train_matmul(cache.norm_residual, w1);
            if (profile) {
                feed_forward_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - feed_forward_start
                ).count();
            }

            auto activation_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            cache.hidden =
                train_relu(cache.hidden_pre);
            if (profile) {
                activation_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - activation_start
                ).count();
            }

            auto feed_forward_output_start =
                profile ? TrainClock::now() : TrainClock::time_point{};
            TrainMatrix feed_forward =
                train_matmul(cache.hidden, w2);
            if (profile) {
                feed_forward_ms += std::chrono::duration<double, std::milli>(
                    TrainClock::now() - feed_forward_output_start
                ).count();
            }

            x = train_add(cache.residual, feed_forward);
        }

        TrainMatrix final_norm;
        TrainBuffer final_mean(train_memory_resource());
        TrainBuffer final_inv(train_memory_resource());
        auto final_norm_start =
            profile ? TrainClock::now() : TrainClock::time_point{};
        train_layer_norm_forward(
            x,
            final_norm,
            final_mean,
            final_inv
        );
        if (profile) {
            normalization_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - final_norm_start
            ).count();
        }

        TrainMatrix lm_head(d, vocab);
        lm_head.values.assign(
            parameters[2]->values.begin(),
            parameters[2]->values.end()
        );

        auto lm_head_start =
            profile ? TrainClock::now() : TrainClock::time_point{};
        TrainMatrix logits =
            train_matmul(final_norm, lm_head);
        if (profile) {
            lm_head_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - lm_head_start
            ).count();
        }

        if (!train_is_finite(logits)) {
            PyErr_SetString(
                PyExc_FloatingPointError,
                "non-finite logits produced by transformer training"
            );
            return nullptr;
        }

        TrainMatrix dlogits(length, vocab);
        double total_loss = 0.0;

        const auto loss_start =
            profile ? TrainClock::now() : TrainClock::time_point{};
        for (std::size_t row = 0; row < length; ++row) {
            const double* logits_row = logits.row_ptr(row);
            double* dlogits_row = dlogits.row_ptr(row);

            double maximum = logits_row[0];
            for (std::size_t column = 1;
                 column < vocab;
                 ++column) {
                maximum = std::max(
                    maximum,
                    logits_row[column]
                );
            }

            double sum = 0.0;
            for (std::size_t column = 0;
                 column < vocab;
                 ++column) {
                const double value =
                    std::exp(logits_row[column] - maximum);
                dlogits_row[column] = value;
                sum += value;
            }

            if (!(sum > 0.0) || !std::isfinite(sum)) {
                PyErr_SetString(
                    PyExc_FloatingPointError,
                    "invalid transformer softmax normalization"
                );
                return nullptr;
            }

            const double inverse_sum = 1.0 / sum;
            for (std::size_t column = 0;
                 column < vocab;
                 ++column) {
                dlogits_row[column] *= inverse_sum;
            }

            const std::size_t target =
                static_cast<std::size_t>(target_ids[row]);
            const double probability =
                std::max(dlogits_row[target], 1e-12);

            total_loss -= std::log(probability);
            dlogits_row[target] -= 1.0;
        }

        const double inverse_length =
            1.0 / static_cast<double>(length);
        for (double& value : dlogits.values) {
            value *= inverse_length;
        }
        if (profile) {
            loss_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - loss_start
            ).count();
        }

        const auto backward_start =
            profile ? TrainClock::now() : TrainClock::time_point{};

        TrainMatrix final_norm_transpose =
            train_transpose(final_norm);

        TrainMatrix grad_lm_head =
            train_matmul(final_norm_transpose, dlogits);

        train_accumulate_parameter_gradient(
            parameter_gradients[2],
            grad_lm_head,
            1.0
        );

        TrainMatrix lm_transpose =
            train_transpose(lm_head);
        TrainMatrix grad_final_norm =
            train_matmul(dlogits, lm_transpose);

        TrainMatrix grad_x =
            train_layer_norm_backward(
                grad_final_norm,
                x,
                final_mean,
                final_inv
            );

        for (std::size_t reverse = 0;
             reverse < layers;
             ++reverse) {
            const std::size_t layer = layers - 1 - reverse;
            const std::size_t base = 3 + layer * 6;
            const StorageObject* wq_storage = parameters[base + 0];
            const StorageObject* wk_storage = parameters[base + 1];
            const StorageObject* wv_storage = parameters[base + 2];
            const StorageObject* wo_storage = parameters[base + 3];
            const StorageObject* w1_storage = parameters[base + 4];
            const StorageObject* w2_storage = parameters[base + 5];

            TrainMatrix wq(d, d);
            TrainMatrix wk(d, d);
            TrainMatrix wv(d, d);
            TrainMatrix wo(d, d);
            TrainMatrix w1(d, ff);
            TrainMatrix w2(ff, d);

            wq.values = wq_storage->values;
            wk.values = wk_storage->values;
            wv.values = wv_storage->values;
            wo.values = wo_storage->values;
            w1.values = w1_storage->values;
            w2.values = w2_storage->values;

            const TrainLayerCache& cache = caches[layer];

            TrainMatrix d_next = grad_x;
            TrainMatrix d_residual = d_next;

            TrainMatrix w2_transpose = train_transpose(w2);
            TrainMatrix d_hidden =
                train_matmul(d_next, w2_transpose);

            TrainMatrix hidden_transpose =
                train_transpose(cache.hidden);

            TrainMatrix d_w2 =
                train_matmul(hidden_transpose, d_next);

            TrainMatrix d_hidden_pre =
                train_relu_backward(
                    d_hidden,
                    cache.hidden_pre
                );

            TrainMatrix w1_transpose = train_transpose(w1);
            TrainMatrix d_norm_residual =
                train_matmul(
                    d_hidden_pre,
                    w1_transpose
                );

            TrainMatrix norm_residual_transpose =
                train_transpose(cache.norm_residual);

            TrainMatrix d_w1 =
                train_matmul(
                    norm_residual_transpose,
                    d_hidden_pre
                );

            TrainMatrix d_residual_norm =
                train_layer_norm_backward(
                    d_norm_residual,
                    cache.residual,
                    cache.mean2,
                    cache.inv2
                );

            train_add_inplace(
                d_residual,
                d_residual_norm
            );

            TrainMatrix wo_transpose = train_transpose(wo);
            TrainMatrix d_merged =
                train_matmul(
                    d_residual,
                    wo_transpose
                );

            TrainMatrix merged_transpose =
                train_transpose(cache.merged);

            TrainMatrix d_wo =
                train_matmul(
                    merged_transpose,
                    d_residual
                );

            TrainMatrix d_q(length, d);
            TrainMatrix d_k(length, d);
            TrainMatrix d_v(length, d);

            const double attention_scale =
                1.0 / std::sqrt(static_cast<double>(head_dim));

            for (std::size_t head = 0; head < heads; ++head) {
                const std::size_t start = head * head_dim;

                TrainMatrix qh =
                    train_slice_columns(cache.q, start, head_dim);
                TrainMatrix kh =
                    train_slice_columns(cache.k, start, head_dim);
                TrainMatrix vh =
                    train_slice_columns(cache.v, start, head_dim);

                TrainMatrix d_attended =
                    train_slice_columns(
                        d_merged,
                        start,
                        head_dim
                    );

                TrainMatrix vh_transpose =
                    train_transpose(vh);
                TrainMatrix d_probs =
                    train_matmul(
                        d_attended,
                        vh_transpose
                    );

                TrainMatrix probs_transpose =
                    train_transpose(
                        cache.probabilities[head]
                    );
                TrainMatrix d_vh =
                    train_matmul(
                        probs_transpose,
                        d_attended
                    );

                TrainMatrix d_scores =
                    train_softmax_backward(
                        cache.probabilities[head],
                        d_probs
                    );

                for (std::size_t row = 0;
                     row < d_scores.rows;
                     ++row) {
                    const double* d_scores_row =
                        d_scores.row_ptr(row);
                    double* q_row =
                        d_scores.row_ptr(row);
                    (void)q_row;
                    for (std::size_t column = 0;
                         column < d_scores.cols;
                         ++column) {
                        d_scores.at(row, column) *=
                            attention_scale;
                    }
                }

                TrainMatrix dk_transpose =
                    train_matmul(
                        train_transpose(d_scores),
                        qh
                    );
                TrainMatrix dq_head =
                    train_matmul(
                        d_scores,
                        kh
                    );

                train_write_slice(
                    d_q,
                    dq_head,
                    start
                );
                train_write_slice(
                    d_k,
                    dk_transpose,
                    start
                );
                train_write_slice(
                    d_v,
                    d_vh,
                    start
                );
            }

            TrainMatrix wq_transpose = train_transpose(wq);
            TrainMatrix wk_transpose = train_transpose(wk);
            TrainMatrix wv_transpose = train_transpose(wv);

            TrainMatrix d_norm_x_q =
                train_matmul(d_q, wq_transpose);
            TrainMatrix d_norm_x_k =
                train_matmul(d_k, wk_transpose);
            TrainMatrix d_norm_x_v =
                train_matmul(d_v, wv_transpose);

            TrainMatrix d_norm_x =
                train_add(
                    d_norm_x_q,
                    train_add(d_norm_x_k, d_norm_x_v)
                );

            TrainMatrix norm_x_transpose =
                train_transpose(cache.norm_x);

            TrainMatrix d_wq =
                train_matmul(
                    norm_x_transpose,
                    d_q
                );
            TrainMatrix d_wk =
                train_matmul(
                    norm_x_transpose,
                    d_k
                );
            TrainMatrix d_wv =
                train_matmul(
                    norm_x_transpose,
                    d_v
                );

            TrainMatrix d_input_norm =
                train_layer_norm_backward(
                    d_norm_x,
                    cache.input,
                    cache.mean1,
                    cache.inv1
                );

            grad_x = d_input_norm;
            train_add_inplace(grad_x, d_residual);

            train_accumulate_parameter_gradient(
                parameter_gradients[base + 0],
                d_wq,
                1.0
            );
            train_accumulate_parameter_gradient(
                parameter_gradients[base + 1],
                d_wk,
                1.0
            );
            train_accumulate_parameter_gradient(
                parameter_gradients[base + 2],
                d_wv,
                1.0
            );
            train_accumulate_parameter_gradient(
                parameter_gradients[base + 3],
                d_wo,
                1.0
            );
            train_accumulate_parameter_gradient(
                parameter_gradients[base + 4],
                d_w1,
                1.0
            );
            train_accumulate_parameter_gradient(
                parameter_gradients[base + 5],
                d_w2,
                1.0
            );
        }

        if (profile) {
            backward_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - backward_start
            ).count();
        }

        const auto gradient_accumulation_start =
            profile ? TrainClock::now() : TrainClock::time_point{};

        for (std::size_t row = 0; row < length; ++row) {
            const int token_id = token_ids[row];

            double* token_gradient =
                parameter_gradients[0].data() +
                static_cast<std::size_t>(token_id) * d;
            double* position_gradient =
                parameter_gradients[1].data() +
                row * d;

            const double* grad_row =
                grad_x.row_ptr(row);

            for (std::size_t column = 0;
                 column < d;
                 ++column) {
                token_gradient[column] += grad_row[column];
                position_gradient[column] += grad_row[column];
            }
        }

        if (profile) {
            gradient_accumulation_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - gradient_accumulation_start
            ).count();
        }

        const auto optimizer_start =
            profile ? TrainClock::now() : TrainClock::time_point{};

        for (std::size_t parameter = 0;
             parameter < expected_parameters;
             ++parameter) {
            StorageObject* storage = parameters[parameter];
            TrainBuffer& gradient =
                parameter_gradients[parameter];

            for (std::size_t index = 0;
                 index < gradient.size();
                 ++index) {
                const double value = storage->values[index];

                if (!std::isfinite(gradient[index])) {
                    PyErr_SetString(
                        PyExc_FloatingPointError,
                        "non-finite transformer gradient"
                    );
                    return nullptr;
                }

                storage->values[index] -=
                    learning_rate * gradient[index];
            }

            for (double value : storage->values) {
                if (!std::isfinite(value)) {
                    PyErr_SetString(
                        PyExc_FloatingPointError,
                        "non-finite transformer parameter after update"
                    );
                    return nullptr;
                }
            }
        }

        if (profile) {
            optimizer_ms += std::chrono::duration<double, std::milli>(
                TrainClock::now() - optimizer_start
            ).count();
            const double total_ms =
                std::chrono::duration<double, std::milli>(
                    TrainClock::now() - profile_start
                ).count();

            std::fprintf(
                stderr,
                "cpp_profile "
                "embedding_ms=%.3f "
                "normalization_ms=%.3f "
                "q_ms=%.3f "
                "k_ms=%.3f "
                "v_ms=%.3f "
                "attention_scores_ms=%.3f "
                "attention_softmax_ms=%.3f "
                "attention_weighted_sum_ms=%.3f "
                "output_projection_ms=%.3f "
                "residual_ms=%.3f "
                "feed_forward_ms=%.3f "
                "activation_ms=%.3f "
                "lm_head_ms=%.3f "
                "loss_ms=%.3f "
                "backward_ms=%.3f "
                "gradient_accumulation_ms=%.3f "
                "optimizer_ms=%.3f "
                "total_ms=%.3f\\n",
                embedding_ms,
                normalization_ms,
                q_projection_ms,
                k_projection_ms,
                v_projection_ms,
                attention_scores_ms,
                attention_softmax_ms,
                attention_weighted_sum_ms,
                output_projection_ms,
                residual_ms,
                feed_forward_ms,
                activation_ms,
                lm_head_ms,
                loss_ms,
                backward_ms,
                gradient_accumulation_ms,
                optimizer_ms,
                total_ms
            );
        }

        return PyFloat_FromDouble(
            total_loss / static_cast<double>(length)
        );
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }
}


PyObject* cpp_transformer_train_batch(PyObject*, PyObject* args) {
    PyObject* parameters_object = nullptr;
    PyObject* inputs_object = nullptr;
    PyObject* targets_object = nullptr;
    Py_ssize_t vocabulary_size;
    Py_ssize_t context_length;
    Py_ssize_t d_model;
    Py_ssize_t n_heads;
    Py_ssize_t feed_forward_size;
    Py_ssize_t n_layers;
    double learning_rate;

    if (!PyArg_ParseTuple(
        args,
        "OOOnnnnnnd:transformer_train_batch",
        &parameters_object,
        &inputs_object,
        &targets_object,
        &vocabulary_size,
        &context_length,
        &d_model,
        &n_heads,
        &feed_forward_size,
        &n_layers,
        &learning_rate
    )) {
        return nullptr;
    }

    PyObject* input_batch = PySequence_Fast(
        inputs_object,
        "inputs must be a sequence of token sequences"
    );
    if (input_batch == nullptr) {
        return nullptr;
    }

    PyObject* target_batch = PySequence_Fast(
        targets_object,
        "targets must be a sequence of token sequences"
    );
    if (target_batch == nullptr) {
        Py_DECREF(input_batch);
        return nullptr;
    }

    const Py_ssize_t batch_size =
        PySequence_Fast_GET_SIZE(input_batch);

    if (batch_size <= 0 ||
        PySequence_Fast_GET_SIZE(target_batch) != batch_size) {
        Py_DECREF(input_batch);
        Py_DECREF(target_batch);
        PyErr_SetString(
            PyExc_ValueError,
            "training batch must contain matching non-empty input/target sequences"
        );
        return nullptr;
    }

    PyObject** input_items =
        PySequence_Fast_ITEMS(input_batch);
    PyObject** target_items =
        PySequence_Fast_ITEMS(target_batch);

    PyObject* sequence_length_object = nullptr;
    PyObject* vocabulary_object = nullptr;
    PyObject* context_object = nullptr;
    PyObject* d_model_object = nullptr;
    PyObject* heads_object = nullptr;
    PyObject* feed_forward_object = nullptr;
    PyObject* layers_object = nullptr;
    PyObject* learning_rate_object = nullptr;

    double total_loss = 0.0;

    try {
        for (Py_ssize_t index = 0; index < batch_size; ++index) {
            PyObject* input_sequence = input_items[index];
            PyObject* target_sequence = target_items[index];

            PyObject* input_fast = PySequence_Fast(
                input_sequence,
                "batch inputs must contain integer sequences"
            );
            if (input_fast == nullptr) {
                throw std::runtime_error(
                    "invalid batch input sequence"
                );
            }

            PyObject* target_fast = PySequence_Fast(
                target_sequence,
                "batch targets must contain integer sequences"
            );
            if (target_fast == nullptr) {
                Py_DECREF(input_fast);
                throw std::runtime_error(
                    "invalid batch target sequence"
                );
            }

            const Py_ssize_t sequence_length =
                PySequence_Fast_GET_SIZE(input_fast);
            const Py_ssize_t target_length =
                PySequence_Fast_GET_SIZE(target_fast);

            if (sequence_length <= 0 ||
                target_length != sequence_length ||
                sequence_length > context_length) {
                Py_DECREF(input_fast);
                Py_DECREF(target_fast);
                throw std::runtime_error(
                    "batch sequence lengths are invalid"
                );
            }

            if (sequence_length_object == nullptr) {
                sequence_length_object =
                    PyLong_FromSsize_t(sequence_length);
                vocabulary_object =
                    PyLong_FromSsize_t(vocabulary_size);
                context_object =
                    PyLong_FromSsize_t(context_length);
                d_model_object =
                    PyLong_FromSsize_t(d_model);
                heads_object =
                    PyLong_FromSsize_t(n_heads);
                feed_forward_object =
                    PyLong_FromSsize_t(feed_forward_size);
                layers_object =
                    PyLong_FromSsize_t(n_layers);
                learning_rate_object =
                    PyFloat_FromDouble(learning_rate);

                if (sequence_length_object == nullptr ||
                    vocabulary_object == nullptr ||
                    context_object == nullptr ||
                    d_model_object == nullptr ||
                    heads_object == nullptr ||
                    feed_forward_object == nullptr ||
                    layers_object == nullptr ||
                    learning_rate_object == nullptr) {
                    Py_DECREF(input_fast);
                    Py_DECREF(target_fast);
                    throw std::bad_alloc();
                }
            } else {
                const Py_ssize_t expected_length =
                    PyLong_AsSsize_t(sequence_length_object);

                if (sequence_length != expected_length) {
                    Py_DECREF(input_fast);
                    Py_DECREF(target_fast);
                    throw std::runtime_error(
                        "all batch sequences must have the same length"
                    );
                }
            }

            // cpp_transformer_train_step owns no Python state beyond the
            // parameter storages; build one native argument tuple and keep
            // the entire training loop inside C++.
            PyObject* step_args = PyTuple_New(11);
            if (step_args == nullptr) {
                Py_DECREF(input_fast);
                Py_DECREF(target_fast);
                throw std::bad_alloc();
            }

            Py_INCREF(parameters_object);
            PyTuple_SET_ITEM(step_args, 0, parameters_object);

            Py_INCREF(input_fast);
            PyTuple_SET_ITEM(step_args, 1, input_fast);

            Py_INCREF(target_fast);
            PyTuple_SET_ITEM(step_args, 2, target_fast);

            Py_INCREF(vocabulary_object);
            PyTuple_SET_ITEM(step_args, 3, vocabulary_object);
            Py_INCREF(context_object);
            PyTuple_SET_ITEM(step_args, 4, context_object);
            Py_INCREF(d_model_object);
            PyTuple_SET_ITEM(step_args, 5, d_model_object);
            Py_INCREF(heads_object);
            PyTuple_SET_ITEM(step_args, 6, heads_object);
            Py_INCREF(feed_forward_object);
            PyTuple_SET_ITEM(step_args, 7, feed_forward_object);
            Py_INCREF(layers_object);
            PyTuple_SET_ITEM(step_args, 8, layers_object);
            Py_INCREF(sequence_length_object);
            PyTuple_SET_ITEM(step_args, 9, sequence_length_object);
            Py_INCREF(learning_rate_object);
            PyTuple_SET_ITEM(step_args, 10, learning_rate_object);

            // cpp_transformer_train_step uses the same verified native
            // implementation. This wrapper eliminates Python's per-step
            // training loop without maintaining a second numerical path.
            PyObject* result =
                cpp_transformer_train_step(nullptr, step_args);

            Py_DECREF(step_args);
            Py_DECREF(input_fast);
            Py_DECREF(target_fast);

            if (result == nullptr) {
                throw std::runtime_error(
                    "native transformer train step failed"
                );
            }

            total_loss += PyFloat_AsDouble(result);
            Py_DECREF(result);

            if (PyErr_Occurred() != nullptr) {
                throw std::runtime_error(
                    "native transformer batch loss conversion failed"
                );
            }
        }
    } catch (const std::bad_alloc&) {
        Py_XDECREF(sequence_length_object);
        Py_XDECREF(vocabulary_object);
        Py_XDECREF(context_object);
        Py_XDECREF(d_model_object);
        Py_XDECREF(heads_object);
        Py_XDECREF(feed_forward_object);
        Py_XDECREF(layers_object);
        Py_XDECREF(learning_rate_object);
        Py_DECREF(input_batch);
        Py_DECREF(target_batch);
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception& exc) {
        Py_XDECREF(sequence_length_object);
        Py_XDECREF(vocabulary_object);
        Py_XDECREF(context_object);
        Py_XDECREF(d_model_object);
        Py_XDECREF(heads_object);
        Py_XDECREF(feed_forward_object);
        Py_XDECREF(layers_object);
        Py_XDECREF(learning_rate_object);
        Py_DECREF(input_batch);
        Py_DECREF(target_batch);
        PyErr_SetString(PyExc_RuntimeError, exc.what());
        return nullptr;
    }

    Py_XDECREF(sequence_length_object);
    Py_XDECREF(vocabulary_object);
    Py_XDECREF(context_object);
    Py_XDECREF(d_model_object);
    Py_XDECREF(heads_object);
    Py_XDECREF(feed_forward_object);
    Py_XDECREF(layers_object);
    Py_XDECREF(learning_rate_object);
    Py_DECREF(input_batch);
    Py_DECREF(target_batch);

    return PyFloat_FromDouble(
        total_loss / static_cast<double>(batch_size)
    );
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
    {"causal_softmax", cpp_causal_softmax, METH_VARARGS,
     "Causal row-wise softmax kernel."},
    {"softmax_backward", cpp_softmax_backward, METH_VARARGS,
     "Softmax backward kernel."},
    {"transpose", cpp_transpose, METH_VARARGS,
     "Rank-2 transpose."},
    {"dense_forward", cpp_dense_forward, METH_VARARGS,
     "Dense forward kernel."},
    {"dense_indexed_forward", cpp_dense_indexed_forward, METH_VARARGS,
     "Dense forward kernel for sparse indexed one-hot inputs."},
    {"dense_indexed_backward", cpp_dense_indexed_backward, METH_VARARGS,
     "Dense backward kernel for sparse indexed one-hot inputs."},
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
    {"transformer_train_step", cpp_transformer_train_step, METH_VARARGS,
     "Fused Level 2 Transformer forward, backward, and SGD step."},
    {"transformer_train_batch", cpp_transformer_train_batch, METH_VARARGS,
     "Fused Level 2 C++ training loop over a batch."},
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
