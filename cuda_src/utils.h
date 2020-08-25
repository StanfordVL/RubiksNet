#include <vector>
#include <sstream>
#include <iterator>
#include <iostream>
#include <type_traits>
#include <cassert>
#include <cstdlib>
#include <ATen/Utils.h>

#if __CUDACC_VER_MAJOR__ == 9
    #include <ATen/ATen.h>
    #include <ATen/core/Tensor.h>
    namespace torch = at;
#else
    // NVCC on CUDA 9 will fail if you include this header. CUDA 10 is fine
    #include <torch/extension.h>
#endif


#ifndef TORCHX_H_
#define TORCHX_H_

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::ostringstream;
using std::initializer_list;
using torch::Tensor;

#ifndef __CUDACC_VER_MAJOR__
using namespace pybind11::literals;
#endif

namespace torchx {

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

typedef torch::IntArrayRef ShapeRef;
typedef vector<int64_t> ShapeVector;
typedef string s_;  // quickly convert const char* to string class

/*
 * ************************
 * printing utils
 * ************************
 */
// Overload torchx_str_hook with your custom class to enable printing with
// to_str(), pyprint(), spyprint(), and sjoin()!
template<typename T>
void torchx_str_hook(std::ostream& oss, const T& object) {
    oss << object;
}

template<typename VectorT>
void _vector_str(
    std::ostream& oss, const VectorT& vec,
    const string& sep, const string& parentheses
) {

    bool first = true;
    if (parentheses.size() > 0)
        oss << parentheses[0];
    for (auto& elem: vec) {
        if (!first)
            oss << sep;
        torchx_str_hook(oss, elem);
        first = false;
    }
    if (parentheses.size() > 0)
        oss << parentheses[1];
}

template<typename E>
void torchx_str_hook(std::ostream& oss, const vector<E>& vec) {
    _vector_str(oss, vec, ", ", "[]");
}

template<typename E>
void torchx_str_hook(std::ostream& oss, const std::initializer_list<E>& vec) {
    _vector_str(oss, vec, ", ", "[]");
}

template<typename E>
void torchx_str_hook(std::ostream& oss, const torch::ArrayRef<E>& vec) {
    _vector_str(oss, vec, ", ", "()");
}

template<typename T>
string to_str(const T& obj) {
    ostringstream oss;
    torchx_str_hook(oss, obj);
    return oss.str();
}

/* printing utils mimicing python interface */
struct PrintOption {
    string _end = "\n";
    string _sep = " ";
    bool _fixed = false;
    bool _scientific = false;
    int _precision = -2; // default
    std::ostream* _stream = &std::cout;

    PrintOption end(const string& end) {
        _end = end;
        return *this;
    }
    PrintOption sep(const string& sep) {
        _sep = sep;
        return *this;
    }
    PrintOption stream(std::ostream& stream) {
        _stream = &stream;
        return *this;
    }
    PrintOption fixed() {
        _fixed = true;
        return *this;
    }
    PrintOption scientific() {
        _scientific = true;
        return *this;
    }
    PrintOption precision(int n) {
        _precision = n;
        return *this;
    }
    PrintOption max_precision() {
        _precision = -1;
        return *this;
    }
    PrintOption _add_iomanip() {
        *_stream << std::boolalpha;
        // delayed execution of iomanip
        if (_fixed)
            *_stream << std::fixed;
        if (_scientific)
            *_stream << std::scientific;
        if (_precision > 0)
            *_stream << std::setprecision(_precision);
        if (_precision == -1)  // max precision
            *_stream << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        return *this;
    }
};


inline void _pyprint_helper(const PrintOption& option) {
    *option._stream << option._end;
}

template<typename T>
void _pyprint_helper(const PrintOption& option, T&& arg) {
    auto& oss = *option._stream;
    torchx_str_hook(oss, arg);
    oss << option._end;
}

template<typename T, typename... Ts>
void _pyprint_helper(const PrintOption& option, T&& arg, Ts&& ... rest) {
    auto& oss = *option._stream;
    torchx_str_hook(oss, arg);
    oss << option._sep;
    _pyprint_helper(option, std::forward<Ts>(rest) ...);
}

template<typename... Ts>
void pyprint(PrintOption option, Ts&& ... args) {
    // pass PrintOption by copy avoids duplicating code for lvalue and rvalue
    _pyprint_helper(option._add_iomanip(), std::forward<Ts>(args) ...);
}

template<typename... Ts>
void pyprint(Ts&& ... args) {
    pyprint(PrintOption(), std::forward<Ts>(args) ...);
}

template<typename... Ts>
string spyprint(PrintOption option, Ts&& ... args) {
    ostringstream oss;
    _pyprint_helper(
        option.stream(oss)._add_iomanip(),
        std::forward<Ts>(args) ...
    );
    return oss.str();
}

template<typename... Ts>
string spyprint(Ts&& ... args) {
    return spyprint(PrintOption().end(""), std::forward<Ts>(args) ...);
}

// must be invoked with at least 2 args
template<typename T1, typename T2, typename... Ts>
string sjoin(const string& sep, T1&& arg1, T2&& arg2, Ts&& ... args) {
    return spyprint(
        PrintOption().sep(sep).end(""),
        std::forward<T1>(arg1), std::forward<T2>(arg2),
        std::forward<Ts>(args) ...
    );
}

// must be invoked with at least 2 args
template<typename Iterable>
string sjoin(const string& sep, const Iterable& vec) {
    ostringstream oss;
    _vector_str(oss, vec, sep, "");
    return oss.str();
}

// mimic python multiply string
template<typename Char, typename Traits, typename Allocator>
std::basic_string<Char, Traits, Allocator> operator*(
    const std::basic_string<Char, Traits, Allocator> s, size_t n
) {
    std::basic_string<Char, Traits, Allocator> result = s;
    for (size_t i = 0; i < n; ++i) {
        result += s;
    }
    return result;
}
// now s * 3 and 3 * s both work
template<typename Char, typename Traits, typename Allocator>
std::basic_string<Char, Traits, Allocator> operator*(
    size_t n, const std::basic_string<Char, Traits, Allocator>& s
) {
    return s * n;
}


inline void ptitle(const string& title, const string& sep = "=") {

    pyprint(sep*30, title, sep*30);
}

/*
 * ************************
 * torch utils
 * ************************
 */
inline string _file_line_errmsg(string file_name, int line_num) {
    return spyprint(" at file \"" + file_name + "\", line", line_num, ":\n");
}

struct AssertError: public std::exception {
    AssertError(string msg, string file_name, int line_num):
        _msg("TorchX AssertError" + _file_line_errmsg(file_name, line_num) + msg) {}
    const char* what() const noexcept {
        return _msg.c_str();
    }
    string _msg;
};


struct ShapeException : public std::exception {
    ShapeException(
        ShapeRef expected,
        ShapeRef actual,
        string msg, string file_name, int line_num) {
        _msg = spyprint(
            "ShapeException" + _file_line_errmsg(file_name, line_num),
            msg, "expected shape =", expected,
            "; actual shape =", actual
        );
    }

    ShapeException(
        int64_t expected_dim,
        int64_t actual_dim,
        string msg, string file_name, int line_num) {
        _msg = spyprint(
            "ShapeException" + _file_line_errmsg(file_name, line_num),
            msg, "expected dimension =", expected_dim,
            "; actual dimension =", actual_dim
        );
    }

    const char* what() const noexcept {
        return _msg.c_str();
    }
    string _msg;
};

// from latest "c10/util/Exception.h"
// Return x if it is non-empty; otherwise return y.
inline string _if_empty_then(const string& x, const string& y) {
    if (x.empty()) {
        return y;
    } else {
        return x;
    }
}
#define TX_ASSERT(cond, ...) \
  if (!(cond)) { \
    throw AssertError( \
        _if_empty_then( \
            spyprint(__VA_ARGS__), \
            "Expected " #cond " to be true, but got false." \
        ), __FILE__, __LINE__ \
    ); \
  }

// NOTE: latest pytorch has TORCH_CHECK to replace AT_CHECK
#define _TX_CHECK_CUDA(x) TX_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define _TX_CHECK_CONTIGUOUS(x) TX_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define TX_CHECK_TENSOR(x) _TX_CHECK_CUDA(x); _TX_CHECK_CONTIGUOUS(x)

#define TX_CHECK_SHAPE(expected, actual, msg) \
    if (expected != actual) { \
        throw ShapeException(expected, actual, msg, __FILE__, __LINE__); \
    }


#define TX_CHECK_DIM(tensor, expected_dim) \
    if (tensor.dim() != expected_dim) { \
        throw ShapeException(expected_dim, tensor.dim(), "tensor " #tensor, __FILE__, __LINE__); \
    }

inline bool is_empty(const Tensor& x) {
    return x.sizes().empty();
}

inline void create_tensor_if_empty(
    Tensor& output, const Tensor& shape_like, const string& shape_err_msg
)
/**
 * Check if the output tensor is empty.
 * - if yes, allocate a new tensor with the same shape as `shape_like`
 * - if no, check if the tensor has the same shape as `shape_like`
 * @param output
 * @param shape
 * @param msg
 */
{
    if (is_empty(output)) {
        output = torch::zeros_like(shape_like);
    }
    else {
        TX_CHECK_TENSOR(output);
        TX_CHECK_SHAPE(shape_like.sizes(), output.sizes(), shape_err_msg);
    }
}

inline void create_tensor_if_empty(
    Tensor& output, ShapeRef shape,
    const torch::TensorOptions& options, const string& shape_err_msg
)
/**
 * Check if the output tensor is empty.
 * - if yes, allocate a new tensor of the expected shape
 * - if no, check if the tensor has the expected shape
 * @param output
 * @param shape
 * @param msg
 */
{
    if (is_empty(output)) {
        output = torch::zeros(shape, options);
    }
    else {
        TX_CHECK_TENSOR(output);
        TX_CHECK_SHAPE(shape, output.sizes(), shape_err_msg);
    }
}

/*
 * ************************
 * Misc
 * ************************
 */

#define SELF_DIR string{__FILE__}.substr(0, string{__FILE__}.rfind("/")+1)

inline string get_root_dir() {
    string file_path = __FILE__;
    return file_path.substr(0, file_path.rfind("/")+1);
}


/**
 * Turn on env variable TORCHX_CPP_DEBUG to allow debugging mode without recompile
 */
inline bool is_sys_debug() {
    const char* env_value  = std::getenv("TORCHX_CPP_DEBUG");
    if (env_value == 0) {
        return false;
    }
    string env_s = env_value;
    return env_s == "true" || env_s == "True" || env_s == "1";
}


class Timer {
public:
    void start() {
        m_StartTime = std::chrono::system_clock::now();
        m_bRunning = true;
    }

    void stop() {
        m_EndTime = std::chrono::system_clock::now();
        m_bRunning = false;
    }

    double elapsed_ms() {
        std::chrono::time_point<std::chrono::system_clock> endTime;
        if(m_bRunning) {
            endTime = std::chrono::system_clock::now();
        }
        else {
            endTime = m_EndTime;
        }
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
    }

    double elapsed_s() {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool                                               m_bRunning = false;
};

/**
 * Overload N-arg macro
 * example:
 * #define mymacro(...)  OVERLOAD_MACRO(mymacro, __VA_ARGS__)
 * #define mymacro2(arg1, arg2)  // implement 2-arg macro version
 * #define mymacro3(arg1, arg2, arg3)  // implement 3-arg macro version
 */
// https://stackoverflow.com/questions/11761703/overloading-macro-on-number-of-arguments
#define OVERLOAD_MACRO(M, ...) _OVERLOAD_MACRO(M, _COUNT_ARGS(__VA_ARGS__)) (__VA_ARGS__)
#define _OVERLOAD_MACRO(macro_name, number_of_args)  _OVERLOAD_MACRO_EXPAND(macro_name, number_of_args)
#define _OVERLOAD_MACRO_EXPAND(macro_name, number_of_args)  macro_name##number_of_args
#define _COUNT_ARGS(...)  _ARG_PATTERN_MATCH(__VA_ARGS__, 9,8,7,6,5,4,3,2,1)
#define _ARG_PATTERN_MATCH(_1,_2,_3,_4,_5,_6,_7,_8,_9, N, ...)   N


}  // namespace torchx

#endif  // TORCHX_H_
