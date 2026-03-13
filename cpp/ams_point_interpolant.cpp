// ----------------------------------------------------------------------------
// Python interfaced version of Adaptive Multigrid Solver's PointInterpolant.cpp
// ----------------------------------------------------------------------------
/*
Copyright (c) 2014, Michael Kazhdan
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "PreProcessor.h"

// #undef USE_DOUBLE								// If enabled, double-precesion is used
#define USE_DOUBLE // If enabled, double-precesion is used
#define WEIGHT_DEGREE 2 // The order of the B-Spline used to splat in the weights for density estimation
#define DEFAULT_FEM_DEGREE 2 // The default finite-element degree
#define DEFAULT_FEM_BOUNDARY BOUNDARY_FREE // The default finite-element boundary type

#include "CmdLineParser.h"
#include "FEMTree.h"
#include "MyMiscellany.h"
#include "PPolynomial.h"
#include "Ply.h"
#include "VertexFactory.h"
#include <algorithm>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include "Image.h"
#include "DataStream.imp.h"
#include "Reconstructors.h"
#include "RegularGrid.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace PoissonRecon;

struct PointInterpolantParameters {
    double* inMemoryValuePositions = nullptr;
    double* inMemoryValues = nullptr;
    size_t inMemoryValueCount = 0;
    double* inMemoryGradientPositions = nullptr;
    double* inMemoryGradients = nullptr;
    size_t inMemoryGradientCount = 0;
    int depth = 8;
    int degree = DEFAULT_FEM_DEGREE;
    int solveDepth = -1;
    int fullDepth = 5;
    int baseDepth = -1;
    int bType = DEFAULT_FEM_BOUNDARY + 1;
    int iters = 8;
    int baseVCycles = 4;
    int maxMemoryGb = 0;
    int parallel = 0;
    int schedule = (int)ThreadPool::Schedule;
    int threadChunkSize = (int)ThreadPool::ChunkSize;
    float valueWeight = 1000.0f;
    float gradientWeight = 1.0f;
    float scale = 1.1f;
    float width = 0.f;
    float cgAccuracy = 1e-3f;
    float iso = 0.f;
    float lapWeight = 0.f;
    float biLapWeight = 1.f;
    bool showResidual = false;
    bool exactInterpolation = true;
    bool verbose = false;
    std::string transformFile = std::string("");
};

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetBoundingBoxXForm(Point<Real, Dim> min, Point<Real, Dim> max, Real scaleFactor)
{
    Point<Real, Dim> center = (max + min) / 2;
    Real scale = max[0] - min[0];
    for (int d = 1; d < Dim; d++)
        scale = std::max<Real>(scale, max[d] - min[d]);
    scale *= scaleFactor;
    for (int i = 0; i < Dim; i++)
        center[i] -= scale / 2;
    XForm<Real, Dim + 1> tXForm = XForm<Real, Dim + 1>::Identity(), sXForm = XForm<Real, Dim + 1>::Identity();
    for (int i = 0; i < Dim; i++)
        sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
    return sXForm * tXForm;
}

template <class Real, unsigned int Dim>
XForm<Real, Dim + 1> GetBoundingBoxXForm(Point<Real, Dim> min, Point<Real, Dim> max, Real width, Real scaleFactor, int& depth)
{
    // Get the target resolution (along the largest dimension)
    Real resolution = (max[0] - min[0]) / width;
    for (int d = 1; d < Dim; d++)
        resolution = std::max<Real>(resolution, (max[d] - min[d]) / width);
    resolution *= scaleFactor;
    depth = 0;
    while ((1 << depth) < resolution)
        depth++;

    Point<Real, Dim> center = (max + min) / 2;
    Real scale = (1 << depth) * width;

    for (int i = 0; i < Dim; i++)
        center[i] -= scale / 2;
    XForm<Real, Dim + 1> tXForm = XForm<Real, Dim + 1>::Identity(), sXForm = XForm<Real, Dim + 1>::Identity();
    for (int i = 0; i < Dim; i++)
        sXForm(i, i) = (Real)(1. / scale), tXForm(Dim, i) = -center[i];
    return sXForm * tXForm;
}

template <typename Real, unsigned int Dim, typename FunctionValueType>
using InputPointStreamInfo = typename FEMTreeInitializer<Dim, Real>::template InputPointStream<FunctionValueType>;

template <typename Real, unsigned int Dim, typename FunctionValueType>
using InputPointStream = typename InputPointStreamInfo<Real, Dim, FunctionValueType>::StreamType;

template <class Real, unsigned int Dim, typename FunctionValueType>
XForm<Real, Dim + 1> GetPointXForm(InputPointStream<Real, Dim, FunctionValueType>& stream, Real width, Real scaleFactor, int& depth)
{
    Point<Real, Dim> min, max;
    InputPointStreamInfo<Real, Dim, FunctionValueType>::BoundingBox(stream, min, max);
    return GetBoundingBoxXForm(min, max, width, scaleFactor, depth);
}
template <class Real, unsigned int Dim, typename FunctionValueType>
XForm<Real, Dim + 1> GetPointXForm(InputPointStream<Real, Dim, FunctionValueType>& stream, Real scaleFactor)
{
    Point<Real, Dim> min, max;
    InputPointStreamInfo<Real, Dim, FunctionValueType>::BoundingBox(stream, min, max);
    return GetBoundingBoxXForm(min, max, scaleFactor);
}

template <unsigned int Dim, typename Real, typename PointSampleData>
struct ConstraintDual;

template <unsigned int Dim, typename Real>
struct ConstraintDual<Dim, Real, DirectSum<Real, Real>> {
    typedef DirectSum<Real, Real> PointSampleData;
    Real vWeight;
    ConstraintDual(Real v)
        : vWeight(v)
    {
    }
    CumulativeDerivativeValues<Real, Dim, 0> operator()(const Point<Real, Dim>& p, const PointSampleData& data) const
    {
        Real value = data.template get<0>();
        CumulativeDerivativeValues<Real, Dim, 0> cdv;
        cdv[0] = value * vWeight;
        return cdv;
    }
};

template <unsigned int Dim, typename Real>
struct ConstraintDual<Dim, Real, DirectSum<Real, Point<Real, Dim>>> {
    typedef DirectSum<Real, Point<Real, Dim>> PointSampleData;
    Real gWeight;
    ConstraintDual(Real g)
        : gWeight(g)
    {
    }
    CumulativeDerivativeValues<Real, Dim, 1> operator()(const Point<Real, Dim>& p, const PointSampleData& data) const
    {
        Point<Real, Dim> gradient = data.template get<0>();
        CumulativeDerivativeValues<Real, Dim, 1> cdv;
        for (int d = 0; d < Dim; d++)
            cdv[1 + d] = gradient[d] * gWeight;
        return cdv;
    }
};

template <unsigned int Dim, typename Real, typename TotalPointSampleData>
struct SystemDual;

template <unsigned int Dim, typename Real>
struct SystemDual<Dim, Real, DirectSum<Real, Real>> {
    CumulativeDerivativeValues<Real, Dim, 0> weight;
    SystemDual(Real v) { weight[0] = v; }
    CumulativeDerivativeValues<Real, Dim, 0> operator()(Point<Real, Dim> p, const DirectSum<Real, Real>& data, const CumulativeDerivativeValues<Real, Dim, 0>& dValues) const
    {
        return dValues * weight;
    }
    CumulativeDerivativeValues<double, Dim, 0> operator()(Point<Real, Dim> p, const DirectSum<Real, Real>& data, const CumulativeDerivativeValues<double, Dim, 0>& dValues) const
    {
        return dValues * weight;
    };
};
template <unsigned int Dim>
struct SystemDual<Dim, double, DirectSum<double, double>> {
    typedef double Real;
    CumulativeDerivativeValues<Real, Dim, 0> weight;
    SystemDual(Real v) { weight[0] = v; }
    CumulativeDerivativeValues<Real, Dim, 0> operator()(Point<Real, Dim> p, const DirectSum<double, double>& data, const CumulativeDerivativeValues<Real, Dim, 0>& dValues) const
    {
        return dValues * weight;
    }
};

template <unsigned int Dim, typename Real>
struct SystemDual<Dim, Real, DirectSum<Real, Point<Real, Dim>>> {
    CumulativeDerivativeValues<Real, Dim, 1> weight;
    SystemDual(Real g)
    {
        weight[0] = 0;
        for (int d = 0; d < Dim; d++)
            weight[d + 1] = g;
    }
    CumulativeDerivativeValues<Real, Dim, 1> operator()(Point<Real, Dim> p, const DirectSum<Real, Point<Real, Dim>>& data, const CumulativeDerivativeValues<Real, Dim, 1>& dValues) const
    {
        return dValues * weight;
    }
    CumulativeDerivativeValues<double, Dim, 1> operator()(Point<Real, Dim> p, const DirectSum<Real, Point<Real, Dim>>& data, const CumulativeDerivativeValues<double, Dim, 1>& dValues) const
    {
        return dValues * weight;
    };
};
template <unsigned int Dim>
struct SystemDual<Dim, double, DirectSum<double, Point<double, Dim>>> {
    typedef double Real;
    CumulativeDerivativeValues<Real, Dim, 1> weight;
    SystemDual(Real g)
    {
        weight[0] = 0;
        for (int d = 0; d < Dim; d++)
            weight[1 + d] = g;
    }
    CumulativeDerivativeValues<Real, Dim, 1> operator()(Point<Real, Dim> p, const DirectSum<Real, Point<Real, Dim>>& data, const CumulativeDerivativeValues<Real, Dim, 1>& dValues) const
    {
        return dValues * weight;
    }
};

// In-memory stream data classes
template <typename Real, unsigned int Dim, typename Factory>
struct MemoryBackedValueStream : public InputDataStream<typename Factory::VertexType> {
    typedef typename Factory::VertexType VertexType;

    const double* positions; // Array of positions: [x0,y0,z0, x1,y1,z1, ...]
    const double* values; // Array of values: [v0, v1, v2, ...]
    size_t count;
    size_t currentIndex;
    Factory factory;

    MemoryBackedValueStream(const double* pos, const double* vals, size_t n, const Factory& f)
        : positions(pos)
        , values(vals)
        , count(n)
        , currentIndex(0)
        , factory(f)
    {
    }

    void reset(void) { currentIndex = 0; }

    bool read(VertexType& v)
    {
        if (currentIndex >= count)
            return false;

        Point<Real, Dim> p;
        for (unsigned int d = 0; d < Dim; d++)
            p[d] = (Real)positions[currentIndex * Dim + d];

        Real val = (Real)values[currentIndex];

        v.template get<0>() = p;
        v.template get<1>().template get<0>() = val;

        currentIndex++;
        return true;
    }
};

template <typename Real, unsigned int Dim, typename Factory>
struct MemoryBackedGradientStream : public InputDataStream<typename Factory::VertexType> {
    typedef typename Factory::VertexType VertexType;

    const double* positions; // Array of positions: [x0,y0,z0, x1,y1,z1, ...]
    const double* gradients; // Array of gradients: [gx0,gy0,gz0, gx1,gy1,gz1, ...]
    size_t count;
    size_t currentIndex;
    Factory factory;

    MemoryBackedGradientStream(const double* pos, const double* grads, size_t n, const Factory& f)
        : positions(pos)
        , gradients(grads)
        , count(n)
        , currentIndex(0)
        , factory(f)
    {
    }

    void reset(void) { currentIndex = 0; }

    bool read(VertexType& v)
    {
        if (currentIndex >= count)
            return false;

        Point<Real, Dim> p;
        for (unsigned int d = 0; d < Dim; d++)
            p[d] = (Real)positions[currentIndex * Dim + d];

        Point<Real, Dim> grad;
        for (unsigned int d = 0; d < Dim; d++)
            grad[d] = (Real)gradients[currentIndex * Dim + d];

        v.template get<0>() = p;
        v.template get<1>().template get<0>() = grad;

        currentIndex++;
        return true;
    }
};

template <unsigned int Dim, typename Real>
class Interpolant {
public:
    using EvalFn = std::function<Real(const Point<Real, Dim>&, unsigned int thread)>;

    // Updated constructor: takes shared_ptr + eval function
    Interpolant(std::shared_ptr<FEMTree<Dim, Real>> treePtr, EvalFn evalFn)
        : m_treePtr(std::move(treePtr))
        , m_evalFn(std::move(evalFn))
    {
    }

    Real eval(const Point<Real, Dim>& p) { return m_evalFn(p, 0); }

    // Convenience overload for numpy arrays (for pybind11)
    py::array_t<Real> eval(py::array_t<double, py::array::c_style | py::array::forcecast> queryPoints)
    {
        auto buf = queryPoints.request();
        size_t n = buf.shape[0];
        double* ptr = static_cast<double*>(buf.ptr);

        py::array_t<Real> result(std::vector<ssize_t> {(ssize_t)n});
        auto resBuf = result.request();
        Real* resPtr = static_cast<Real*>(resBuf.ptr);

        {
            py::gil_scoped_release release;
            ThreadPool::ParallelFor(0, n, [&](unsigned int thread, size_t i) {
                Point<Real, Dim> p;
                for (unsigned int d = 0; d < Dim; d++)
                    p[d] = (Real)ptr[i * Dim + d];
                resPtr[i] = m_evalFn(p, thread); // safe - no ThreadPool inside
            });
        }

        return result;
    }

private:
    std::shared_ptr<FEMTree<Dim, Real>> m_treePtr; // kept alive by the lambda too
    EvalFn m_evalFn;
};

template <class Real, unsigned int... FEMSigs>
Interpolant<sizeof...(FEMSigs), Real> Execute(
    UIntPack<FEMSigs...>,
    PointInterpolantParameters& param)
{
    // ] -----------------------------------------------------------------
    static const int Dim = sizeof...(FEMSigs);
    typedef UIntPack<FEMSigs...> Sigs;
    typedef UIntPack<FEMSignature<FEMSigs>::Degree...> Degrees;
    typedef UIntPack<FEMDegreeAndBType<WEIGHT_DEGREE, DerivativeBoundary<FEMSignature<FEMSigs>::BType, 1>::BType>::Signature...> DataSigs;
    typedef typename FEMTree<Dim, Real>::template DensityEstimator<WEIGHT_DEGREE> DensityEstimator;

    typedef typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 0> ValueInterpolationInfo;
    typedef typename FEMTree<Dim, Real>::template InterpolationInfo<Real, 1> GradientInterpolationInfo;

    // The factory for constructing the function's values
    typedef VertexFactory::Factory<Real, VertexFactory::ValueFactory<Real>> FunctionValueFactory;
    // The factory for constructing the function's gradients
    typedef VertexFactory::Factory<Real, VertexFactory::NormalFactory<Real, Dim>> FunctionGradientFactory;

    // The factory for constructing the value data
    typedef VertexFactory::Factory<Real, VertexFactory::PositionFactory<Real, Dim>, FunctionValueFactory> InputSampleValueFactory;
    // The factory for constructing the gradient data
    typedef VertexFactory::Factory<Real, VertexFactory::PositionFactory<Real, Dim>, FunctionGradientFactory> InputSampleGradientFactory;

    // The type of the function values
    typedef typename FunctionValueFactory::VertexType FunctionValueType;
    // The type of the function gradients
    typedef typename FunctionGradientFactory::VertexType FunctionGradientType;

    // The type of the input value
    typedef typename InputSampleValueFactory::VertexType InputSampleValueType;
    // The type of the input gradient
    typedef typename InputSampleGradientFactory::VertexType InputSampleGradientType;

    typedef InputDataStream<InputSampleValueType> InputPointValueStream;
    typedef InputDataStream<InputSampleGradientType> InputPointGradientStream;

    struct XInputPointValueStream : public InputDataStream<InputSampleValueType> {
        InputDataStream<InputSampleValueType>& stream;
        XForm<Real, Dim + 1> pointTransform;
        XInputPointValueStream(InputDataStream<InputSampleValueType>& stream, XForm<Real, Dim + 1> modelToUnitCube)
            : stream(stream)
            , pointTransform(modelToUnitCube)
        {
        }
        bool read(InputSampleValueType& s)
        {
            if (stream.read(s)) {
                s.template get<0>() = pointTransform * s.template get<0>();
                return true;
            } else
                return false;
        }
        void reset(void) { return stream.reset(); }
    };
    struct XInputPointGradientStream : public InputDataStream<InputSampleGradientType> {
        //    G(p) = F( A*p )
        // F(p+d) = F(p) + < \nabla F(p) , d >
        // G(p+d) = F( A*p ) + < \nabla F(A*p) , A*d >
        //        = G(p) + < A^t * \nabla F(A*p) , d >
        // => \nabla G(p) = A^t * \nabla F(A*p)

        InputDataStream<InputSampleGradientType>& stream;
        XForm<Real, Dim + 1> pointTransform;
        XForm<Real, Dim> gradientTransform;
        XInputPointGradientStream(InputDataStream<InputSampleGradientType>& stream, XForm<Real, Dim + 1> modelToUnitCube)
            : stream(stream)
        {
            pointTransform = modelToUnitCube;
            gradientTransform = XForm<Real, Dim>(pointTransform).inverse().transpose();
        }
        bool read(InputSampleGradientType& s)
        {
            if (stream.read(s)) {
                s.template get<0>() = pointTransform * s.template get<0>();
                s.template get<1>() = gradientTransform * s.template get<1>().template get<0>();
                return true;
            } else
                return false;
        }
        void reset(void) { return stream.reset(); }
    };

    FunctionValueFactory functionValueFactory;
    FunctionGradientFactory functionGradientFactory;

    InputSampleValueFactory inputSampleValueFactory(VertexFactory::PositionFactory<Real, Dim>(), functionValueFactory);
    InputSampleGradientFactory inputSampleGradientFactory(VertexFactory::PositionFactory<Real, Dim>(), functionGradientFactory);

    typedef RegularTreeNode<Dim, FEMTreeNodeData, depth_and_offset_type> FEMTreeNode;
    std::vector<std::string> comments;
    if (param.verbose) {
        std::cout << "***********************************************" << std::endl;
        std::cout << "***********************************************" << std::endl;
        std::cout << "** Running Point Interpolant (Version " << ADAPTIVE_SOLVERS_VERSION << ") **" << std::endl;
        std::cout << "***********************************************" << std::endl;
        std::cout << "***********************************************" << std::endl;
    }

    ThreadPool::ParallelizationType = (ThreadPool::ParallelType)param.parallel;

    XForm<Real, Dim + 1> modelToUnitCube, unitCubeToModel;
    if (!param.transformFile.empty()) {
        FILE* fp = fopen(param.transformFile.c_str(), "r");
        if (!fp) {
            MK_WARN("Could not read x-form from: ", param.transformFile.c_str());
            modelToUnitCube = XForm<Real, Dim + 1>::Identity();
        } else {
            for (int i = 0; i < Dim + 1; i++)
                for (int j = 0; j < Dim + 1; j++) {
                    float f;
                    if (fscanf(fp, " %f ", &f) != 1)
                        MK_THROW("Failed to read xform");
                    modelToUnitCube(i, j) = (Real)f;
                }
            fclose(fp);
        }
    } else
        modelToUnitCube = XForm<Real, Dim + 1>::Identity();

    double startTime = Time();

    std::shared_ptr<FEMTree<Dim, Real>> treePtr = std::make_shared<FEMTree<Dim, Real>>(MEMORY_ALLOCATOR_BLOCK_SIZE);
    Profiler profiler(20);

    if (param.depth > 0 && param.width > 0) {
        MK_WARN("Both depth and width parameters were set, ignoring width");
        param.width = 0;
    }

    size_t pointValueCount, pointGradientCount;

    std::vector<typename FEMTree<Dim, Real>::PointSample>* valueSamples = NULL;
    std::vector<FunctionValueType>* valueSampleData = NULL;
    std::vector<typename FEMTree<Dim, Real>::PointSample>* gradientSamples = NULL;
    std::vector<FunctionGradientType>* gradientSampleData = NULL;

    // Read in the samples
    {
        profiler.reset();
        InputPointValueStream* pointValueStream = NULL;
        InputPointGradientStream* pointGradientStream = NULL;
        Point<Real, Dim> valueMin, valueMax, gradientMin, gradientMax;
        std::vector<InputSampleValueType> inCorePointsAndValues;
        std::vector<InputSampleGradientType> inCorePointsAndGradients;

        // { ------------------------------------------------------------------
        // Modify reading inputs to include in-memory data
        if (param.valueWeight > 0) {
            valueSampleData = new std::vector<FunctionValueType>();

            // Check for in-memory data first
            if (param.inMemoryValuePositions != nullptr && param.inMemoryValues != nullptr && param.inMemoryValueCount > 0) {
                if (param.verbose)
                    std::cout << "Using in-memory value data (" << param.inMemoryValueCount << " points)" << std::endl;

                pointValueStream = new MemoryBackedValueStream<Real, Dim, InputSampleValueFactory>(
                    param.inMemoryValuePositions, param.inMemoryValues, param.inMemoryValueCount, inputSampleValueFactory);
            } else {
                MK_THROW("No value data provided");
            }

            // Rest of the value processing code remains the same
            XInputPointValueStream _pointStream(*pointValueStream, modelToUnitCube);
            FunctionValueType s = functionValueFactory();
            InputPointStreamInfo<Real, Dim, FunctionValueType>::BoundingBox(_pointStream, s, valueMin, valueMax);
        }
        if (param.gradientWeight > 0) {
            gradientSampleData = new std::vector<FunctionGradientType>();

            // Check for in-memory data first
            if (param.inMemoryGradientPositions != nullptr && param.inMemoryGradients != nullptr && param.inMemoryGradientCount > 0) {
                // Use in-memory data
                if (param.verbose)
                    std::cout << "Using in-memory gradient data (" << param.inMemoryGradientCount << " points)" << std::endl;

                pointGradientStream = new MemoryBackedGradientStream<Real, Dim, InputSampleGradientFactory>(
                    param.inMemoryGradientPositions, param.inMemoryGradients, param.inMemoryGradientCount, inputSampleGradientFactory);
            } else {
                MK_THROW("No gradient data provided");
            }

            // Rest of the gradient processing code remains the same
            XInputPointGradientStream _pointStream(*pointGradientStream, modelToUnitCube);
            FunctionGradientType s = functionGradientFactory();
            InputPointStreamInfo<Real, Dim, FunctionGradientType>::BoundingBox(_pointStream, s, gradientMin, gradientMax);
        }
        // ] ------------------------------------------------------------------

        {
            Point<Real, Dim> min, max;
            if (param.valueWeight > 0 && param.gradientWeight > 0)
                for (int d = 0; d < Dim; d++)
                    min[d] = std::min<Real>(valueMin[d], gradientMin[d]), max[d] = std::max<Real>(valueMax[d], gradientMax[d]);
            else if (param.valueWeight > 0)
                min = valueMin, max = valueMax;
            else if (param.gradientWeight > 0)
                min = gradientMin, max = gradientMax;

            if (param.width > 0) {
                modelToUnitCube = GetBoundingBoxXForm(min, max, (Real)param.width, (Real)(param.scale > 0 ? param.scale : 1.), param.depth) * modelToUnitCube;
                if (param.solveDepth < 0)
                    param.solveDepth = param.depth;
                if (param.solveDepth > param.depth) {
                    MK_WARN("Solution depth cannot exceed system depth: ", param.solveDepth, " <= ", param.depth);
                    param.solveDepth = param.depth;
                }
                if (param.fullDepth > param.depth) {
                    MK_WARN("Full depth cannot exceed system depth: ", param.fullDepth, " <= ", param.depth);
                    param.fullDepth = param.depth;
                }
                if (param.baseDepth > param.fullDepth) {
                    if (param.baseDepth > 0)
                        MK_WARN("Base depth must be smaller than full depth: ", param.baseDepth, " <= ", param.fullDepth);
                    param.baseDepth = param.fullDepth;
                }
            } else
                modelToUnitCube = param.scale > 0 ? GetBoundingBoxXForm(min, max, (Real)param.scale) * modelToUnitCube : modelToUnitCube;
        }

        if (param.valueWeight > 0) {
            valueSamples = new std::vector<typename FEMTree<Dim, Real>::PointSample>();
            XInputPointValueStream _pointStream(*pointValueStream, modelToUnitCube);
            auto ProcessData = [](const Point<Real, Dim>& p, FunctionValueType& d) { return (Real)1.; };
            FunctionValueType zeroValue = functionValueFactory();
            typename FEMTreeInitializer<Dim, Real>::StreamInitializationData sid;
            {
                using ExternalType = std::tuple<Point<Real, Dim>, FunctionValueType>;
                using InternalType = std::tuple<InputSampleValueType>;
                auto converter = [](const InternalType& iType) {
                    ExternalType xType;
                    std::get<0>(xType) = std::get<0>(iType).template get<0>();
                    std::get<1>(xType) = std::get<0>(iType).template get<1>();
                    return xType;
                };
                InputDataStreamConverter<InternalType, ExternalType> __pointStream(_pointStream, converter, InputSampleValueType(Point<Real, Dim>(), zeroValue));
                pointValueCount = FEMTreeInitializer<Dim, Real>::template Initialize<FunctionValueType>(sid, treePtr->spaceRoot(), __pointStream, zeroValue, param.depth, *valueSamples, *valueSampleData, treePtr->nodeAllocators.size() ? treePtr->nodeAllocators[0] : NULL, treePtr->initializer(), ProcessData);
            }
            delete pointValueStream;
        } else
            pointValueCount = 0;

        if (param.gradientWeight > 0) {
            gradientSamples = new std::vector<typename FEMTree<Dim, Real>::PointSample>();
            XInputPointGradientStream _pointStream(*pointGradientStream, modelToUnitCube);
            auto ProcessData = [](const Point<Real, Dim>& p, FunctionGradientType& d) { return (Real)1.; };
            FunctionGradientType zeroGradient = functionGradientFactory();
            typename FEMTreeInitializer<Dim, Real>::StreamInitializationData sid;
            {
                using ExternalType = std::tuple<Point<Real, Dim>, FunctionGradientType>;
                using InternalType = std::tuple<InputSampleGradientType>;
                auto converter = [](const InternalType& iType) {
                    ExternalType xType;
                    std::get<0>(xType) = std::get<0>(iType).template get<0>();
                    std::get<1>(xType) = std::get<0>(iType).template get<1>();
                    return xType;
                };
                InputDataStreamConverter<InternalType, ExternalType> __pointGradientStream(_pointStream, converter, InputSampleGradientType(Point<Real, Dim>(), zeroGradient));
                pointGradientCount = FEMTreeInitializer<Dim, Real>::template Initialize<FunctionGradientType>(sid, treePtr->spaceRoot(), __pointGradientStream, zeroGradient, param.depth, *gradientSamples, *gradientSampleData, treePtr->nodeAllocators.size() ? treePtr->nodeAllocators[0] : NULL, treePtr->initializer(), ProcessData);
            }
            delete pointGradientStream;
        } else
            pointGradientCount = 0;

        unitCubeToModel = modelToUnitCube.inverse();

        if (param.verbose) {
            if (valueSamples)
                std::cout << "Input Value Points / Value Samples: " << pointValueCount << " / " << valueSamples->size() << std::endl;
            if (gradientSamples)
                std::cout << "Input Gradient Points / Gradient Samples: " << pointGradientCount << " / " << gradientSamples->size() << std::endl;
            std::cout << "# Read input into tree: " << profiler << std::endl;
        }
    }

    DenseNodeData<Real, Sigs> solution;
    {
        DenseNodeData<Real, Sigs> constraints;
        ValueInterpolationInfo* valueInterpolationInfo = NULL;
        GradientInterpolationInfo* gradientInterpolationInfo = NULL;
        int solveDepth = param.depth;

        treePtr->resetNodeIndices(0);

        if (param.valueWeight > 0) {
            assert(valueSamples != nullptr);
            assert(valueSampleData != nullptr);
            assert(valueSamples->size() == valueSampleData->size());

            auto dataPtr = GetPointer(*valueSampleData);
            if (param.exactInterpolation) {
                valueInterpolationInfo = FEMTree<Dim, Real>::template InitializeExactPointAndDataInterpolationInfo<Real, FunctionValueType, 0>(*treePtr, *valueSamples, GetPointer(*valueSampleData), ConstraintDual<Dim, Real, FunctionValueType>((Real)param.valueWeight), SystemDual<Dim, Real, FunctionValueType>((Real)param.valueWeight), true, false);
            } else
                valueInterpolationInfo = FEMTree<Dim, Real>::template InitializeApproximatePointAndDataInterpolationInfo<Real, FunctionValueType, 0>(*treePtr, *valueSamples, GetPointer(*valueSampleData), ConstraintDual<Dim, Real, FunctionValueType>((Real)param.valueWeight), SystemDual<Dim, Real, FunctionValueType>((Real)param.valueWeight), true, param.depth, 1);
        }
        if (param.gradientWeight > 0) {
            if (param.exactInterpolation)
                gradientInterpolationInfo = FEMTree<Dim, Real>::template InitializeExactPointAndDataInterpolationInfo<Real, FunctionGradientType, 1>(*treePtr, *gradientSamples, GetPointer(*gradientSampleData), ConstraintDual<Dim, Real, FunctionGradientType>((Real)param.gradientWeight), SystemDual<Dim, Real, FunctionGradientType>((Real)param.gradientWeight), true, false);
            else
                gradientInterpolationInfo = FEMTree<Dim, Real>::template InitializeApproximatePointAndDataInterpolationInfo<Real, FunctionGradientType, 1>(*treePtr, *gradientSamples, GetPointer(*gradientSampleData), ConstraintDual<Dim, Real, FunctionGradientType>((Real)param.gradientWeight), SystemDual<Dim, Real, FunctionGradientType>((Real)param.gradientWeight), true, param.depth, 1);
        }

        // Prepare for multigrid
        {
            profiler.reset();

            auto addNodeFunctor = [&](int d, const int off[Dim]) { return d <= param.fullDepth; };
            if (param.valueWeight > 0 && param.gradientWeight > 0)
                treePtr->template finalizeForMultigrid<Degrees::Max(), Degrees::Max()>(param.baseDepth, addNodeFunctor, [](const FEMTreeNode*) { return true; }, std::make_tuple(valueInterpolationInfo, gradientInterpolationInfo));
            else if (param.valueWeight > 0)
                treePtr->template finalizeForMultigrid<Degrees::Max(), Degrees::Max()>(param.baseDepth, addNodeFunctor, [](const FEMTreeNode*) { return true; }, std::make_tuple(valueInterpolationInfo));
            else if (param.gradientWeight > 0)
                treePtr->template finalizeForMultigrid<Degrees::Max(), Degrees::Max()>(param.baseDepth, addNodeFunctor, [](const FEMTreeNode*) { return true; }, std::make_tuple(gradientInterpolationInfo));

            if (param.verbose)
                std::cout << "#       Finalized tree: " << profiler << std::endl;
        }

        // Add the interpolation constraints
        {
            profiler.reset();
            constraints = treePtr->initDenseNodeData(Sigs());
            if (param.valueWeight > 0 && param.gradientWeight > 0)
                treePtr->addInterpolationConstraints(constraints, solveDepth, std::make_tuple(valueInterpolationInfo, gradientInterpolationInfo));
            else if (param.valueWeight > 0)
                treePtr->addInterpolationConstraints(constraints, solveDepth, std::make_tuple(valueInterpolationInfo));
            else if (param.gradientWeight > 0)
                treePtr->addInterpolationConstraints(constraints, solveDepth, std::make_tuple(gradientInterpolationInfo));
            if (param.verbose)
                std::cout << "#Set point constraints: " << profiler << std::endl;
        }

        if (param.verbose) {
            std::cout << "All Nodes / Active Nodes / Ghost Nodes: " << treePtr->allNodes() << " / " << treePtr->activeNodes() << " / " << treePtr->ghostNodes() << std::endl;
            std::cout << "Memory Usage: " << float(MemoryInfo::Usage()) / (1 << 20) << " MB" << std::endl;
        }

        // Solve the linear system
        {
            profiler.reset();
            typename FEMTree<Dim, Real>::SolverInfo sInfo;
            sInfo.cgDepth = 0, sInfo.cascadic = true, sInfo.vCycles = 1, sInfo.iters = param.iters, sInfo.cgAccuracy = param.cgAccuracy, sInfo.verbose = param.verbose, sInfo.showResidual = param.showResidual, sInfo.showGlobalResidual = SHOW_GLOBAL_RESIDUAL_NONE, sInfo.sliceBlockSize = 1;
            sInfo.baseVCycles = param.baseVCycles;
            typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim, 2>> F({0., (double)param.lapWeight, (double)param.biLapWeight});
            if (param.valueWeight > 0 && param.gradientWeight > 0) {
                solution = treePtr->solveSystem(Sigs(), F, constraints, param.baseDepth, param.solveDepth, sInfo, std::make_tuple(valueInterpolationInfo, gradientInterpolationInfo));
            } else if (param.valueWeight > 0) {
                solution = treePtr->solveSystem(Sigs(), F, constraints, param.baseDepth, param.solveDepth, sInfo, std::make_tuple(valueInterpolationInfo));
            } else if (param.gradientWeight > 0) {
                solution = treePtr->solveSystem(Sigs(), F, constraints, param.baseDepth, param.solveDepth, sInfo, std::make_tuple(gradientInterpolationInfo));
            }
            if (param.verbose)
                std::cout << "# Linear system solved: " << profiler << std::endl;
            delete valueInterpolationInfo, valueInterpolationInfo = NULL;
            delete gradientInterpolationInfo, gradientInterpolationInfo = NULL;
        }
    }

    if (param.verbose) {
        if (valueSamples) {
            typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 0> evaluator(treePtr.get(), solution);
            std::pair<double, double> valueStat(0, 0);
            std::vector<std::pair<double, double>> valueStats(ThreadPool::NumThreads(), std::pair<double, double>(0, 0));
            ThreadPool::ParallelFor(0, valueSamples->size(), [&](unsigned int thread, size_t j) {
                ProjectiveData<Point<Real, Dim>, Real>& sample = (*valueSamples)[j].sample;
                Real w = sample.weight;
                if (w > 0) {
                    CumulativeDerivativeValues<Real, Dim, 0> values = evaluator.values(sample.data / sample.weight, thread, (*valueSamples)[j].node);
                    Real v1 = values[0];
                    Real v2 = (*valueSampleData)[j].template get<0>() / w;
                    valueStats[thread].first += (v1 - v2) * (v1 - v2) * w;
                    valueStats[thread].second += (v1 * v1 + v2 * v2) * w;
                }
            });
            for (unsigned int t = 0; t < ThreadPool::NumThreads(); t++)
                valueStat.first += valueStats[t].first, valueStat.second += valueStats[t].second;
            if (param.verbose)
                std::cout << "Value Error: " << sqrt(valueStat.first / valueStat.second) << std::endl;
        }
        if (gradientSamples) {
            typename FEMTree<Dim, Real>::template MultiThreadedEvaluator<Sigs, 1> evaluator(treePtr.get(), solution);
            std::pair<double, double> gradientStat(0, 0);
            std::vector<std::pair<double, double>> gradientStats(ThreadPool::NumThreads(), std::pair<double, double>(0, 0));
            ThreadPool::ParallelFor(0, gradientSamples->size(), [&](unsigned int thread, size_t j) {
                ProjectiveData<Point<Real, Dim>, Real>& sample = (*gradientSamples)[j].sample;
                Real w = sample.weight;
                if (w > 0) {
                    CumulativeDerivativeValues<Real, Dim, 1> values = evaluator.values(sample.data / sample.weight, thread, (*gradientSamples)[j].node);
                    Point<Real, Dim> g1;
                    for (int d = 0; d < Dim; d++)
                        g1[d] = values[d + 1];
                    Point<Real, Dim> g2 = (*gradientSampleData)[j].template get<0>() / w;
                    gradientStats[thread].first += Point<Real, Dim>::SquareNorm(g1 - g2) * w;
                    gradientStats[thread].second += (Point<Real, Dim>::SquareNorm(g1) + Point<Real, Dim>::SquareNorm(g2)) * w;
                }
            });
            for (unsigned int t = 0; t < ThreadPool::NumThreads(); t++)
                gradientStat.first += gradientStats[t].first, gradientStat.second += gradientStats[t].second;
            if (param.verbose)
                std::cout << "Gradient Error: " << sqrt(gradientStat.first / gradientStat.second) << std::endl;
        }
    }

    delete valueSamples, valueSamples = NULL;
    delete gradientSamples, gradientSamples = NULL;
    delete valueSampleData, valueSampleData = NULL;
    delete gradientSampleData, gradientSampleData = NULL;

    if (param.verbose)
        std::cout << "#          Total Solve: " << Time() - startTime << " (s), " << MemoryInfo::PeakMemoryUsageMB() << " (MB)" << std::endl;

    auto sharedTree = std::shared_ptr<FEMTree<Dim, Real>>(std::move(treePtr));
    auto sharedSolution = std::make_shared<DenseNodeData<Real, Sigs>>(std::move(solution));
    auto sharedXForm = std::make_shared<XForm<Real, Dim + 1>>(modelToUnitCube);

    // Construct the evaluator ONCE here, not inside the lambda
    using Evaluator = typename FEMTree<sizeof...(FEMSigs), Real>::template MultiThreadedEvaluator<Sigs, 0>;
    auto sharedEvaluator = std::make_shared<Evaluator>(sharedTree.get(), *sharedSolution);

    typename Interpolant<sizeof...(FEMSigs), Real>::EvalFn evalFn =
        [sharedTree, sharedEvaluator, sharedXForm, sharedSolution](const Point<Real, sizeof...(FEMSigs)>& p, unsigned int thread) -> Real {
        Point<Real, sizeof...(FEMSigs)> xp = (*sharedXForm) * p;
        return sharedEvaluator->values(xp, thread, nullptr)[0];
    };

    return Interpolant<sizeof...(FEMSigs), Real>(std::move(sharedTree), std::move(evalFn));
}

// [ ------------------------------------------------------------------
// Updated to the new signature of the Execute function above
template <unsigned int Dim, class Real, BoundaryType BType>
Interpolant<Dim, Real> Execute(PointInterpolantParameters& param)
{
    switch (param.degree) {
    case 2:
        return Execute<Real>(IsotropicUIntPack<Dim, FEMDegreeAndBType<2, BType>::Signature>(), param);
    case 3:
        return Execute<Real>(IsotropicUIntPack<Dim, FEMDegreeAndBType<3, BType>::Signature>(), param);
    default:
        MK_THROW("Only B-Splines of degree 2 - 3 are supported");
    }
}
// ] ------------------------------------------------------------------

template <unsigned int Dim, class Real>
Interpolant<Dim, Real> Execute(PointInterpolantParameters& param)
{
    switch (param.bType) {
    case BOUNDARY_FREE + 1:
        return Execute<Dim, Real, BOUNDARY_FREE>(param);
    case BOUNDARY_NEUMANN + 1:
        return Execute<Dim, Real, BOUNDARY_NEUMANN>(param);
    case BOUNDARY_DIRICHLET + 1:
        return Execute<Dim, Real, BOUNDARY_DIRICHLET>(param);
    default:
        MK_THROW("Not a valid boundary type: ", param.bType);
    }
}

// ---------------------------------------------------------------------
// Python interface
// ---------------------------------------------------------------------

/*
Modified version of PointInterpolant.cpp from Adaptive Multigrid Solvers (https://github.com/mkazhdan/PoissonRecon) to accept numpy arrays
*/
template <unsigned int Dim, typename Real>
Interpolant<Dim, Real> pointInterpolant(
    py::array_t<double, py::array::c_style | py::array::forcecast> positionsValues,
    py::array_t<double, py::array::c_style | py::array::forcecast> values,
    py::array_t<double, py::array::c_style | py::array::forcecast> positionsGradients,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradients,
    int depth = 8,
    int degree = DEFAULT_FEM_DEGREE,
    int solveDepth = -1,
    int fullDepth = 5,
    int baseDepth = -1,
    int bType = DEFAULT_FEM_BOUNDARY + 1,
    int iters = 8,
    int baseVCycles = 4,
    int maxMemoryGb = 0,
    int parallel = 0,
    int schedule = (int)ThreadPool::Schedule,
    int threadChunkSize = (int)ThreadPool::ChunkSize,
    float valueWeight = 1000.0f,
    float gradientWeight = 1.0f,
    float scale = 1.1f,
    float width = 0.f,
    float cgAccuracy = 1e-3f,
    float iso = 0.f,
    float lapWeight = 0.f,
    float biLapWeight = 1.f,
    bool showPerformance = false,
    bool showResidual = false,
    bool exactInterpolation = true,
    bool verbose = false,
    std::string transformFile = std::string(""))
{
    Timer timer;

    // Check sizes
    if (positionsValues.size() == 0 && positionsGradients.size() == 0) {
        throw py::value_error("Empty inputs? At least values or gradients should be set!");
    }
    if (values.shape(0) != positionsValues.shape(0)) {
        throw py::value_error("The number of positions for values ('positions_values') and values, should be the same.");
    }
    if (gradients.shape(0) != positionsGradients.shape(0)) {
        throw py::value_error("The number of positions for gradients ('positions_gradients') and gradients, should be the same.");
    }
    ssize_t dim;
    if (positionsValues.size() > 0)
        dim = positionsValues.shape(1);
    else
        dim = positionsGradients.shape(0);
    if (dim < 2 || dim > 3) {
        throw py::value_error("Only 2D or 3D positions are supported.");
    }
    if (positionsValues.size() > 0 && positionsGradients.size() > 0 && positionsValues.shape(1) != positionsGradients.shape(1)) {
        throw py::value_error("The dimension of positions for values ('positions_values') and gradients ('positions_gradients') should be the same.");
    }

    // Get the buffers
    py::buffer_info positionsValuesBuf = positionsValues.request();
    py::buffer_info valuesBuf = values.request();
    py::buffer_info positionsGradientsBuf = positionsGradients.request();
    py::buffer_info gradientsBuf = gradients.request();

    // Get the raw pointers to the data
    double* positionsValuesPtr = static_cast<double*>(positionsValuesBuf.ptr);
    double* valuesPtr = static_cast<double*>(valuesBuf.ptr);
    double* positionsGradientsPtr = static_cast<double*>(positionsGradientsBuf.ptr);
    double* gradientsPtr = static_cast<double*>(gradientsBuf.ptr);

    // Number of samples (values and gradients)
    int valueCount = static_cast<int>(positionsValuesBuf.shape[0]);
    int gradientCount = static_cast<int>(positionsGradientsBuf.shape[0]);

    // Call the PointInterpolant
    if (valueCount == 0 && gradientCount == 0) {
        MK_THROW("Must provide either values or gradients (both counts are 0)");
    }

    if (valueCount > 0 && (positionsValuesPtr == nullptr || valuesPtr == nullptr)) {
        MK_THROW("valueCount > 0 but position or value pointers are null");
    }

    if (gradientCount > 0 && (positionsGradientsPtr == nullptr || gradientsPtr == nullptr)) {
        MK_THROW("gradientCount > 0 but position or gradient pointers are null");
    }

    // Put all parameters in a struct to make things readable
    PointInterpolantParameters param;
    param.inMemoryValuePositions = positionsValuesPtr;
    param.inMemoryValues = valuesPtr;
    param.inMemoryValueCount = valueCount;
    param.inMemoryGradientPositions = positionsGradientsPtr;
    param.inMemoryGradients = gradientsPtr;
    param.inMemoryGradientCount = gradientCount;
    param.depth = depth;
    param.degree = degree;
    param.solveDepth = solveDepth;
    param.fullDepth = fullDepth;
    param.baseDepth = baseDepth;
    param.bType = bType;
    param.iters = iters;
    param.baseVCycles = baseVCycles;
    param.maxMemoryGb = maxMemoryGb;
    param.parallel = parallel;
    param.schedule = schedule;
    param.threadChunkSize = threadChunkSize;
    param.valueWeight = valueWeight;
    param.gradientWeight = gradientWeight;
    param.scale = scale;
    param.width = width;
    param.cgAccuracy = cgAccuracy;
    param.iso = iso;
    param.lapWeight = lapWeight;
    param.biLapWeight = biLapWeight;
    param.showResidual = showResidual;
    param.exactInterpolation = exactInterpolation;
    param.verbose = verbose;
    param.transformFile = transformFile;

    // Initial parameter check (more checks performed in Execute)
    if (valueCount == 0)
        param.valueWeight = 0.0;
    if (gradientCount == 0)
        param.gradientWeight = 0.0;
    if (param.maxMemoryGb > 0)
        SetPeakMemoryMB(param.maxMemoryGb << 10);
    ThreadPool::ChunkSize = param.threadChunkSize;
    ThreadPool::Schedule = (ThreadPool::ScheduleType)param.schedule;

    if (param.lapWeight < 0)
        MK_THROW("Laplacian weight must be non-negative: ", param.lapWeight, " > 0");
    if (param.biLapWeight < 0)
        MK_THROW("Bi-Laplacian weight must be non-negative: ", param.biLapWeight, " > 0");
    if (!param.lapWeight && !param.biLapWeight)
        MK_THROW("Eiter Laplacian or bi-Laplacian weight must be positive");

    if (param.baseDepth < 0)
        param.baseDepth = param.fullDepth;
    if (param.baseDepth > param.fullDepth) {
        if (param.baseDepth > 0)
            MK_WARN("Base depth must be smaller than full depth: ", param.baseDepth, " <= ", param.fullDepth);
        param.baseDepth = param.fullDepth;
    }
    if (param.solveDepth < 0)
        param.solveDepth = param.depth;
    if (param.solveDepth > param.depth) {
        MK_WARN("Solution depth cannot exceed system depth: ", param.solveDepth, " <= ", param.depth);
        param.solveDepth = param.depth;
    }

    // Call Execute
    Interpolant<Dim, Real> interpolant = Execute<Dim, Real>(param);

    if (showPerformance) {
        printf("Time (Wall/CPU): %.2f / %.2f\n", timer.wallTime(), timer.cpuTime());
        printf("Peak Memory (MB): %d\n", MemoryInfo::PeakMemoryUsageMB());
    }

    return interpolant;
}

template <int Dim, typename Real>
void bind_interpolant(py::module& m, const std::string& name)
{
    py::class_<Interpolant<Dim, Real>>(m, name.c_str())
        .def("eval", [](Interpolant<Dim, Real>& self, py::array_t<double, py::array::c_style | py::array::forcecast> pts) {
            return self.eval(pts);
        });
}

py::list getParallelTypes()
{
    py::list result;
    for (const auto& name : ThreadPool::ParallelNames) {
        std::string nameCorrected;
        std::copy_if(name.begin(), name.end(), std::back_inserter(nameCorrected),
            [](unsigned char c) { return !std::isspace(c); });
        result.append(nameCorrected);
    }
    return result;
}

PYBIND11_MODULE(ams_point_interpolant, m)
{
    m.doc() = "C++ module for Adaptive Multigrid Solvers' Point Interpolant";

    m.def("ams_point_interpolant_2d",
        &pointInterpolant<2, double>,
        py::return_value_policy::take_ownership,
        "Fits a function to a set of sample values and/or gradients by finding the coefficients of the function that minimize an energy composed of a point-wise interpolation term and Laplacian and bi-Laplacian smoothness terms.",
        py::arg("positions_values") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("values") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("positions_gradients") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("gradients") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("depth") = 8,
        py::arg("degree") = DEFAULT_FEM_DEGREE,
        py::arg("solve_depth") = -1,
        py::arg("full_depth") = 5,
        py::arg("base_depth") = -1,
        py::arg("b_type") = DEFAULT_FEM_BOUNDARY + 1,
        py::arg("iters") = 8,
        py::arg("base_v_cycles") = 4,
        py::arg("max_memory_gb") = 0,
        py::arg("parallel") = 0,
        py::arg("schedule") = (int)ThreadPool::Schedule,
        py::arg("thread_chunk_size") = (int)ThreadPool::ChunkSize,
        py::arg("value_weight") = 1000.0f,
        py::arg("gradient_weight") = 1.0f,
        py::arg("scale") = 1.1f,
        py::arg("width") = 0.f,
        py::arg("cg_accuracy") = 1e-3f,
        py::arg("iso") = 0.f,
        py::arg("lap_weight") = 0.f,
        py::arg("bi_lap_weight") = 1.f,
        py::arg("show_performance") = false,
        py::arg("show_residual") = false,
        py::arg("exact_interpolation") = true,
        py::arg("verbose") = false,
        py::arg("transform_file") = std::string(""));

    m.def("ams_point_interpolant_3d",
        &pointInterpolant<3, double>,
        py::return_value_policy::take_ownership,
        "Fits a function to a set of sample values and/or gradients by finding the coefficients of the function that minimize an energy composed of a point-wise interpolation term and Laplacian and bi-Laplacian smoothness terms.",
        py::arg("positions_values") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("values") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("positions_gradients") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("gradients") = py::array_t<double, py::array::c_style | py::array::forcecast>(),
        py::arg("depth") = 8,
        py::arg("degree") = DEFAULT_FEM_DEGREE,
        py::arg("solve_depth") = -1,
        py::arg("full_depth") = 5,
        py::arg("base_depth") = -1,
        py::arg("b_type") = DEFAULT_FEM_BOUNDARY + 1,
        py::arg("iters") = 8,
        py::arg("base_v_cycles") = 4,
        py::arg("max_memory_gb") = 0,
        py::arg("parallel") = 0,
        py::arg("schedule") = (int)ThreadPool::Schedule,
        py::arg("thread_chunk_size") = (int)ThreadPool::ChunkSize,
        py::arg("value_weight") = 1000.0f,
        py::arg("gradient_weight") = 1.0f,
        py::arg("scale") = 1.1f,
        py::arg("width") = 0.f,
        py::arg("cg_accuracy") = 1e-3f,
        py::arg("iso") = 0.f,
        py::arg("lap_weight") = 0.f,
        py::arg("bi_lap_weight") = 1.f,
        py::arg("show_performance") = false,
        py::arg("show_residual") = false,
        py::arg("exact_interpolation") = true,
        py::arg("verbose") = false,
        py::arg("transform_file") = std::string(""));

    m.def("get_parallel_types",
        &getParallelTypes,
        "Returns the list of available parallelization strategy types (corresponds to the `parallel` argument).");

    bind_interpolant<2, double>(m, "Interpolant2D");
    bind_interpolant<3, double>(m, "Interpolant3D");
}