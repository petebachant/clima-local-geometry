using ClimaCore: Geometry, Operators, MatrixFields
import ClimaCore
import LazyBroadcast: lazy

# Alternatively, we could use Vec₁₂₃, Vec³, etc., if that is more readable.
const C1 = Geometry.Covariant1Vector
const C2 = Geometry.Covariant2Vector
const C12 = Geometry.Covariant12Vector
const C3 = Geometry.Covariant3Vector
const C123 = Geometry.Covariant123Vector
const CT1 = Geometry.Contravariant1Vector
const CT2 = Geometry.Contravariant2Vector
const CT12 = Geometry.Contravariant12Vector
const CT3 = Geometry.Contravariant3Vector
const CT123 = Geometry.Contravariant123Vector
const UVW = Geometry.UVWVector

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const split_divₕ = Operators.SplitDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
const ᶜdivᵥ = Operators.DivergenceF2C()
const ᶜgradᵥ = Operators.GradientF2C()

# Tracers do not have advective fluxes through the top and bottom cell faces.
const ᶜadvdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.SetValue(CT3(0)),
)

# Subsidence has extrapolated tendency at the top, and has no flux at the bottom.
# TODO: This is not accurate and causes some issues at the domain top.
const ᶜsubdivᵥ = Operators.DivergenceF2C(
    bottom = Operators.SetValue(CT3(0)),
    top = Operators.Extrapolate(),
)

# Precipitation has no flux at the top, but it has free outflow at the bottom.
const ᶜprecipdivᵥ = Operators.DivergenceF2C(top = Operators.SetValue(CT3(0)))

const ᶠright_bias = Operators.RightBiasedC2F() # for free outflow in ᶜprecipdivᵥ
const ᶜleft_bias = Operators.LeftBiasedF2C()
const ᶜright_bias = Operators.RightBiasedF2C()

# TODO: Implement proper extrapolation instead of simply reusing the first
# interior value at the surface.
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

# TODO: Replace these boundary conditions with NaN's, since they are
# meaningless and we only need to specify them in order to be able to
# materialize broadcasts. Any effect these boundary conditions have on the
# boundary values of Y.f.u₃ is overwritten when we call set_velocity_at_surface!.
# Ideally, we would enforce the boundary conditions on Y.f.u₃ by filtering it
# immediately after adding the tendency to it. However, this is not currently
# possible because our implicit solver is unable to handle filtering, which is
# why these boundary conditions are 0's rather than NaN's.
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(C3(0)),
    top = Operators.SetGradient(C3(0)),
)
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(CT12(0, 0)),
    top = Operators.SetCurl(CT12(0, 0)),
)
const upwind_biased_grad = Operators.UpwindBiasedGradient()
const ᶠupwind1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)
@static if pkgversion(ClimaCore) ≥ v"0.14.22"
    const ᶠlin_vanleer = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = Operators.MonotoneLocalExtrema(), # (Mono5)
    )
end

const ᶜinterp_matrix = MatrixFields.operator_matrix(ᶜinterp)
const ᶜleft_bias_matrix = MatrixFields.operator_matrix(ᶜleft_bias)
const ᶜright_bias_matrix = MatrixFields.operator_matrix(ᶜright_bias)
const ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ᶜdivᵥ)
const ᶜadvdivᵥ_matrix = MatrixFields.operator_matrix(ᶜadvdivᵥ)
const ᶜprecipdivᵥ_matrix = MatrixFields.operator_matrix(ᶜprecipdivᵥ)
const ᶠright_bias_matrix = MatrixFields.operator_matrix(ᶠright_bias)
const ᶠinterp_matrix = MatrixFields.operator_matrix(ᶠinterp)
const ᶠwinterp_matrix = MatrixFields.operator_matrix(ᶠwinterp)
const ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ᶠgradᵥ)
const ᶠupwind1_matrix = MatrixFields.operator_matrix(ᶠupwind1)
const ᶠupwind3_matrix = MatrixFields.operator_matrix(ᶠupwind3)

# Helper functions to extract components of vectors
u_component(u::Geometry.LocalVector) = u.u
v_component(u::Geometry.LocalVector) = u.v
w_component(u::Geometry.LocalVector) = u.w

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;
using Test
using ClimaComms
ClimaComms.@import_required_backends
using StaticArrays, IntervalSets
import BenchmarkTools
import StatsBase
import DataStructures
using ClimaCore.Geometry: ⊗
import ClimaCore.DataLayouts

FT = Float32
horizontal_layout_type = ClimaCore.DataLayouts.IJFH
z_elems = 63
helem = 30
Nq = 4
cspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; zelem=z_elems, helem, Nq, horizontal_layout_type)
fspace = ClimaCore.Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
import ClimaCore: Domains, Meshes, Spaces, Fields, Operators, Topologies
import ClimaCore.Domains: Geometry

using BenchmarkTools
field_vars(::Type{FT}) where {FT} = (;
    x = FT(1),
    uₕ = Geometry.Covariant12Vector(FT(0), FT(0)),
    uₕ2 = Geometry.Covariant12Vector(FT(0), FT(0)),
    curluₕ = Geometry.Contravariant12Vector(FT(0), FT(0)),
    w = Geometry.Covariant3Vector(FT(0)),
    contra3 = Geometry.Contravariant3Vector(FT(0)),
    y = FT(1),
    D = FT(0),
    U = FT(0),
    ∇x = Geometry.Covariant3Vector(FT(0)),
    ᶠu³ = Geometry.Contravariant3Vector(FT(0)),
    ᶠuₕ³ = Geometry.Contravariant3Vector(FT(1)),
    ᶠw = Geometry.Covariant3Vector(FT(0)),
)
cfield = fill(field_vars(FT), cspace)
cfield2 = fill(field_vars(FT), cspace)
ffield = fill(field_vars(FT), fspace)
# @. ᶠlin_vanleer(ffield.ᶠu³, cfield.x, 0.1f0)
ᶜJ = Fields.local_geometry_field(axes(cfield)).J
 ᶠJ = Fields.local_geometry_field(axes(ffield)).J


# @. cfield.x = ᶜinterp(ᶠinterp($ᶜJ))

# @elapsed CUDA.@sync @. ffield.x = ᶠinterp(ᶜJ)
# @elapsed CUDA.@sync   @. cfield.x = ᶜinterp(ᶠinterp(ᶜJ))
# @elapsed CUDA.@sync     @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ)))
# @elapsed CUDA.@sync     @. cfield.x = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ))))
# @elapsed CUDA.@sync     @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ)))))
# @elapsed CUDA.@sync     @. cfield.x = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ))))))
# @elapsed CUDA.@sync     @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ)))))))

CUDA.@profile trace=true begin
    @. ffield.x = ᶠinterp(ᶜJ)
    @. cfield.x = ᶜinterp(ᶠinterp(ᶜJ))
    @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ)))
    @. cfield.x = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ))))
    @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ)))))
    @. cfield.x = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ))))))
    @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ)))))))
end

CUDA.@profile trace=true begin
    @. ffield.x = ᶠinterp(ᶜJ * cfield.y / cfield.x)
    @. cfield.x = ᶜinterp(ᶠinterp(ᶜJ * cfield.y / cfield.x))
    @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ * cfield.y / cfield.x)))
    @. cfield.x = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ * cfield.y / cfield.x))))
    @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ * cfield.y / cfield.x)))))
    @. cfield.x = ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ * cfield.y / cfield.x))))))
    @. ffield.x = ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ᶠinterp(ᶜJ * cfield.y / cfield.x)))))))
end
# # @. -(ᶜadvdivᵥ(ᶠinterp(ᶜJ) / ᶠJ) * ᶠlin_vanleer(ffield.ᶠu³, cfield.x, 0.1f0))
# # const ᶜgradᵥ = Operators.GradientF2C()

# @. ᶜgradᵥ(ᶠinterp(ᶜJ))
# # @. -(ᶜadvdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠlin_vanleer(ffield.ᶠu³, cfield.x, 0.1f0)))
# @. -(ᶜadvdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠupwind1(ffield.ᶠu³, cfield.x)))
# @.  ᶠgradᵥ(ᶜJ - ᶜinterp(ᶠJ))

# ρ_flux_uₕ_surface = fill(C3(FT(0)) ⊗ C12(FT(0), FT(0)), cspace)
# ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
#                top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
#                bottom = Operators.SetValue(ρ_flux_uₕ_surface),
#            )


# @. ᶜgradᵥ(ᶠwinterp((cfield.x + ᶜinterp(ffield.x)) / cfield.y, cfield.ᶠu³))
# @. ᶜgradᵥ(ᶠwinterp((cfield.x ) , cfield.ᶠu³))

# function big_ret(x)
#     return (x + 1.0f0, x - 1.9f0,  x + 1.0f0, x - 1.9f0,)
# end
# take_some(x) = x[1] + x[2]

# @. ᶠinterp(big_ret(take_some(big_ret((cfield.x + ᶜinterp(ffield.x + ffield.y + ffield.D)) / cfield.y + cfield.D + cfield.U + cfield2.y + cfield2.D + cfield2.U))))

# tri = fill(ClimaCore.MatrixFields.TridiagonalMatrixRow(0.0f0, 0.0f0, 0.0f0), fspace)
# tri2 = fill(ClimaCore.MatrixFields.TridiagonalMatrixRow(0.0f0, 0.0f0, 0.0f0), cspace)
# # out = @.  2.0f0 * tri * ClimaCore.MatrixFields.DiagonalMatrixRow(3.2f0 * cfield.x / cfield.y)
# using LinearAlgebra
# @.  (dot(C123(cfield.uₕ), CT123(cfield.w)) + ᶜinterp(dot(C123(ffield.uₕ), CT123(ffield.w))) + 2 * dot(C123(cfield.uₕ), CT123(cfield.w))) + cfield.x


# ᶜdivᵥ_q = Operators.DivergenceF2C(
#     top = Operators.SetValue(C3(FT(0))),
#     bottom = Operators.SetValue(C3(FT(0))),
# )

# ᶜa = @. lazy(cfield.y / cfield.x + cfield.D)
# @. ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(cfield.y / cfield.x + cfield.D)))



# # CUDA.@profile trace=true begin
#     @. ᶜinterp(ffield.uₕ / ffield.y * 1/ ᶠinterp(cfield.x))
#     @. ᶠinterp(ᶜinterp(ffield.uₕ / ffield.y * 1/ ᶠinterp(cfield.x)))
#     @. ᶜinterp(ᶠinterp(ᶜinterp(ffield.uₕ / ffield.y * 1/ ᶠinterp(cfield.x))))
#     @. ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ffield.uₕ / ffield.y * 1/ ᶠinterp(cfield.x)))))
# # end

# # CUDA.@profile trace=true begin
#     @. ᶜinterp(ffield.x^2 / ffield.y + ᶠinterp(cfield.x))
#     @. ᶠinterp(ᶜinterp(ffield.x^2 / ffield.y + ᶠinterp(cfield.x)))
#     @. ᶜinterp(ᶠinterp(ᶜinterp(ffield.x^2 / ffield.y + ᶠinterp(cfield.x))))
#     @. ᶠinterp(ᶜinterp(ᶠinterp(ᶜinterp(ffield.x^2 / ffield.y + ᶠinterp(cfield.x)))))
# # end

# # TODO: van leer

# bi_fs = (fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), fspace))

# bi_cs=(fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.BidiagonalMatrixRow(0.0f0, 0.0f0), cspace))

# di_fs = (fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), fspace))

# di_cs = (fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace))

# di_cs2 = (fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace))

# di_cs3 = (fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace),
#        fill(ClimaCore.MatrixFields.DiagonalMatrixRow(0.0f0), cspace))

# # CUDA.@profile trace=true begin
#     # @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]) + (bi_fs[7] * di_cs[7] * bi_cs[7]) + (bi_fs[8] * di_cs[8] * bi_cs[8]) + (bi_fs[9] * di_cs[9] * bi_cs[9])) - tri
#     # @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]) + (bi_fs[7] * di_cs[7] * bi_cs[7]) + (bi_fs[8] * di_cs[8] * bi_cs[8])) - tri
#     # @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]) + (bi_fs[7] * di_cs[7] * bi_cs[7])) - tri
#     @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6])) - tri
#     @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5])) - tri
#     @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4])) - tri
#     # @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]) + (bi_fs[3] * di_cs[3] * bi_cs[3])) - tri
#     # @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) - tri
#     # @. ((bi_fs[1] * di_cs[1] * bi_cs[1])) - tri
# # end


# # CUDA.@profile trace=true begin
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]) + (bi_fs[7] * bi_cs[7]) + (bi_fs[8] * bi_cs[8]) + (bi_fs[9] * bi_cs[9]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]) + (bi_fs[7] * bi_cs[7]) + (bi_fs[8] * bi_cs[8]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]) + (bi_fs[7] * bi_cs[7]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]))
# #     @. tri - ((bi_fs[1] * bi_cs[1]))
# # end


# # CUDA.@profile trace=true begin
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]) + (bi_fs[7] * bi_cs[7]) + (bi_fs[8] * bi_cs[8]) + (bi_fs[9] * bi_cs[9]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]) + (bi_fs[7] * bi_cs[7]) + (bi_fs[8] * bi_cs[8]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]) + (bi_fs[7] * bi_cs[7]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]) + (bi_fs[6] * bi_cs[6]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]) + (bi_fs[5] * bi_cs[5]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]) + (bi_fs[4] * bi_cs[4]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]) + (bi_fs[3] * bi_cs[3]))
# #     @. ((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2]))
# #     @. ((bi_fs[1] * bi_cs[1]))
# # end


# # CUDA.@profile trace=true begin
# #     @. (((bi_fs[1] * bi_cs[1]) + (bi_fs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]) + (bi_fs[7] * di_cs[7] * bi_cs[7]) + (bi_fs[8] * di_cs[8] * bi_cs[8]) + (bi_fs[9] * di_cs[9] * bi_cs[9]))
# #     @. (((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]) + (bi_fs[7] * di_cs[7] * bi_cs[7]) + (bi_fs[8] * di_cs[8] * bi_cs[8]))
# #     @. (((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]) + (bi_fs[7] * di_cs[7] * bi_cs[7]))
# #     @. (((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]) + (bi_fs[6] * di_cs[6] * bi_cs[6]))
# #     @. (((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]) + (bi_fs[5] * di_cs[5] * bi_cs[5]))
# #     @. (((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]) + (bi_fs[4] * di_cs[4] * bi_cs[4]))
# #     @. (((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2])) + (bi_fs[3] * di_cs[3] * bi_cs[3]))
# #     @. ((bi_fs[1] * di_cs[1] * bi_cs[1]) + (bi_fs[2] * di_cs[2] * bi_cs[2]))
# #     @. ((bi_fs[1] * di_cs[1] * bi_cs[1]))
# # end

# # CUDA.@profile trace=true begin
# #     @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) + (di_cs[5] * di_cs2[5] * di_cs3[5]) + (di_cs[6] * di_cs2[6] * di_cs3[6]) + (di_cs[7] * di_cs2[7] * di_cs3[7]) + (di_cs[8] * di_cs2[8] * di_cs3[8]) + (di_cs[9] * di_cs2[9] * di_cs3[9]))
# #     @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) + (di_cs[5] * di_cs2[5] * di_cs3[5]) + (di_cs[6] * di_cs2[6] * di_cs3[6]) + (di_cs[7] * di_cs2[7] * di_cs3[7]) + (di_cs[8] * di_cs2[8] * di_cs3[8]))
# #     @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) + (di_cs[5] * di_cs2[5] * di_cs3[5]) + (di_cs[6] * di_cs2[6] * di_cs3[6]) + (di_cs[7] * di_cs2[7] * di_cs3[7]))
# #     @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) + (di_cs[5] * di_cs2[5] * di_cs3[5]) + (di_cs[6] * di_cs2[6] * di_cs3[6]))
# #     @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) + (di_cs[5] * di_cs2[5] * di_cs3[5]))
# #     @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) )
# #     # @. tri2 - ((di_cs[1] * di_cs2[1] * di_cs3[1]) + (di_cs[2] * di_cs2[2] * di_cs3[2]) + (di_cs[3] * di_cs2[3] * di_cs3[3]) + (di_cs[4] * di_cs2[4] * di_cs3[4]) + (di_cs[5] * di_cs2[5] * di_cs3[5]) + (di_cs[6] * di_cs2[6] * di_cs3[6]) + (di_cs[7] * di_cs2[7] * di_cs3[7]))

# # end
