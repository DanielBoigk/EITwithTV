# This file contains all the functions we need. My Jupyter-Notebook became too long.
using Ferrite
using SparseArrays
using LinearAlgebra
using Revise
using Interpolations
using Plots
using Statistics
using IterativeSolvers
using LinearMaps

# First the structs:
mutable struct EITMode
    u::AbstractVector
    λ::AbstractVector
    δσ::AbstractVector
    f::AbstractVector
    g::AbstractVector
    rhs::AbstractVector
    error::Float64
    length::Int64
    m::Int64
end
function EITMode(g::AbstractVector, f::AbstractVector)
    L = length(g)
    M = length(f)
    return EITMode(zeros(L), zeros(L), zeros(L), f, g, zeros(L), 0.0, L, M)
end



# Let's start witch all the assemblers:


# This is the mass matrix: ∫(u*v)dΩ
# Also used for Tikhonov L² regularization
# Mainly we need this as a projector unto FE Space 
function assemble_M(cellvalues::CellValues,dh::DofHandler)
    M = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Me, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)      
            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    φⱼ = shape_value(cellvalues, q_point, j)
                    Me[i,j] += φᵢ * φⱼ * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Me)
    end
    return M, cholesky(M)
end   


# This is: ∫(∇(u)⋅∇(v))dΩ the stiffness matrix without specified coefficients. It is used for Tikhonov H¹ regularization. 
function assemble_K(cellvalues::CellValues, dh::DofHandler)
    K = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K, cholesky(K)
end



# This is matrix assembly on a function. How do I do it if the conductivity is given as a coefficient vector for Q1 FE Space?
# This is: ∫(γ * ∇(u)⋅∇(v))dΩ 
function assemble_K(cellvalues::CellValues, dh::DofHandler, γ, ϵ = 0.0)
    K = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            x = spatial_coordinate(cellvalues, q, getcoordinates(cell))
            σ = γ(x)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q, j)
                    Ke[i, j] += σ * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    if ϵ ≠ 0.0
        K += ϵ * I
    end
    return K
end

function assemble_K(cellvalues::CellValues, dh::DofHandler, γ, ϵ = 0.0)
    K = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            x = spatial_coordinate(cellvalues, q, getcoordinates(cell))
            σ = γ(x)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q, j)
                    Ke[i, j] += σ * (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    if ϵ ≠ 0.0
        K += ϵ * I
    end
    return K
end


# This function assembles the stiffness matrix from a given vector.
# This is: ∫(γ * ∇(u)⋅∇(v))dΩ 
function assemble_K(cellvalues::CellValues, dh::DofHandler, γ::AbstractVector,ϵ::Float64)
    K = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            γe = γ[celldofs(cell)] # (Edit) Could be done more efficiently by copying into preallocated array
            σ = function_value(cellvalues, q_point, γe)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                #u = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += σ* (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    if ϵ ≠ 0.0
        K += ϵ * I
    end
    return K
end

function assemble_K(cellvalues::CellValues, dh::DofHandler, γ::AbstractVector)
    K = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            γe = γ[celldofs(cell)] # (Edit) Could be done more efficiently by copying into preallocated array
            σ = function_value(cellvalues, q_point, γe)
            for i in 1:n_basefuncs
                ∇v = shape_gradient(cellvalues, q_point, i)
                #u = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += σ* (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    return K
end


function assemble_function_vector(cellvalues::CellValues, dh::DofHandler, f, M_cholesky)
    F = zeros(ndofs(dh))
    n_basefuncs = getnbasefunctions(cellvalues)
    Fe = zeros(n_basefuncs)
    cdofs = zeros(Int, n_basefuncs)

    for cell in CellIterator(dh)
        fill!(Fe, 0.0)
        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        cdofs = celldofs(cell)
        for q in 1:getnquadpoints(cellvalues)
            x_q = spatial_coordinate(cellvalues, q, coords)
            f_val = f(x_q)
            dΩ = getdetJdV(cellvalues, q)

            for i in 1:n_basefuncs
                Fe[i] += f_val * shape_value(cellvalues, q, i) * dΩ
            end
        end  
        assemble!(F, cdofs,Fe)
    end
    return M_cholesky \ F
end

# Assemble right-hand side for the projection of ∇(u) ⋅ ∇(λ) onto the FE space.
# This computes rhs_i = ∫ (∇u ⋅ ∇λ) ϕ_i dΩ for each test function ϕ_i.
# Assuming u and λ are scalar fields in the same FE space.
# cellvalues should be CellScalarValues(qr, ip) where qr is QuadratureRule, ip is Interpolation.
function calculate_bilinear_map(a::AbstractVector, b::AbstractVector, cellvalues::CellValues, dh::DofHandler, M_cholesky)
    n = ndofs(dh)
    rhs = zeros(n)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    re = zeros(n_basefuncs)
    
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(re, 0.0)
        
        ae = a[dofs]
        be = b[dofs]
        
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            
            ∇a_q = zero(Vec{2,Float64})
            ∇b_q = zero(Vec{2,Float64})
            
            for j in 1:n_basefuncs
                ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                ∇a_q += ae[j] * ∇ϕⱼ
                ∇b_q += be[j] * ∇ϕⱼ
            end
            
            grad_dot_product = ∇a_q ⋅ ∇b_q
            
            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                re[i] += grad_dot_product * ϕᵢ * dΩ
            end
        end
        assemble!(rhs, dofs, re)
    end
    
    return M \ rhs
end

function calculate_bilinear_map!(out::AbstractVector , rhs::AbstractVector, a::AbstractVector, b::AbstractVector, cellvalues::CellValues, dh::DofHandler, M_cholesky)
    n = ndofs(dh)
    fill!(rhs,0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    re = zeros(n_basefuncs)
    
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(re, 0.0)
        
        ae = a[dofs]
        be = b[dofs]
        
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            
            ∇a_q = zero(Vec{2,Float64})
            ∇b_q = zero(Vec{2,Float64})
            
            for j in 1:n_basefuncs
                ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                ∇a_q += ae[j] * ∇ϕⱼ
                ∇b_q += be[j] * ∇ϕⱼ
            end
            
            grad_dot_product = ∇a_q ⋅ ∇b_q
            
            for i in 1:n_basefuncs
                ϕᵢ = shape_value(cellvalues, q, i)
                re[i] += grad_dot_product * ϕᵢ * dΩ
            end
        end
        assemble!(rhs, dofs, re)
    end
    
    out = M \ rhs
    return out
end


# These are functions for calculating the gradient of our minimiaation functional:

# Use this for real problems:
function state_adjoint_step_cg!(mode::EITMode, K::AbstractMatrix, M, d,∂d ,down,up,dh::DofHandler, cellvalues::CellValues, maxiter=500)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    cg!(mode.u,K, mode.g; maxiter = maxiter)
    b = down(mode.u)
    mean = Statistics.mean(b) 
    b .-= mean
    mode.u .-= mean 
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂u/∂𝐧 = ∂ₓd(u,f)
    cg!(mode.λ, K, up(∂d(b,mode.f)); maxiter = maxiter)
    mode.error = d(b,mode.f)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here: 
    mode.δσ = calculate_bilinear_map(mode.λ, mode.u, cellvalues, dh, M)    
    # Check whether this needs + or - as a sign.
end
# Use this for toy problems:
function state_adjoint_step_fac!(mode::EITMode, K_factorized, M, d,∂d ,down,up,dh::DofHandler, cellvalues::CellValues)
    # We solve the state equation ∇⋅(σ∇uᵢ) = 0 : σ∂u/∂𝐧 = g
    mode.u = K_factorized \ mode.g
    # Projection from down:Ω → ∂Ω
    b = down(mode.u) 
    # Normalize: ∫(uᵢ)d∂Ω = 0
    mean = Statistics.mean(b) 
    b .-= mean
    mode.u .-= mean 
    # We solve the adjoint equation ∇⋅(σ∇λᵢ) = 0 : σ∂u/∂𝐧 = ∂ₓd(u,f)
    mode.λ = K_factorized \ up(∂d(b,mode.f)) 
    # Note: we have projection up: ∂Ω → Ω (fill in zeros)
    # ∂ₓd is gradient of pseudo metric d(x,y)
    mode.error = d(b,mode.f) # Error according to pseudo metric d(x,y)
    # Calculate ∇(uᵢ)⋅∇(λᵢ) here: 
    mode.δσ = calculate_bilinear_map(mode.λ, mode.u, cellvalues, dh, M)    
    # Check whether this needs + or - as a sign.
end



# These are the functions we need for regularization:

# This is all for TV regularization
mutable struct TV
    δ::AbstractArray # Is supposed to hold the error
    rhs::AbstractArray # 
    err_vec::AbstractArray
    error::Float64
    total_residual::Float64
    total_volume::Float64
end
function TV(n::Int64)
    TV(zeros(n), zeros(n), zeros(n), 0.0, 0.0, 0.0)
end
# This is the function which has a differentiable FEM version of the Total Variation Error and Gradient:
function calc_tv_gradient!(σ::AbstractVector,tv::TV, cellvalues::CellValues, dh::DofHandler, M, ε::Float64 = 1e-8)
    n = ndofs(dh)
    rhs = tv.rhs
    
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    re = zeros(n_basefuncs)
    total_residual = 0.0 # Put into init_tv(...) function
    total_volume = 0.0 # Put into init_tv(...) function
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(re, 0.0)
        ue = σ[dofs]
        for q in 1:qpoints
            dΩ = getdetJdV(cellvalues, q)
            total_volume += dΩ
            ∇u_q = zero(Vec{2,Float64})  # assuming 2D
            for j in 1:n_basefuncs
                ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                ∇u_q += ue[j] * ∇ϕⱼ
            end
            grad_norm_sq = ∇u_q ⋅ ∇u_q + ε^2
            grad_norm = sqrt(grad_norm_sq)
            ∇u_normalized = ∇u_q / grad_norm
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                re[i] += -∇u_normalized ⋅ ∇ϕᵢ * dΩ
            end
            total_residual += grad_norm * dΩ
        end 
        assemble!(rhs, dofs, re)
    end
    tv.δ = M \ rhs
    
    tv.error = total_residual / total_volume
    
    return tv.δ, tv.error
end
# It looks correct and logical tome however I should think about devising a test on whether that 










# Here comes lot's of the glue functions:
# This function given a DofHandler and the boundary definition given us two operators that project project a vector unto a longer version (force vector) or a shorter version (values on boundary)
function produce_nonzero_positions(v, atol=1e-8, rtol=1e-5)
    approx_zero(x; atol=atol, rtol=rtol) = isapprox(x, 0; atol=atol, rtol=rtol)
    non_zero_count = count(x -> !approx_zero(x), v)
    non_zero_positions = zeros(Int, non_zero_count)
    non_zero_indices = findall(x -> !approx_zero(x), v)
    g_down = (x) -> x[non_zero_indices]
    g_up = (x) -> begin
        v = zeros(eltype(x), length(v))
        v[non_zero_indices] = x
        return v
    end
    return non_zero_count, non_zero_positions, g_down, g_up
end
function produce_nonzero_positions(facetvalues::FacetValues, dh::DofHandler, ∂Ω)
    f = zeros(ndofs(dh))
        for facet in FacetIterator(dh, ∂Ω)
        fe = zeros(ndofs_per_cell(dh))
        reinit!(facetvalues, facet)
        for q_point in 1:getnquadpoints(facetvalues)
            dΓ = getdetJdV(facetvalues, q_point)            
            for i in 1:getnbasefunctions(facetvalues)
                δu = shape_value(facetvalues, q_point, i)
                fe[i] += δu * dΓ
            end
        end
        assemble!(f, celldofs(facet), fe)
    end
     nzc, nzpos, g_down, g_up = produce_nonzero_positions(f)
     return  nzc, nzpos, g_down, g_up, f
end

# Simple function that returns mean zero noise of specified length
function mean_zero_noise(n::Int64, σ::Float64)
    out = σ * randn(n)
    mean = Statistics.mean(out)
    out .- mean
end


# Does SVD on potential node vectors:
# reduce the number of modes according to used SVD modes
function do_svd(F,G)
    Λ = F * G'
    num_modes = (size(Λ, 1) - 1)
    V,Σ,U = svd(Λ)
    U = U[:,1:num_modes]
    V = V[:,1:num_modes]
    col_means = mean(U, dims=1)
    U .-= col_means
    col_means = mean(V, dims=1)
    V .-= col_means
    # Not unimportant:
    V = V * diagm(Σ[1:num_modes]) 
    return V, Σ, U, Λ, num_modes
end


# Now the Optimizers:
#Let's start with Gauss-Newton
mutable struct GaussNewtonState
    J::AbstractArray
    r::AbstractVector
    δ::AbstractVector
    M::LinearMaps.WrappedMap
end
function GaussNewtonState(n::Int64,k::Int64) # Just some default constructor
    J = zeros(k,n)
    r = zeros(k)
    δ = n
    M = LinearMap(I(n))
    GaussNewtonState(J,r,δ,M)
end

function gauss_newton_cg!(gns::GaussNewtonState, α::Float64 = 1.0, maxiter = 500)
    J = gns.J
    r = gns.r
    M = gns.M
    J_map = LinearMap(J)
    if α ≠ 0.0
        A_map = J_map' * J_map + α * M
    else
        A_map = J_map' * J_map
    end
    b = -(J' * r)
    cg!(gns.δ, A_map, b; maxiter = maxiter)
    gns.δ
end
# for reference with svd but only with Levenberg Marquardt
function gauss_newton_svd(J::Matrix{Float64}, r::Vector{Float64}; λ::Float64=1e-3)
    U, Σ, V = svd(J, full=false)
    n = length(Σ)
    Σ_damped = zeros(n)
    for i in 1:n
        Σ_damped[i] = Σ[i] / (Σ[i]^2 + λ) # Levenberg-Marquardt regularization
    end
    V * (Σ_damped .* (U' * r))
end
function gauss_newton_svd!(gns::GaussNewtonState; λ::Float64=1e-3)
    J = gns.J
    r = gns.r
    U, Σ, V = svd(J, full=false)
    n = length(Σ)
    Σ_damped = zeros(n)
    for i in 1:n
        Σ_damped[i] = Σ[i] / (Σ[i]^2 + λ) # Levenberg-Marquardt regularization
    end
    gns.δ = V * (Σ_damped .* (U' * r))
end
function update_M!(gns::GaussNewtonState, regularizers::Tuple{Float64, <:AbstractMatrix}...)
    if isempty(regularizers)
        # Handle case with no regularizers, e.g., set M to a zero map
        n = size(gns.J, 2)
        gns.M = LinearMap(zeros(n, n))
        return
    end

    n = size(regularizers[1][2], 1)
    M_sum = zeros(n, n)

    # Calculate the weighted sum: M_sum = Σ (λᵢ * Mᵢ)
    for (λ, M) in regularizers
        @assert size(M) == (n, n) "All M matrices must have the same dimensions."
        M_sum .+= λ .* M
    end
    gns.M = LinearMap(M_sum, issymmetric=true)
end

function update_Jr(gns, mode_dict, n::Int)
    k = length(mode_dict)
    if n>0
        k = min(n,k)
    end
    
    size = size(gns.J)
    if size[1] != k
        gns.J = zeros(k, size[2])
        gns.r = zeros(k)  
    end
    
    for i in  in 1:k
        gns.J[i,:] = mode_dict[i].δσ
        gns.r[i] = mode_dict[i].error
    end
end
