using Pkg
Pkg.activate("EnvFerrite")

using TypedPolynomials
@polyvar x y
p = 2x*y^2 +y -2x
q = 3x*y^3 - 4x + 4

# This function is correct.
function nabladotnabla(a,b)
    diff_a = differentiate(a,(x,y))
    diff_b = differentiate(b,(x,y))
    # There was a small typo in your original function here (diff_b[2] twice)
    # Correcting it to diff_a[2]*diff_b[2]
    return diff_a[1]*diff_b[1] + diff_a[2]*diff_b[2]
end
r = nabladotnabla(p,q)

# This function is correct.
function get_func(p)
    (z) -> p(x=>z[1],y=>z[2])
end

p_func = get_func(p)
q_func = get_func(q)
r_func = get_func(r)

using Ferrite
using SparseArrays
using LinearAlgebra

grid = generate_grid(Quadrilateral, (16, 16));
dim = Ferrite.getspatialdim(grid)

# --- FIX 1: Increase polynomial and quadrature order ---
order = 3 # Increased from 1 to 3 to exactly represent p and q
ip = Lagrange{RefQuadrilateral, order}()
qr = QuadratureRule{RefQuadrilateral}(5) # Increased from 2 to 5 for accurate integration
# ----------------------------------------------------

cellvalues = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# This function is correct.
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
M,MC = assemble_M(cellvalues,dh)

# This function is correct.
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

p_vec = assemble_function_vector(cellvalues, dh, p_func, MC)
q_vec = assemble_function_vector(cellvalues, dh, q_func, MC)
r_vec = assemble_function_vector(cellvalues, dh, r_func, MC)

# Your function was correct all along! No changes are needed here.
# It correctly computes the L2 projection of (∇pₕ ⋅ ∇qₕ)
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

r_test = calculate_bilinear_map(p_vec,q_vec, cellvalues,dh,MC)

# --- FIX 2: Check against a much smaller tolerance ---
println(norm(r_vec - r_test))
@assert norm(r_vec - r_test) < 1e-9
# ----------------------------------------------------