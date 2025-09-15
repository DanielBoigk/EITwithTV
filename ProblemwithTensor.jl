using Pkg
Pkg.activate("EnvFerrite")

using TypedPolynomials
@polyvar x y
p = 2x*y^2 +y -2x
q = 3x*y^3 - 4x + 4
function nabladotnabla(a,b)
    diff_a = differentiate(a,(x,y))
    diff_b = differentiate(b,(x,y))
    return diff_a[1]*diff_b[1]+diff_b[2]*diff_b[2]
end
r = nabladotnabla(p,q)

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
order = 1

ip = Lagrange{RefQuadrilateral, order}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# for that we also gonna assemble the Mass Matrix:
# Mass Matrix  ∫(u*v)dΩ
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

# now we gonna project conductivity unto a vector:
function assemble_function_vector(cellvalues::CellValues, dh::DofHandler, f, M)
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
    return M \ F

end

p_vec = assemble_function_vector(cellvalues, dh, p_func, MC)
q_vec = assemble_function_vector(cellvalues, dh, q_func, MC)
r_vec = assemble_function_vector(cellvalues, dh, r_func, MC)

# Write assembler here for: ∇(a)⋅∇(b)
# All the syntax exists but this function is incorrect:
function calculate_bilinear_map_rhs(a::AbstractVector,b::AbstractVector, cellvalues::CellValues,dh::DofHandler, M)
    n = ndofs(dh)
    rhs = zeros(n)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    ae = zeros(n_basefuncs)
    be = zeros(n_basefuncs)
    re = zeros(n_basefuncs)
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues,cell)
        for q in 1:qpoints 
            dΩ = getdetJdV(cellvalues, q)
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)    
                ϕᵢ = shape_value(cellvalues, q, i)
                for j in 1:n_basefuncs
                    ∇ϕⱼ = shape_gradient(cellvalues, q, j)
                    re[i] = ae[i]*be[j]*(∇ϕᵢ⋅∇ϕⱼ)* dΩ
                end
            end
        end
        assemble!(rhs, dofs, re)
    end
    return M \ rhs
end

r_test = calculate_bilinear_map_rhs(p_vec,q_vec, cellvalues,dh,MC)

@assert norm(r_vec - r_test) < 10.0