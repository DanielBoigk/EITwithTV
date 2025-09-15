using Ferrite
using SparseArrays
using LinearAlgebra
using IterativeSolvers


grid = generate_grid(Quadrilateral, (16, 16));
dim = Ferrite.getspatialdim(grid)
order = 1

ip = Lagrange{RefQuadrilateral, order}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# first we are gonna define just some function on [-1,1]^2

conductivity = (x) -> 1.1 + sin(x[1]) * cos(x[2])

# With that we are gonna assemble a stiffness matrix:
# Hopefully this is: ∫(γ * ∇(u)⋅∇(v))dΩ 
function assemble_K(cellvalues::CellValues, dh::DofHandler, γ)
    K = allocate_matrix(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        for q in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q)
            x = spatial_coordinate(cellvalues, q, coords)
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
    return K
end
K_func = assemble_K(cellvalues, dh, conductivity)


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
        println(coords)
        cdofs = celldofs(cell)
        for q in 1:getnquadpoints(cellvalues)
            x_q = spatial_coordinate(cellvalues, q, coords)
            f_val = f(x_q)
            dΩ = getdetJdV(cellvalues, q)

            for i in 1:n_basefuncs
                Fe[i] += f_val * shape_value(cellvalues, q, i) * dΩ
            end
        end

        
        # Assemble local vector into global vector
  
        assemble!(F, cdofs,Fe)
    end
    return M \ F

end

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
cond_vec = assemble_function_vector(cellvalues, dh, conductivity, MC)

# now we go on to define the assembler that assembles K from a vector.
# Hopefully this is: ∫(γ * ∇(u)⋅∇(v))dΩ 
function assemble_K!(K::SparseMatrixCSC ,cellvalues::CellValues, dh::DofHandler, γ::AbstractVector)
    fill!(K,0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            # How do I get the index of the vector γ?
            σ = γ[cellid(cell)]
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
function assemble_K(cellvalues::CellValues, dh::DofHandler, γ::AbstractVector)
    K = allocate_matrix(dh)
    return assemble_K!(K, cellvalues, dh, γ)
end
K_vec = assemble_K(cellvalues, dh,cond_vec)

# Implement a sanity check if the two matrices assembled from the function and the vector are roughly the same (use relatively coarse ≈ )
Matrix_norm = norm(K_vec - K_func)
println("Norm of Matrix difference: ",Matrix_norm)
@assert Matrix_norm < 10.0

