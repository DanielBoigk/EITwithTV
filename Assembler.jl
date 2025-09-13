


# Assembly functions:
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

# Function -> Coefficient Vector : Interpolates function f(x) as FE coefficients
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


# This is matrix assembly for stiffness matrix K on a function.
# Hopefully this is: ∫(γ * ∇(u)⋅∇(v))dΩ 
function assemble_K(cellvalues::CellValues, dh::DofHandler, γ)
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
    return K
end

# This function assembles the stiffness matrix from a given vector.
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
            # How do I get the index x of the vector γ?
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

# This assembles ∫(g*v)d∂Ω
# But we don't really need this function
function assemble_rhs_func(facetvalues::FacetValues, dh::DofHandler, g_func, ∂Ω)
    f = zeros(ndofs(dh))
    fe = zeros(ndofs_per_cell(dh))
    for facet in FacetIterator(dh, ∂Ω)
        fill!(fe,0.0)
        reinit!(facetvalues, facet)
        coords = getcoordinates(facet)
        dofs = celldofs(facet)
        for q_point in 1:getnquadpoints(facetvalues)
            x = spatial_coordinate(facetvalues, q_point, coords)
            g = g_func(x)
            dΓ = getdetJdV(facetvalues, q_point)
            for i in 1:getnbasefunctions(facetvalues)
                ϕᵢ = shape_value(facetvalues, q_point, i)
                fe[i] += ϕᵢ * g * dΓ
            end
        end
        assemble!(f, dofs, fe)
    end
    return f
end


function calculate_bilinear_map_rhs(rhs::AbstractVector,a::AbstractVector,b::AbstractVector, cellvalues::CellValues,dh::DofHandler)
    n = ndofs(dh)
    fill!(rhs,0.0) # Initialize the output vector
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)

    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues,cell)

        for q in 1:qpoints 
            for i in 1:n_basefuncs
                for i in 1:n_basefuncs
                    re[] = ae[i]*be[j]*(∇ϕᵢ⋅∇ϕⱼ)* dΩ
                end
            end
        end
        assemble! (rhs, dofs, re)
    end


end
