function assemble_tv_gradient(u_vec::AbstractVector, cellvalues::CellValues, dh::DofHandler, M, ε::Float64 = 1e-8)
    n = ndofs(dh)
    rhs = zeros(n)
    n_basefuncs = getnbasefunctions(cellvalues)
    qpoints = getnquadpoints(cellvalues)
    re = zeros(n_basefuncs)
    total_residual = 0.0
    total_volume = 0.0
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        reinit!(cellvalues, cell)
        fill!(re, 0.0)
        ue = u_vec[dofs]
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
            # Normalized gradient: ∇u/|∇u|_ε
            ∇u_normalized = ∇u_q / grad_norm
            for i in 1:n_basefuncs
                ∇ϕᵢ = shape_gradient(cellvalues, q, i)
                re[i] += -∇u_normalized ⋅ ∇ϕᵢ * dΩ
            end
            total_residual += grad_norm * dΩ
        end 
        assemble!(rhs, dofs, re)
    end
    tv_grad_vec = M \ rhs
    
    error_estimate = total_residual / total_volume
    
    return tv_grad_vec, error_estimate
end