# note: If you want to use truncated SVD as regularization one can pass a smaller number than num_modes
function cheat_step!(M,γ,σ::AbstractVector ,modes::Dict{Int64,EITMode}, num_modes::Int64,tv::TV,  d,∂d ,down,up,dh::DofHandler, cellvalues::CellValues, do_TV::Bool =true, β::Float64 = 1e-5)
    # Assemble Matrix: (from vector)
    K = assemble_K(cellvalues,dh,σ)
    if do_TV
        J = zeros(num_modes+1,ndofs(dh))
        r = zeros(num_modes+1)
        # Launch TV regularizer:
        tv_task = Threads.@spawn begin
            calc_TV_step!(σ,tv, dh,cellvalues,M)
        end
    else
        J = zeros(num_modes,ndofs(dh))
        r = zeros(num_modes)
    end
    # solve adjoint state method
    Threads.@threads for i in 1:num_modes
        state_adjoint_step!(mode_dict[i], K, M,  d,∂d ,down,up,dh, cellvalues)
    end

    # Fetch gradients & errors
    for i in 1:num_modes
        J[i,:] = mode_dict[i].δσ
        r[i] = mode_dict[i].error
    end
    if do_TV
        # Fetch TV regularization
        fetch(tv_task)
        J[num_modes+1,:] = tv.δ
        r[num_modes+1] = β * tv.error 
    end    
    # calculate steps with Gauss-Newton
    δσ = gauss_newton(J, r, λ=1e-3)
    # update σ
    α = - dot(γ-σ, δσ )
    σ .-= α*δσ
    σ .= max.(σ ,1e-12) # Ensure positivity
    return δσ,r,J
end

function full_step_initial!(M,σ::AbstractVector ,modes::Dict{Int64,EITMode}, num_modes::Int64,tv::TV,  d,∂d ,down,up, dh::DofHandler, cellvalues::CellValues)
    # Assemble Matrix: (from vector)
    K = assemble_K(cellvalues,dh,σ)
    K_LU = lu(K)
    J = zeros(num_modes,ndofs(dh))
    r = zeros(num_modes)
    # solve adjoint state method
    Threads.@threads for i in 1:num_modes
        state_adjoint_step!(mode_dict[i], K, M,  d, ∂d ,down , up, dh, cellvalues)
    end

    # Fetch gradients & errors
    for i in 1:num_modes
        J[i,:] = mode_dict[i].δσ
        r[i] = mode_dict[i].error
    end
    # calculate steps with Gauss-Newton
    δσ = gauss_newton(J, r, λ=1e-3)
    # update σ
    σ .+=  δσ
    σ .= max.(σ ,1e-12) # Ensure positivity
    return K,δσ,r,J
end