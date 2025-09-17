using Pkg
Pkg.activate("EnvFerrite")

function create_test_function()
    u_analytical(coords) = sin(π * coords[1]) * cos(π * coords[2])
    
    # Analytical gradient: ∇u = [π*cos(πx)*cos(πy), -π*sin(πx)*sin(πy)]
    function grad_u_analytical(coords)
        x, y = coords[1], coords[2]
        return [π * cos(π * x) * cos(π * y), -π * sin(π * x) * sin(π * y)]
    end
    
    # Analytical TV gradient: ∇·(∇u/|∇u|)
    # This is complex, so we'll use finite differences for verification
    function tv_gradient_analytical_approx(coords)
        h = 1e-6
        x, y = coords[1], coords[2]
        
        # Compute |∇u| at neighboring points
        grad_center = grad_u_analytical(coords)
        grad_norm_center = norm(grad_center)
        
        if grad_norm_center < 1e-12
            return 0.0  # Handle singularity
        end
        
        # Finite difference approximation of ∇·(∇u/|∇u|)
        # ∂/∂x (∇u_x/|∇u|) + ∂/∂y (∇u_y/|∇u|)
        
        grad_xplus = grad_u_analytical([x + h, y])
        grad_xminus = grad_u_analytical([x - h, y])
        grad_yplus = grad_u_analytical([x, y + h])
        grad_yminus = grad_u_analytical([x, y - h])
        
        norm_xplus = norm(grad_xplus)
        norm_xminus = norm(grad_xminus)
        norm_yplus = norm(grad_yplus)
        norm_yminus = norm(grad_yminus)
        
        # Avoid division by zero
        ux_norm_xplus = norm_xplus > 1e-12 ? grad_xplus[1] / norm_xplus : 0.0
        ux_norm_xminus = norm_xminus > 1e-12 ? grad_xminus[1] / norm_xminus : 0.0
        uy_norm_yplus = norm_yplus > 1e-12 ? grad_yplus[2] / norm_yplus : 0.0
        uy_norm_yminus = norm_yminus > 1e-12 ? grad_yminus[2] / norm_yminus : 0.0
        
        # Central differences
        d_dx_term = (ux_norm_xplus - ux_norm_xminus) / (2 * h)
        d_dy_term = (uy_norm_yplus - uy_norm_yminus) / (2 * h)
        
        return d_dx_term + d_dy_term
    end
    
    return u_analytical, grad_u_analytical, tv_gradient_analytical_approx
end