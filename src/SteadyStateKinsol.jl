module SteadyStateKinsol

import PALEOboxes as PB

import PALEOmodel

import Sundials
using LinearAlgebra
import Infiltrator

using ForwardDiff
using SparseArrays
using SparseDiffTools


"""
    steadystate_ptc(run, initial_state, modeldata, tspan, deltat_initial; 
        [,deltat_fac=2.0] [,tss_output] [,outputwriter] [,createkwargs] [,solvekwargs]
        [, use_jac_preconditioner] [,jac_cellranges] [, use_directional_ad] [, directional_ad_eltypestomap]
        [,verbose] [,  BLAS_num_threads]
    )

Find steady-state solution and write to `outputwriter`, using naive pseudo-transient-continuation
with first order implicit Euler pseudo-timesteps and [`PALEOmodel.Kinsol`](@ref) as the non-linear solver.

Each pseudo-timestep solves the nonlinear system
S(t+Δt) = S(t) + Δt dS/dt(t+Δt)
for S(t+Δt), using a variant of Newton's method (preconditioned Newton-Krylov, with the Jacobian as preconditioner)

Initial pseudo-timestep Δt is `deltat_initial`, this is multiplied by `deltat_fac` for the next iteration
until pseudo-time `tss_max` is reached. If an iteration fails, Δt is divided by `deltat_fac` and the iteration retried.

NB: this is a _very_ naive initial implementation, there is currently no reliable error control to adapt pseudo-timesteps 
to the rate of convergence, so requires some trial-and-error to set an appropiate `deltat_fac` for each problem.

Solver [`PALEOmodel.Kinsol`](@ref) options are set by arguments `createkwargs`
(passed through to [`PALEOmodel.Kinsol.kin_create`](@ref))
and `solvekwargs` (passed through to [`PALEOmodel.Kinsol.kin_solve`](@ref)).

If `use_jac_ad_preconditioner` is `true`, the AD Jacobian is used as preconditioner.
The preconditioner (Jacobian) calculation can be modified by `jac_cellranges`, to specify a operatorIDs 
so use only a subset of Reactions in order to  calculate an approximate Jacobian to use as the preconditioner.

If `use_directional_ad` is `true`, the Jacobian-vector product will be calculated using automatic differentiation (instead of 
the default finite difference approximation). 
`directional_ad_eltypestomap` can be used to specify Variable :datatype tags (strings)
that should be mapped to the AD directional derivative datatype hence included in the AD directional derivative.
"""
function steadystate_ptc(
    run, initial_state, modeldata, tspan, deltat_initial::Float64; 
    deltat_fac=2.0,
    tss_output=[],
    ptc_iter_max=1000,
    outputwriter=run.output,
    createkwargs::NamedTuple=NamedTuple{}(), 
    solvekwargs::NamedTuple=NamedTuple{}(),
    use_jac_ad_preconditioner=true,
    jac_full=false,
    request_adchunksize=10,
    jac_cellranges=modeldata.cellranges_all,   
    use_directional_ad=false,
    directional_ad_eltypestomap=String[],
    verbose=false,
    BLAS_num_threads=1
)
    PB.check_modeldata(run.model, modeldata)

    # start, end times
    tss, tss_max = tspan

    # workaround Julia BLAS default (mis)configuration that defaults to multi-threaded
    LinearAlgebra.BLAS.set_num_threads(BLAS_num_threads)
    @info "steadystate_ptc:  using BLAS with $(LinearAlgebra.BLAS.get_num_threads()) threads"
 
    sv = modeldata.solver_view_all
    # We only support explicit ODE-like configurations (no DAE constraints or implicit variables)
    iszero(PALEOmodel.num_total(sv))                 || error("implicit total variables not supported")
    iszero(PALEOmodel.num_algebraic_constraints(sv)) || error("algebraic constraints not supported")

    if use_jac_ad_preconditioner
        # Define preconditioner setup function
        @info "steadystate:  using Jacobian :ForwardDiffSparse as preconditioner"
        jac, jac_prototype = PALEOmodel.JacobianAD.jac_config_ode(
            :ForwardDiffSparse, run.model, initial_state, modeldata, tss,
            request_adchunksize=request_adchunksize,
            jac_cellranges=jac_cellranges
        )    
    else
        jac, jac_prototype = nothing, []
    end

    # workspace arrays
    current_state   = copy(initial_state)
    deriv_worksp    = similar(initial_state)
    Jnorm = copy(jac_prototype)  # workspace array
    # current pseudo-timestep
    deltat = Ref(deltat_initial)
    tmodel = Ref(tss)
    # factorized Jacobian
    J_lu = Ref{Any}()

    if use_directional_ad
        directional_context = PALEOmodel.JacobianAD.directional_config(
            run.model, modeldata, modeldata.cellranges_all, eltypestomap=directional_ad_eltypestomap, use_base_transfer_jacobian=false,
        )
    else
        directional_context = NamedTuple()
    end

    userdata = merge(
        (;modeldata, jac, jac_prototype, jac_full),
        directional_context,
        (;deriv_worksp, current_state, Jnorm, deltat, tmodel, J_lu),
    )

    # calculate residual F = S - Sinit - deltat*dS/dt    
    function ssf!(resid, u, d)       
             
        
        PALEOmodel.set_tforce!(d.modeldata.solver_view_all, d.tmodel[])
        PALEOmodel.set_statevar!(d.modeldata.solver_view_all, u) 

        PB.do_deriv(d.modeldata.dispatchlists_all)

        PALEOmodel.get_statevar_sms!(d.deriv_worksp, d.modeldata.solver_view_all)
        resid .=  (u .- d.current_state .- d.deltat[].*d.deriv_worksp)

        return nothing
    end


    function psolve(
        u, uscale, 
        fval, fscale,
        v, d
    )
            
        retval = 1 # recoverable error

        if isdefined(d.J_lu, 1) && issuccess(d.J_lu[])           
            v .= d.J_lu[] \ v
            retval = 0
        end

        return retval
    end

    # Calculate and factor sparse approximate Jacobian = dF/dS = I - deltat * odeJac
    function psetup(
        u, uscale, 
        fval, fscale,
        d
    )
        if !d.jac_full
            # calculate "normal" derivative as partial Jacobian relies on some of these Variables
            PALEOmodel.set_tforce!(d.modeldata.solver_view_all, d.tmodel[])
            PALEOmodel.set_statevar!(d.modeldata.solver_view_all, u)                 
            PB.do_deriv(d.modeldata.dispatchlists_all)
        end

        # jac calculates un-normalized J
        d.jac(d.Jnorm, u, d, d.tmodel[])
        # convert J  = I - deltat * odeJac  
        for j in 1:size(d.Jnorm)[2]
            # idx is index in SparseMatrixCSC compressed storage, i is row index
            for idx in d.Jnorm.colptr[j]:(d.Jnorm.colptr[j+1]-1)
                i = d.Jnorm.rowval[idx]

                d.Jnorm.nzval[idx] = -d.deltat[]*d.Jnorm.nzval[idx]
                if i == j
                    d.Jnorm.nzval[idx] += 1.0
                end

            end
        end

        d.J_lu[] = lu(d.Jnorm, check=false)


        return !issuccess(d.J_lu[])
    end

    function directional_jv(
        v, Jv, 
        u, new_u,
        d
    )
        # PALEOmodel.JacobianAD.directional_forwarddiff!(Jv, u, v, d, d.tmodel[])

        PALEOmodel.set_tforce!(d.directional_sv, d.tmodel[])
        for i in eachindex(u)
            d.directional_workspace[i] = ForwardDiff.Dual(u[i], v[i])
        end
        PALEOmodel.set_statevar!(d.directional_sv, d.directional_workspace)                       

        PB.do_deriv(d.directional_dispatchlists)

        # resid .=  (u .- d.current_state - d.deltat[].*d.deriv_worksp)

        PALEOmodel.get_statevar_sms!(d.directional_workspace, d.directional_sv)
        for i in eachindex(u)
            # use: partials(Dual(u,v)) = v for first term
            #      d.current_state is a constant so no second term
            Jv[i] = v[i] - d.deltat[]*ForwardDiff.partials(d.directional_workspace[i], 1)
        end

        return 0  # Success
    end

    # create kinsol instance
    kin = PALEOmodel.Kinsol.kin_create(
        ssf!, initial_state, 
        linear_solver = :FGMRES, 
        psetupfun=use_jac_ad_preconditioner ? psetup : nothing,
        psolvefun=use_jac_ad_preconditioner ? psolve : nothing,
        userdata=userdata,
        jvfun=use_directional_ad ? directional_jv : nothing;
        createkwargs...
    )

    # Vectors to accumulate solution at requested tss_output
    # Always write initial state as first entry (whether requested or not)
    iout = 1
    tsoln = [userdata.tmodel[]]                       # vector of pseudo-times
    soln = [copy(userdata.current_state)]        # vector of state vectors at each pseudo-time
    if !isempty(tss_output) && (tss_output[1] == userdata.tmodel[])
        # don't repeat initial state if that was requested
        iout += 1
    end

    # outer loop over pseudo-timesteps
    ptc_iter = 1
    sol = nothing
 
    @time while userdata.tmodel[] < tss_max && ptc_iter <= ptc_iter_max
        # limit timestep if necessary, to get to next output
        # keep track of the deltat we could have used
        deltat_full = userdata.deltat[]  
        # limit last timestep to get to tss_max
        userdata.deltat[] = min(userdata.deltat[], tss_max - userdata.tmodel[]) 
        if iout < length(tss_output)
            # limit timestep to get to next requested output
            userdata.deltat[] = min(userdata.deltat[], tss_output[iout] - userdata.tmodel[]) 
        end

        userdata.tmodel[] += userdata.deltat[]

        verbose && @info lpad("", 80, "=")
        @info "steadystate: ptc_iter $ptc_iter tss $(userdata.tmodel[]) "*
            "deltat=$(userdata.deltat[]) deltat_full=$(deltat_full) calling kinsol..."
        verbose && @info lpad("", 80, "=")
        
        sol_ok = true
        try
            # solve nonlinear system for this pseudo-timestep
            @time (sol, kinstats) = PALEOmodel.Kinsol.kin_solve(
                kin, userdata.current_state;                                                   
                solvekwargs...
            )
            
            if verbose                
                @info " * $kinstats"                                
                ssf!(deriv_worksp, sol, userdata)
                @info "  check Fnorm inf-norm $(norm(deriv_worksp, Inf)) 2-norm $(norm(deriv_worksp, 2))"
            else
                @info "    Residual 2-norm: $(kinstats.FuncNorm) Iterations: $(kinstats.NumNonlinSolvIters) "*
                    "ReturnFlag $(kinstats.ReturnFlag)"
            end

            sol_ok = kinstats.ReturnFlag in (Sundials.KIN_SUCCESS, Sundials.KIN_INITIAL_GUESS_OK)
                      
        catch e
            if isa(e, LinearAlgebra.SingularException)
                @warn "LinearAlgebra.SingularException"
                sol_ok = false # will force timestep reduction and retry
            else
                throw(e) # rethrow and fail
            end
        end

        # very crude pseudo-timestep adaptation (increase on success, reduce on failure)
        if sol_ok
            if userdata.deltat[] == deltat_full
                # we used the full deltat and it worked - increase deltat
                userdata.deltat[] *= deltat_fac
            else
                # we weren't using the full timestep (as an output was requested), so go back to full
                userdata.deltat[] = deltat_full
            end
            
            userdata.current_state .= sol
      
            if isempty(tss_output) ||                       # all records requested, or ...
                    (iout <= length(tss_output) &&              # (not yet done last requested record
                    userdata.tmodel[] >= tss_output[iout])      # and just gone past a requested record)              
                @info "    writing output record at tmodel = $(userdata.tmodel[])"
                push!(tsoln, userdata.tmodel[])
                push!(soln, copy(userdata.current_state))
                iout += 1
            end
        else
            @warn "iter failed, reducing deltat"
            userdata.tmodel[] -= userdata.deltat[]
            userdata.deltat[] /= deltat_fac^2
        end
        
        ptc_iter += 1
    end

    ptc_iter <= ptc_iter_max ||
        @warn "    max ptc iters $ptc_iter_max exceeded"

    # always write the last record even if it wasn't explicitly requested
    if tsoln[end] != userdata.tmodel[]
        @info "    writing output record at tmodel = $(userdata.tmodel[])"
        push!(tsoln, userdata.tmodel[])
        push!(soln, copy(userdata.current_state))
    end

    PALEOmodel.ODE.calc_output_sol!(outputwriter, run.model, tsoln, soln, modeldata)
    return nothing    
end

steadystate_ptc(
    run, initial_state, modeldata, tss::Float64, deltat_initial::Float64, tss_max::Float64; kwargs...
) = steadystate_ptc(run, initial_state, modeldata, (tss, tss_max), deltat_initial; kwargs...)
  
end # module
