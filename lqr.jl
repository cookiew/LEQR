################################
# lqr.jl      solve finite horizon , discrete time LQR, LEQR problem
# author:     mingyuw@stanford.edu
################################

###############################
# optimal control problem of the form
#     min     sum_{t=1}^{T+1} (x_{t}'Q_tx_{t} + u_{t}'R_tu_{t})
#     s.t.    x_{t+1} = A_t x_t + B_t u_t
# finite horizon linear time varying (LTV) system
# #############################
# notice index shift by 1 compared to notes
# It is assumed that R_{T+1} = 0, Q_{T+1} = H is the final cost
###############################
function lqr(R, Q, A, B)
    # convention: stage cost: (x_{t}'Q_tx_{t} + u_{t}'R_tu_{t})
    # input: R, Q, A, B are lists of cost and dynamic matrices
    # Q, R are of length (steps+1)
    # A, B are of length (steps)
    # return: control gain Lt and value matrix P in backward sequence

    steps = length(A)

    # ###############################
    # initialize for the last time step
    # ###############################
    P_future = [Q[end]]    # optimal value (length = steps + 1)
    L_future = []          # optimal policy feedback gain (length = steps)

    # ###############################
    # recurrence step
    # ###############################
    for i in range(steps, stop=1, step=-1)    # number of loops in total: steps
        At = A[i]
        Bt = B[i]
        Qt = Q[i]
        Rt = R[i]
        # control gain: L_t = -(B'_tP_{t+1})B + R_{t}^-1 B'_tP_{t+1}A_t
        Pt1 = P_future[end]
        Lt = -inv(Rt + Bt' * Pt1 * Bt) * Bt' * Pt1 * At
        # optimal value: Discrete Riccati Equations   v(x) = 1/2 x' P_t x
        Pt = Qt + Lt' * Rt * Lt + (At + Bt * Lt)' * Pt1 * (At + Bt * Lt)

        # append
        push!(P_future, Pt)
        push!(L_future, Lt)
    end

    return P_future, L_future
end


# #####################
# linear quadratic regulator in general form
# #####################
function lqr_general(C, c, F, f)
    # ########################################
    # lqr of the form
    #      min \sum_{t=0}^{T}c(xt, ut)
    #      s.t. x_t = f(x_{t-1}, u_{t-1})
    #  where f(xt, ut) = Ft [xt; ut] + ft
    #        c(xt, ut) = 1/2 [xt; ut]^T C_t [xt; ut] + [xt; ut]^T c_t
    # assumes that no terminal cost on control (u_{T} does not count, C_{uu,T} = C_{ux,T} = c_{u,T} = 0)
    # #######################################
    # ref:
    # Optimal Control, Trajectory Optimization, and Planning, CS 294-112:
    # Deep Reinforcement Learning, Sergey Levine
    # ########################################
    # C, c are of length (steps+1)
    # F, f are of length (steps)
    # ########################################

    # #####################
    # optimal control: u = Kt x + kt
    # optimal value:  1/2 x^T V_t x + x^T v_t
    # #####################
    # println(" - inside lqr solver - ")
    steps = length(F)
    state_dim, _ = size(F[1])

    # ###############################
    # initialize for the last time step
    # ##############################

    VT = C[end][1:state_dim, 1:state_dim]
    vT = c[end][1:state_dim,:]

    V_future = [VT]    # optimal value (length = steps + 1)
    v_future = [vT]
    K_future = []          # optimal policy feedback gain (length = steps)
    k_future = []

    # ###############################
    # recurrence step
    # ###############################
    for i in range(steps, stop=1, step=-1)    # number of loops in total: steps
        Vt1 = V_future[end]
        vt1 = v_future[end]

        # expression of Q function
        Qt = C[i] + F[i]' * Vt1 * F[i]
        # println(" this is the Q ", Qt, " for step ", i)
        qt = c[i][:] + F[i]' * Vt1 * f[i] + F[i]' * vt1
        # println(" this is the q ", qt, " for step ", i)

        Cxx = Qt[1:state_dim, 1:state_dim]
        Cxu = Qt[1:state_dim, state_dim+1:end]
        Cux = Qt[state_dim+1:end, 1:state_dim]
        Cuu = Qt[state_dim+1:end, state_dim+1:end]
        Cuu_inv = inv(Cuu)
        cx = qt[1:state_dim,:]
        cu = qt[state_dim+1:end,:]


        Kt = -Cuu_inv * Cux
        kt = -Cuu_inv * cu
        Vt = Cxx + Cxu * Kt + Kt' * Cux + Kt' * Cuu * Kt
        vt = cx + Cxu * kt + Kt' * cu + Kt' * Cuu * kt

        # append
        push!(V_future, Vt)
        push!(v_future, vt)
        push!(K_future, Kt)
        push!(k_future, kt)
    end

    return V_future, v_future, K_future, k_future
end


# #####################
# linear exponential quadratic regulator in general form
# #####################
function leqr_general(C, c, F, W, gamma)
    # ########################################
    # leqr of the form
    #      min R_{gamma}(\sum_{t=0}^{T}c(xt, ut))
    #      s.t. x_t = f_{t-1}(x_{t-1}, u_{t-1}, w_{t-1})
    #  where f(xt, ut) = Ft [xt; ut] + wt
    #        c(xt, ut) = 1/2 [xt; ut]^T C_t [xt; ut] + [xt; ut]^T c_t
    #        R_{gamma}() is the risk sensitive function
    # assumes that no terminal cost on control (u_{T} does not count, C_{uu,T} = C_{ux,T} = 0)
    # #######################################
    # ref:
    #    derivation of simple case is from EE266 (stochastic control) lecture notes
    # ########################################
    # C, c are of length (steps+1)
    # F, w are of length (steps)
    # ########################################

    # #####################
    # optimal control: u = Kt x + vt
    # optimal value:  1/2 (x^T Pt x + x^T st + rt)
    # #####################
    # println(" - inside leqr solver -")
    steps = length(F)
    state_dim = size(F[1])[1]
    @assert(length(C) == steps + 1)
    @assert(length(c) == steps + 1)
    @assert(length(W) == steps)

    # ###############################
    # initialize for the last time step
    # ###############################

    PT = C[end][1:state_dim, 1:state_dim]
    sT = 2*c[end][1:state_dim]

    P_future = [PT]
    s_future = [sT]
    K_future = []
    v_future = []

    # ###############################
    # recurrence step
    # ###############################
    for i in range(steps, stop=1, step=-1)    # number of loops in total: steps
        # optimal cost function for the next time step
        Pt1 = P_future[end]
        st1 = s_future[end]

        # compute the coefficients in Q function
        W_inv = inv(W[i])
        W_cur = W[i]

        Qt = C[i] + F[i]' * (Pt1 + gamma * Pt1 * inv(W_inv - gamma * Pt1) * Pt1) * F[i]
        qt = c[i] + 0.5 * F[i]' * W_inv * inv(W_inv - gamma * Pt1) * st1

        Qxx = Qt[1:state_dim, 1:state_dim]
        Quu = Qt[state_dim+1:end, state_dim+1:end]
        Quu_inv = inv(Quu)
        Qux = Qt[state_dim+1:end, 1:state_dim]
        Qxu = Qt[1:state_dim, state_dim+1:end]
        qx = qt[1:state_dim]
        qu = qt[state_dim+1:end]

        Kt = -Quu_inv * Qux
        vt = -Quu_inv * qu

        Pt = Qxx + Qxu*Kt + Kt'*Qux + Kt'*Quu*Kt
        st = 2 * (qx + Qxu * vt + Kt'*qu + Kt'*Quu*vt)

        # if the current gamma value is too high, then return false flag so planner should decrease the value
        if det(I - gamma * Pt1 * W_cur) < 0
            # println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # println("@@@ important check in lqr.jl function leqr_general at step ", i)
            # println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print("this is the determinant", det(I - gamma * Pt1 * W_cur))
            # println("this is gamma ", gamma)
            # println("this is pt ", Pt1)
            # println(" this is the eigenvalues of pt ", det(Pt1))
            # println("this is noise covariance ", W_cur)
            return -1, -1, -1, -1, false
        end

        # append
        push!(P_future, Pt)
        push!(s_future, st)
        push!(K_future, Kt)
        push!(v_future, vt)
    end

    return P_future, s_future, K_future, v_future, true
end



# #####################
# combine LQR and LEQR
# #####################
function mixed_leqr_general(Cn, cn, Cr, cr, F, W, gamma)
    # ########################################
    # this approach separate the cost into risk-neutral ones and risk-sensitive ones
    # leqr of the form
    #      min R_{gamma}(\sum_{t=0}^{T}c(xt, ut))
    #      s.t. x_t = f_{t-1}(x_{t-1}, u_{t-1}, w_{t-1})
    #  where f(xt, ut) = Ft [xt; ut] + wt
    #        c(xt, ut) = 1/2 [xt; ut]^T C_t [xt; ut] + [xt; ut]^T c_t
    #        R_{gamma}() is the risk sensitive function
    # assumes that no terminal cost on control (u_{T} does not count, C_{uu,T} = C_{ux,T} = 0)
    # #######################################
    # ref:
    #    derivation of simple case is from EE266 (stochastic control) lecture notes
    # ########################################
    # C, c are of length (steps+1)
    # F, w are of length (steps)
    # ########################################

    # #####################
    # optimal control: u = Kt x + vt
    # optimal value:  1/2 (x^T Pt x + x^T st + rt)
    # #####################
    # println(" - inside leqr solver -")
    steps = length(F)
    state_dim, _ = size(F[1])
    @assert(length(Cn) == steps + 1)
    @assert(length(cn) == steps + 1)
    @assert(length(Cr) == steps + 1)
    @assert(length(cr) == steps + 1)
    @assert(length(W) == steps)

    # ###############################
    # initialize for the last time step
    # ###############################

    PnT = Cn[end][1:state_dim, 1:state_dim]
    snT = 2*cn[end][1:state_dim]
    PrT = Cr[end][1:state_dim, 1:state_dim]
    srT = 2*cr[end][1:state_dim]

    Pn_future = [PnT]
    sn_future = [snT]
    Pr_future = [PrT]
    sr_future = [srT]
    K_future = []
    v_future = []

    # ###############################
    # recurrence step
    # ###############################
    for i in range(steps, stop=1, step=-1)    # number of loops in total: steps
        # optimal cost function for the next time step
        Pnt1 = Pn_future[end]
        snt1 = sn_future[end]
        Prt1 = Pr_future[end]
        srt1 = sr_future[end]

        # compute the coefficients in Q function
        W_inv = inv(W[i])
        W_cur = W[i]
        Qnt = Cn[i] + F[i]' * Pnt1 * F[i]
        Qrt = Cr[i] + F[i]' * (Prt1 + gamma * Prt1 * inv(W_inv - gamma * Prt1) * Prt1) * F[i]
        qnt = cn[i] + 0.5 * F[i]' * snt1
        qrt = cr[i] + 0.5 * F[i]' * W_inv * inv(W_inv - gamma * Prt1) * srt1
        Qt = Qnt + Qrt
        qt = qnt + qrt

        Qxx = Qt[1:state_dim, 1:state_dim]
        Quu = Qt[state_dim+1:end, state_dim+1:end]
        Quu_inv = inv(Quu)
        Qux = Qt[state_dim+1:end, 1:state_dim]
        Qxu = Qt[1:state_dim, state_dim+1:end]
        qx = qt[1:state_dim]
        qu = qt[state_dim+1:end]

        Kt = -Quu_inv * Qux
        vt = -Quu_inv * qu



        # use the separated Q values for value function

        Qnxx = Qnt[1:state_dim, 1:state_dim]
        Qnuu = Qnt[state_dim+1:end, state_dim+1:end]
        Qnux = Qnt[state_dim+1:end, 1:state_dim]
        Qnxu = Qnt[1:state_dim, state_dim+1:end]
        qnx = qnt[1:state_dim]
        qnu = qnt[state_dim+1:end]
        Qrxx = Qrt[1:state_dim, 1:state_dim]
        Qruu = Qrt[state_dim+1:end, state_dim+1:end]
        Qrux = Qrt[state_dim+1:end, 1:state_dim]
        Qrxu = Qrt[1:state_dim, state_dim+1:end]
        qrx = qrt[1:state_dim]
        qru = qrt[state_dim+1:end]



        Pnt = Qnxx + Qnxu*Kt + Kt'*Qnux + Kt'*Qnuu*Kt
        snt = 2 * (qnx + Qnxu * vt + Kt'*qnu + Kt'*Qnuu*vt)
        Prt = Qrxx + Qrxu*Kt + Kt'*Qrux + Kt'*Qruu*Kt
        srt = 2 * (qrx + Qrxu * vt + Kt'*qru + Kt'*Qruu*vt)

        # if current gamma value is too high, then return false so the caller should decrease the risk parameter
        if det(I - gamma * Prt1 * W_cur) < 0
            return -1, -1, -1, -1, -1, -1, false
        end

        # append
        push!(Pn_future, Pnt)
        push!(sn_future, snt)
        push!(Pr_future, Prt)
        push!(sr_future, srt)
        push!(K_future, Kt)
        push!(v_future, vt)
    end

    return Pn_future, sn_future, Pr_future, sr_future, K_future, v_future, true
end
