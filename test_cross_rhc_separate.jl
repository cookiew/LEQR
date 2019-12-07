# extended unicycle model avoiding obstacle with uncertain speed change
# the costs are separated meaning only collision cost is considered to be risk sensitive
# other costs (tracking, control) are risk neutral

# mingyuw@stanford.edu

using ForwardDiff, PyPlot, LinearAlgebra, Distributions


################################
# include:
# linear dynamics quadratic cost solver
################################
include("lqr.jl")
include("plot_sim.jl")

################################
# global parameters
################################
radius = 0.25
state_dim = 7
ctrl_dim = 2

steps = 600
horizon = 6
plan_steps = 150
DT = horizon/steps
replan_steps = 50

################################
# kinematic bicycle model, augumented by the position and velocity of another vehicle
# state = [x, y, v, theta, ox, oy, ov]
# control = [acc, yaw_rate]
################################

function dynamics_forward(s, u, dt)
    x, y, vx, vy, ox, oy, ov = s
    accx, accy = u
    x_new = x + vx * dt
    y_new = y + vy * dt
    vx_new = vx + accx * dt
    vy_new = vy + accy * dt
    ox_new = ox
    oy_new = oy + dt * ov
    ov_new = ov
    return [x_new, y_new, vx_new, vy_new, ox_new, oy_new, ov_new]
end

function dynamics_linear(s, u, dt)
    x, y, vx, vy, ox, oy, ov = s
    A = [1 0 dt 0 0 0 0;
    0 1 0 dt 0 0 0;
    0 0 1 0 0 0 0;
    0 0 0 1 0 0 0;
    0 0 0 0 1 0 0;
    0 0 0 0 0 1 dt;
    0 0 0 0 0 0 1]
    B = [0 0; 0 0; dt 0; 0 dt; 0 0; 0 0; 0 0]
    return A, B
end


# ##################################
# cost functions
# stage cost: do not collide with the obstable and control cost
# final cost: get to goal location
# the cost functions are separated such that collision avoidance is risk-sensitive where as others
# are risk neutral
# ##################################
function stage_cost_neutral(s)
    state = s[1:state_dim]
    control = s[state_dim+1:state_dim+ctrl_dim]
    ref = s[state_dim+ctrl_dim+1:end]
    ox, oy, ov = state[5:end]
    dist = sqrt((state[1] - ox)^2 + (state[2] - oy)^2) - (2 * radius) - 0.5
    return 1 * (
                1*control[1]^2 + 1*control[2]^2 +
                1*(state[1] - ref[1])^2 + 1*(state[2] - ref[2])^2)
end

function stage_cost_sensitive(s)
    state = s[1:state_dim]
    control = s[state_dim+1:state_dim+ctrl_dim]
    ref = s[state_dim+ctrl_dim+1:end]
    ox, oy, ov = state[5:end]
    dist = sqrt((state[1] - ox)^2 + (state[2] - oy)^2) - (2 * radius) - 0.5
    return 1 * (
            8/((0.2*dist + 1)^10))
end

function final_cost_neutral(s)
    state = s[1:state_dim]
    ox, oy, ov = state[5:end]
    dist = sqrt((state[1] - ox)^2 + (state[2] - oy)^2) - (2 * radius) - 0.5
    ref = s[state_dim+ctrl_dim+1:end]
    return 1*(
                10 * (state[1] - ref[1])^2 + 10 * (state[2] - ref[2])^2)
end

function final_cost_sensitive(s)
    state = s[1:state_dim]
    ox, oy, ov = state[5:end]
    dist = sqrt((state[1] - ox)^2 + (state[2] - oy)^2) - (2 * radius) - 0.5
    ref = s[state_dim+ctrl_dim+1:end]
    return 1*(
                8/((0.2*dist + 1)^10))
end


# ##################################
# risk sensitive simulation and
# ##################################

function simulation(ax, x_init, gamma, d_cov, traj_ref, plot_flag)

    W_trajectory = [d_cov for i=1:plan_steps]

    # ###########################
    # initialize the iteration (forward simulation)
    # ###########################

    x_trajectory = zeros(plan_steps+1, state_dim)
    x_trajectory[1,:] = x_init
    u_trajectory = zeros(plan_steps, ctrl_dim)
    u_trajectory[:, 1] .= 0
    u_trajectory[:, 2] .= 0

    # forward simulation
    for t=1:plan_steps
        x_trajectory[t+1, :] = dynamics_forward(x_trajectory[t, :], u_trajectory[t,:], DT)
    end

    ##################
    # iteration
    ################
    tol = 0.1
    itr = 1
    err = 10
    prev_err = 10
    max_itr = 500

    x_trajectory_prev = x_trajectory
    u_trajectory_prev = u_trajectory
    K_list = []
    k_list = []
    while (abs(err) > tol && itr < max_itr)

        ###########################
        #  linear quadratic approximation
        # ###########################
        F_trajectory = []
        Cn_trajectory = []
        cn_trajectory = []
        Cr_trajectory = []
        cr_trajectory = []
        for t=1:plan_steps
            At, Bt = dynamics_linear(x_trajectory_prev[t, :], u_trajectory_prev[t, :], DT)
            Ft = [At Bt]
            Cnt = ForwardDiff.hessian(stage_cost_neutral, [x_trajectory_prev[t,:]; u_trajectory_prev[t,:]; traj_ref[t,:]])[1:state_dim+ctrl_dim, 1:state_dim+ctrl_dim]
            cnt = ForwardDiff.gradient(stage_cost_neutral, [x_trajectory_prev[t,:]; u_trajectory_prev[t,:]; traj_ref[t,:]])[1:state_dim+ctrl_dim]
            Crt = ForwardDiff.hessian(stage_cost_sensitive, [x_trajectory_prev[t,:]; u_trajectory_prev[t,:]; traj_ref[t,:]])[1:state_dim+ctrl_dim, 1:state_dim+ctrl_dim]
            crt = ForwardDiff.gradient(stage_cost_sensitive, [x_trajectory_prev[t,:]; u_trajectory_prev[t,:]; traj_ref[t,:]])[1:state_dim+ctrl_dim]
            push!(F_trajectory, Ft)
            push!(Cn_trajectory, Cnt)
            push!(cn_trajectory, cnt)
            push!(Cr_trajectory, Crt)
            push!(cr_trajectory, crt)
        end
        Cnt = ForwardDiff.hessian(final_cost_neutral, [x_trajectory_prev[end,:]; u_trajectory_prev[end,:]; traj_ref[end,:]])[1:state_dim+ctrl_dim, 1:state_dim+ctrl_dim]
        cnt = ForwardDiff.gradient(final_cost_neutral, [x_trajectory_prev[end,:]; u_trajectory_prev[end,:]; traj_ref[end,:]])[1:state_dim+ctrl_dim]
        Crt = ForwardDiff.hessian(final_cost_sensitive, [x_trajectory_prev[end,:]; u_trajectory_prev[end,:]; traj_ref[end,:]])[1:state_dim+ctrl_dim, 1:state_dim+ctrl_dim]
        crt = ForwardDiff.gradient(final_cost_sensitive, [x_trajectory_prev[end,:]; u_trajectory_prev[end,:]; traj_ref[end,:]])[1:state_dim+ctrl_dim]
        push!(Cn_trajectory, Cnt)
        push!(cn_trajectory, cnt)
        push!(Cr_trajectory, Crt)
        push!(cr_trajectory, crt)


        # ###########################
        # backward computation
        # ###########################
        flag = false
        test_gamma = gamma
        while !flag
            _, _, _, _, K_list, k_list, flag = mixed_leqr_general(Cn_trajectory, cn_trajectory, Cr_trajectory, cr_trajectory, F_trajectory, W_trajectory, test_gamma)
            test_gamma /= 2
        end
        # if test_gamma != gamma
        #     println(" ---------------------------- ")
        #     println(" non exploding gamma ", test_gamma)
        # end

        ###########################
        # forward simulation
        # ###########################
        s_size = 1
        u_trajectory = zeros(plan_steps, ctrl_dim)
        x_trajectory = zeros(plan_steps + 1, state_dim)
        x_trajectory[1,:] = x_init
        for t=1:plan_steps
            u_trajectory[t,:] = (K_list[end-t+1] * (x_trajectory[t, :] - x_trajectory_prev[t,:]) + k_list[end-t+1]*0.1) + u_trajectory_prev[t,:]
            # cap the control input to prevent from diverging
            u_trajectory[t, 1] = min(2, u_trajectory[t, 1])
            u_trajectory[t, 2] = min(2, u_trajectory[t, 2])
            x_trajectory[t+1, :] = dynamics_forward(x_trajectory[t, :], u_trajectory[t,:], DT)
        end

        ###########################
        # print for debugging
        # ###########################
        if itr % 10 == 0
            cost, cost_state, cost_control, cost_final = evaluate_traj_separately(x_trajectory, u_trajectory, traj_ref)
            println("--- at itr: ", itr, " residual is ", err, " cost is ", cost)
        end


        ###########################
        # book keeping and convergence test
        # ###########################
        prev_err = err
        err = sum(abs.(x_trajectory_prev - x_trajectory))

        itr += 1
        x_trajectory_prev = x_trajectory
        u_trajectory_prev = u_trajectory
    end

    return x_trajectory_prev, u_trajectory_prev, K_list, k_list
end


function evaluate_traj_separately(x_trajectory, u_trajectory, traj_ref)
    cost_state = 0
    cost_control = 0
    cost_tracking = 0
    for t=1:size(u_trajectory)[1]
        state = x_trajectory[t,:]
        control = u_trajectory[t,:]
        ox, oy, ov = state[5:end]
        ref = traj_ref[t,:]
        dist = sqrt((state[1] - ox)^2 + (state[2]-oy)^2) - (2 * radius) - 0.5
        cost_state +=  8/((0.2*dist + 1)^10)
        cost_control += 1*control[1]^2 + 1*control[2]^2
        cost_tracking += 1*(state[1] - ref[1])^2 + 1*(state[2] - ref[2])^2
    end
    # final state
    state = x_trajectory[end,:]
    control = u_trajectory[end,:]
    ox, oy, ov = state[5:end]
    ref = traj_ref[end,:]
    dist = sqrt((state[1] - ox)^2 + (state[2]-oy)^2) - (2 * radius) - 0.5
    cost_state +=  8/((0.2*dist + 1)^10)
    cost_tracking += 1*(state[1] - ref[1])^2 + 1*(state[2] - ref[2])^2

    # total cost
    cost = cost_state + cost_control + cost_tracking
    return cost, cost_state, cost_control, cost_tracking
end


function evaluate_receding_horizon(d_cov, ax, axpgain, axdgain, axctrl, num, gamma, color, plot_flag)

    d = MvNormal(zeros(state_dim), d_cov)
    x_init = [-4 0 1.4 0.0 0 -4 1.3]

    # ########################
    # get a reference trajectory (longer than simulation trajectory)
    # #######################
    x_ref = zeros(steps*2, state_dim)
    x_ref[1,:] = x_init
    for i=1:steps*2 - 1
        x_ref[i+1,:] = dynamics_forward(x_ref[i,:], zeros(ctrl_dim), DT)
    end

    # ########################
    # book keeping
    # #######################
    cost_list, cost_coll_list, cost_control_list, cost_track_list = [], [], [], []
    min_dist_list, cross_position_list = [], []

    # ########################
    # solve for receding horion policy
    # #######################
    for i=1:num     # outer loop for simulation
        println(" =================   sim ", i, " ============ " )

        x_history = zeros(steps+1, state_dim)
        x_history[1,:] = x_init
        u_history = zeros(steps, ctrl_dim)

        # start one simulation
        x_nominal, u_nominal, K_list, k_list = [], [], [], []
        for t=1:steps
            if t%replan_steps == 1     # replan 100 time steps (one second)
                x_nominal, u_nominal, K_list, k_list = simulation(ax, x_history[t,:], gamma, d_cov, x_ref[t:t+plan_steps,:], plot_flag)   # return length is 200, starting from current time step
                pcost, pcost_state, pcost_control, pcost_tracking = evaluate_traj_separately(x_nominal, u_nominal, x_ref[t:t+plan_steps,:])

            end
            function my_mod(val)
                if val % replan_steps == 0
                    return replan_steps
                else
                    return val % replan_steps
                end
            end
            # save control and step forward
            u_history[t,:] = (K_list[end - my_mod(t) + 1] * (x_history[t, :] - x_nominal[my_mod(t),:]) + k_list[end - my_mod(t) + 1]) + u_nominal[my_mod(t),:]
            noise = rand(d)
            x_history[t+1, :] = dynamics_forward(x_history[t, :], u_history[t,:], DT) + noise
            if plot_flag
                if t%50 == 1     # replan 100 time steps (one second)
                    ax.clear()
                    ax.scatter(4, 0, marker="D", s=100, c="y")
                    ax.scatter(-4, 0, marker="X", s=100, c="y")
                    ax.scatter(0, -4, marker="X", s=100, c="m")

                    plot_obstacle(ax, [x_history[t,1], x_history[t,2]], radius, "green", steps/steps)
                    plot_obstacle(ax, [x_history[t,5], x_history[t,6]], radius, "red", steps/steps)
                    ax.plot(x_nominal[:,1], x_nominal[:,2])
                    ax.plot(x_nominal[:,5], x_nominal[:,6])
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    # ax.axis("equal")
                    ax.set_xlim(-5, 5)
                    ax.set_xticks([-3,-1, 1, 3])
                    ax.set_ylim(-5, 5)
                    ax.set_yticks([-3,-1, 1, 3])
                    # fn = string("data/03/fig/sim", string(t), ".jpeg")
                    # savefig(fn, format="jpeg",dpi=1000);
                    pause(0.1)
                end
            end

        end

        cost, cost_state, cost_control, cost_tracking = evaluate_traj_separately(x_history, u_history, x_ref)
        min_d = minimum_distance(x_history)
        cross_p = crossing_position(x_history)
        push!(cost_list, cost)
        push!(cost_coll_list, cost_state)
        push!(cost_control_list, cost_control)
        push!(cost_track_list, cost_tracking)
        push!(cross_position_list, cross_p)
        push!(min_dist_list, min_d)
    end


    return cost_list, cost_coll_list, cost_control_list, cost_track_list, min_dist_list, cross_position_list
end


function minimum_distance(x_history)
    curr_min = 10000000
    for t=1:size(x_history)[1]
        curr_min = min(curr_min, norm([x_history[t,1:2] - x_history[t,5:6]]))
    end
    return curr_min
end

function crossing_position(x_history)
    for t=1:size(x_history)[1]
        if abs(x_history[t,1]) < 0.01
            return [x_history[t,1], x_history[t,5]]
        end
    end
    return [x_history[end, 1], x_history[end, 5]]
end

function compare_noise(gamma, plot_flag)
    color_list = ["r", "g", "b"]
    ov_cov_list = [0.002]

    d_cov = zeros(state_dim, state_dim)
    x_cov = 1e-10
    y_cov = 1e-10
    v_cov = 1e-10
    the_cov = 1e-10
    ox_cov = 1e-10
    oy_cov = 1e-10
    d_cov[1,1] = x_cov
    d_cov[2,2] = y_cov
    d_cov[3,3] = v_cov
    d_cov[4,4] = the_cov
    d_cov[5,5] = ox_cov
    d_cov[6,6] = oy_cov

    figpgain = figure(figsize=(3,3))
    axpgain = figpgain.add_subplot(111)
    axpgain.set_ylabel("feedback proportional gain")
    axpgain.set_xlabel("time step ")

    figdgain = figure(figsize=(3,3))
    axdgain = figdgain.add_subplot(111)
    axdgain.set_xlabel("time step ")
    axdgain.set_ylabel("feedback detivative gain")

    figctrl = figure(figsize=(4,4))
    axctrl = figctrl.add_subplot(111)
    axctrl.set_xlabel("time step ")
    axctrl.set_ylabel("control input")

    figtraj = figure(figsize=(5,5))
    axtraj = figtraj.add_subplot(111)


    for i=1:length(ov_cov_list)
        d_cov[7,7] = ov_cov_list[i]
        d_cov *= DT


        cost_list, cost_coll_list, cost_control_list, cost_track_list, min_dist_list, cross_position_list =
                    evaluate_receding_horizon(d_cov, axtraj, axpgain, axdgain, axctrl, 1, gamma, color_list[i], plot_flag)

    end
end



# ##################################
# functions to test
# ##################################
println(" risk neutral case ")
compare_noise(0, true)
println(" risk sensitive case ")
compare_noise(99, true)
