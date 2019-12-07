################################
# plot_sim.jl     functions plot trajectory/ road
# author:     mingyuw@stanford.edu
################################

using PyPlot

function plot_road(ax)
    # define the center line of main line and on ramp
    mainLine0_cl = zeros(N_ML, 2)
    mainLine1_cl = ones(N_ML, 2)  * LANE_WIDTH
    onramp_cl = zeros(N_OR, 2)

    for i=1:N_ML
        mainLine0_cl[i, 1] = TOTAL_LENGTH / N_ML * (i - 20)
        mainLine1_cl[i, 1] = TOTAL_LENGTH / N_ML * (i - 20)
    end

    for i=1:N_OR
        if i < 20
            onramp_cl[i ,1] = TOTAL_LENGTH/ N_ML * (i - 20)
            onramp_cl[i, 2] = onramp_cl[i, 2] + onramp_cl[i, 1] * tan(ONRAMP_ANG)
        else
            onramp_cl[i, 1] = TOTAL_LENGTH / N_ML * (i - 20)
        end
    end
    ax.plot(mainLine0_cl[1:16, 1], mainLine0_cl[1:16, 2] .- 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    ax.plot(mainLine0_cl[20:end, 1], mainLine0_cl[20:end, 2] .- 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    ax.plot(mainLine0_cl[:, 1], mainLine0_cl[:, 2] .+ 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    for i=1:N_ML-1
        section = mainLine0_cl[i:i+1, :]
        xs = [section[1, 1], section[1, 1], section[end, 1], section[end, 1]]
        ys = [section[1, 2] - 1 / 2.0 * LANE_WIDTH, section[1, 2] + 1 / 2.0 * LANE_WIDTH, section[end, 2] + 1 / 2.0 * LANE_WIDTH, section[end, 2] - 1 / 2.0 * LANE_WIDTH]
        ax.fill(xs ,ys, color=:gray)
        section = mainLine1_cl[i:i+1, :]
        xs = [section[1, 1], section[1, 1], section[end, 1], section[end, 1]]
        ys = [section[1, 2] - 1 / 2.0 * LANE_WIDTH, section[1, 2] + 1 / 2.0 * LANE_WIDTH, section[end, 2] + 1 / 2.0 * LANE_WIDTH, section[end, 2] - 1 / 2.0 * LANE_WIDTH]
        ax.fill(xs ,ys, color=:gray)
    end
    for i=1:N_OR-1
        section = onramp_cl[i:i+1, :]
        xs = [section[1, 1], section[1, 1], section[end, 1], section[end, 1]]
        ys = [section[1, 2] - 1 / 2.0 * LANE_WIDTH, section[1, 2] + 1 / 2.0 * LANE_WIDTH, section[end, 2] + 1 / 2.0 * LANE_WIDTH, section[end, 2] - 1 / 2.0 * LANE_WIDTH]
        ax.fill(xs ,ys, color=:gray)
    end
    ax.plot(mainLine1_cl[:, 1], mainLine1_cl[:, 2] .- 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    ax.plot(mainLine1_cl[:, 1], mainLine1_cl[:, 2] .+ 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    ax.plot(onramp_cl[:, 1], onramp_cl[:, 2] .- 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    ax.plot(onramp_cl[10:16, 1], onramp_cl[10:16, 2] .+ 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    # ax.plot(onramp_cl[6:end, 1], onramp_cl[6:end, 2] .+ 1/2.0 * LANE_WIDTH, color=:black, linewidth=1.5, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlim(-30, 120)
    ax.set_ylim(-15,10)
    ax.set_yticks((-LANE_WIDTH, 0, LANE_WIDTH))
    tight_layout()
    grid(true)
end


function plot_trajectory(traj, ax, col)
    ax.plot(traj[:,1], traj[:,2], color=col, linestyle="-", marker="*")
end

function plot_trajectory_rel(traj, ref, ax, col)
    ax.plot(traj[:,1], traj[:,2], color=col, linestyle="-", marker="*")
    # for i=1:size(traj)[1]
    #     plot([traj[i,1], ref[i,1]], [traj[i,2], ref[i,2]])
    # end
end


function plot_trajectory_3d(xtraj, zvalue, ax, col)
    ax.plot(xtraj[:,1], xtraj[:,2], zvalue, color=col)
end

function plot_obstacle(ax, pos, radius, col="red", alpha=1)
    theta = -pi:0.01:pi
    x = [radius * cos(th) for th in theta] .+ pos[1]
    y = [radius * sin(th) for th in theta] .+ pos[2]
    ax.plot(x, y, color=col, alpha=alpha)
end


function plot_test_field(ax, size)
    coor = size/2
    ax.plot([-coor, coor], [-coor, -coor], color="black")
    ax.plot([coor, coor], [-coor, coor], color="black")
    ax.plot([-coor, coor], [coor, coor], color="black")
    ax.plot([-coor, -coor], [-coor, coor], color="black")
end
