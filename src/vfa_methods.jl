function get_coord_list(grid_ranges)
    len = length(grid_ranges[1])*length(grid_ranges[2])*length(grid_ranges[3])
    coord_list = Array{Float64,2}(undef, length(grid_ranges),len)
    global c = 0
    for i in 1:length(grid_ranges[1])
        for j in 1:length(grid_ranges[2])
            for k in 1:length(grid_ranges[3])
                global c = c + 1
                coord_list[:,c] = [grid_ranges[1][i],grid_ranges[2][j],grid_ranges[3][k]]
            end
        end
    end
    return coord_list
end

function get_grad_and_end_point_list(coord_list, dudt)
    grad_list = Array{Float64,2}(undef, size(coord_list)[1], size(coord_list)[2])
    end_coord_list = Array{Float64,2}(undef, size(coord_list)[1], size(coord_list)[2])
    global d = 0
    for i in 1:length(coord_list)
        global d = d + 1
        grad_list[:,d] = Flux.data(dudt(coord_list[:,i]))
        end_coord_list[:,d] = coord_list[i].+grad_list[:,i]
    end
    return grad_list, end_coord_list
end
