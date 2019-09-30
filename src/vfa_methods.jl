#returns points on a grid ad array
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
function get_coord_list_1d(grid_range)
    len = length(grid_range[1])
    coord_list = Array{Float64}(undef, len)
    for i in 1:len
        coord_list[i] = grid_range[1][i]
    end
    return coord_list
end
# returns gradients and end coordiante as arrays
function get_grad_and_end_point_list(coord_list, dudt)
    grad_list = Array{Float64,2}(undef, size(coord_list)[1], size(coord_list)[2])
    end_coord_list = Array{Float64,2}(undef, size(coord_list)[1], size(coord_list)[2])
    global d = 1
    for i in 1:size(coord_list)[2]
        grad_list[:,d] = Flux.data(dudt(coord_list[:,i]))
        end_coord_list[:,d] = coord_list[i].+grad_list[:,i]
        global d = d + 1
    end
    return grad_list, end_coord_list
end
# returns gradients and end coordiante as arrays
function get_grad_and_end_point_list_1d(coord_list, dudt)
    len = length(coord_list)
    grad_list = Array{Float32}(undef, len)
    end_coord_list = Array{Float32}(undef, len)

    for i in 1:len
        print("gradient is ", Flux.data(dudt([coord_list[i]]))[1], "an stelle ",i)
        print("\n")
        grad_list[i] = Flux.data(dudt([coord_list[i]]))[1]
        end_coord_list[i] = coord_list[i].+grad_list[i]
        print("start is ", coord_list[i], "an stelle ",i)
        print("\n")
        print("ende is ", end_coord_list[i], "an stelle ",i)
        print("\n")
    end
    return grad_list, end_coord_list
end

function dostuff(node_func, list_inits)
    all = []
    for i in list_inits
        push!(all, Flux.data(node_func([i])))
    end
    return all
end
