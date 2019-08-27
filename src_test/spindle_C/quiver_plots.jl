using PyPlot

quiver(coord_list,)


coord_list[][1]

len =1331
coord_list = Array{Array{Float64},2}(undef, length(grid_ranges),len)
for i in 1:length(grid_ranges[1])
    for j in 1:length(grid_ranges[2])
        for k in 1:length(grid_ranges[3])
            print(coord_list[i,j,k])
            coord_list[i,j,k] = [grid_ranges[1][i],grid_ranges[2][j],grid_ranges[3][k]]
        end
    end
end
