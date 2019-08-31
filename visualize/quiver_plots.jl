using Plots
# Here I am using Plots' 3d quiver plot.
plt = quiver(coord_list[1,:], coord_list[2,:], coord_list[3,:], quiver=(grad_list[1,:],grad_list[2,:], grad_list[3,:]), projection="3d")
