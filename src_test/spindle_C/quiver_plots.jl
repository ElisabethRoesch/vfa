using Plots
x=[1,2,3]
y=[1,2,3]
z=[1,2,3]
u=[1,2,3]
v=[1,2,3]
w=[1,2,3
# here I am using Plots quiver plot
plt = quiver(coord_list[1,:], coord_list[2,:], coord_list[3,:], quiver=(grad_list[1,:],grad_list[2,:], grad_list[3,:]), projection="3d")
