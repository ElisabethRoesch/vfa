push!(LOAD_PATH, "/Users/eroesch/github")
using vfa

vfa_1()
spindle_C =get_spindle_data()
a = CCNB1[:,2]
b = pMPS1[:,2]
c = pENSA[:,2]
plt=plot(a, b, c, xlab = "CCNB1", ylab = "pMPS1", zlab  = "pENSA", grid = "off", legend = :bottomright)
scatter!(a, b, c)
