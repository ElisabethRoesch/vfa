function get_saddle_data()
    a= [0	1	2	3	4	5	6	7	8	9	10]
    b= [1.0	0.5	0.333334	0.249999	0.2	0.166666	0.142858	0.125	0.111111	0.1	0.0909095]
    x = Array{Float32,2}(undef,11,2)
    for i in 1:length(a)
        x[i,:]= [ a[i],b[i]]
    end
    return x
end
# function saddle_data()
#     a= [0	1	2	3	4	5	6	7	8	9	10]
#     b= [1.0	0.5	0.333334	0.249999	0.2	0.166666	0.142858	0.125	0.111111	0.1	0.0909095]
#     return a,b
# end

function read_bifurfile(bif_type, alpha, init)
    path = "/Users/eroesch/Documents/master-projects/bifurcations/public_html/assets/js/feel_data/"
    ending = string(path, bif_type,"_",alpha,"_",init,".csv")
    df = CSV.read(ending)
    df_as_arr = convert( Array , df )
    return df_as_arr
end
