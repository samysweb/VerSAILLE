using NCubeV
using JLD
using Glob
using PyCall
using Plots

function summarize_and_load(folder, prefix)
    println("Loading results from $folder/$prefix-*.jld")
    results = []
    metadata = nothing
    for file in glob("$prefix-*.jld",folder)
        if occursin("summary",file)
            continue
        end
        cur_results = load(file)
        if haskey(cur_results,"backup_meta")
            metadata = cur_results["backup_meta"]
        end
        append!(results,cur_results["result"])
    end
    result_summary = NCubeV.VerifierInterface.reduce_results(results)
    save("$folder/$prefix-summary.jld","result",result_summary,"args",metadata)
    return (result_summary, metadata)
end

py"""
import numpy as np
import polytope as pc
from polytope.solvers import lpsolve
def cheby_ball(poly1):
    #logger.debug('cheby ball')
    if (poly1._chebXc is not None) and (poly1._chebR is not None):
        # In case chebyshev ball already calculated and stored
        return poly1._chebR, poly1._chebXc
    if isinstance(poly1, pc.Region):
        maxr = 0
        maxx = None
        for poly in poly1.list_poly:
            rc, xc = cheby_ball(poly)
            if rc > maxr:
                maxr = rc
                maxx = xc
        poly1._chebXc = maxx
        poly1._chebR = maxr
        return maxr, maxx
    if pc.is_empty(poly1):
        return 0, None
    # `poly1` is nonempty
    r = 0
    xc = None
    A = poly1.A
    c = np.negative(np.r_[np.zeros(np.shape(A)[1]), 1])
    norm2 = np.sqrt(np.sum(A * A, axis=1))
    G = np.c_[A, norm2]
    h = poly1.b
    sol = lpsolve(c, G, h)
    #return sol
    if sol['status'] == 0 or (sol['status'] == 4 and pc.is_inside(poly1,sol['x'][0:-1])):
        r = sol['x'][-1]
        if r < 0:
            return 0, None
        xc = sol['x'][0:-1]
    else:
        # Polytope is empty
        poly1 = pc.Polytope(fulldim=False)
        return 0, None
    poly1._chebXc = np.array(xc)
    poly1._chebR = np.double(r)
    return poly1._chebR, poly1._chebXc
"""
cheby_ball = py"cheby_ball"

py"""
import numpy as np
import polytope as pc
def get_extremes(poly1):
    import matplotlib as mpl
    V = pc.extreme(poly1)
    rc, xc = cheby_ball(poly1)
    x = V[:, 1] - xc[1]
    y = V[:, 0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x / mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2 * (y < 0)
    angle = angle * corr
    ind = np.argsort(angle)
    return V[ind, :]
"""
get_extremes = py"get_extremes"

function intersect(pc, p1, p2)
    iA = [p1.A; p2.A]
    ib = append!(p1.b,p2.b)

    return pc.Polytope(iA, ib)
end

function acc_bound_fun(rPos;scaler=1.0)
    # rPos >= rVel^2 / (2*A)
    # =>
    #rVel = 
    return -sqrt((1/scaler)*rPos*2*5.5)
end

function acc_draw_regions(ce_list;reuse=false,color=:yellow,detailed=true,drawThreshold=0.0,scaler=1.0)
    pc = pyimport("polytope")
    first=true
    xpts = range(0.1,100., length=500)
    eps=0.1
    if !reuse
        plot(grid=false)
    end
    supressed = 0
    for (i, star) in enumerate(ce_list)
        if ((star.bounds[1][2]-star.bounds[1][1])*(star.bounds[2][2]-star.bounds[2][1])) < drawThreshold
            supressed += 1
            continue
        end
        bounds = pc.box2poly([[star.bounds[1][1],star.bounds[1][2]],[star.bounds[2][1],star.bounds[2][2]]])
        p = intersect(pc,pc.Polytope(star.constraint_matrix,star.constraint_bias),bounds)
        drawn = false
        try
            # Manually compute cheby ball because other thing is broken
            cheby_ball(p)
            if pc.is_fulldim(p) && detailed
                points = get_extremes(p)
                if first
                    plot!(Shape(convert(Vector,points[:,1]),convert(Vector,points[:,2])), label="Unsafe",fillcolor = "#ff6f91",linecolor=plot_color(:white, 0.0))
                    first=false
                else
                    plot!(Shape(convert(Vector,points[:,1]),convert(Vector,points[:,2])), label="",fillcolor = "#ff6f91",linecolor=plot_color(:white, 0.0))
                end
                drawn = true
            else
                #plot!(Shape(convert(Vector,points[:,1]),convert(Vector,points[:,2])), label="",fillcolor = plot_color(:grey, 0.3))
                if detailed
                    println("Empty")
                end
            end
        catch e
            println(e)
            println("Failed")
        end
        if !drawn
            plot!(Shape(
            [star.bounds[1][1],star.bounds[1][1],star.bounds[1][2]+eps,star.bounds[1][2]+eps,star.bounds[1][1]],
            [star.bounds[2][1],star.bounds[2][2]+eps,star.bounds[2][2]eps,star.bounds[2][1],star.bounds[2][1]]),
            label="", fillcolor = "#ff6f91", linecolor=plot_color(:white, 0.0))
        end
    end
    println("Supressed the drawing of ",supressed, " tiny regions")
    return plot!(xpts,acc_bound_fun.(xpts;scaler=scaler),label="State Space Bound",linecolor="#ffc75f", linewidth=5)
end