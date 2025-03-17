using LineSearches

ϕ(x) = (x - π)^4
dϕ(x) = 4*(x-π)^3
ϕdϕ(x) = ϕ(x),dϕ(x)

α0 = 9.0
ϕ0 = ϕ(0.0)
dϕ0 = dϕ(0.0)

for ls in (Static,BackTracking,HagerZhang,MoreThuente,StrongWolfe)
    res = (ls())(ϕ, dϕ, ϕdϕ, α0, ϕ0,dϕ0)
    println(ls, ": ", res)
end