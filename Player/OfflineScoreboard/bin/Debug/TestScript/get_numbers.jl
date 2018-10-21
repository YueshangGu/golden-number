# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

metaline = readline()
metaArray = split(metaline, '\t')
rowCount = parse(Int, metaArray[1])
columnCount = parse(Int, metaArray[2])

glist = Float64[]
for i in 1:rowCount
    values = [parse(Float64, ss) for ss in split(readline(), '\t')]
    push!(glist, values[1])
end

if rowCount == 0
    output = 17
else
    output = glist[end]
end

print(output)
print("\t")
print(output)
