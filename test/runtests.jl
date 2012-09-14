using RandomForest, Stage, RDatasets, DataStructures, DataArrays, DataFrames
import Ollam: print_confusion_matrix, train_svm, test_classification
import Base: length, start, done, next

# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------
immutable EachRow{T}
  matrix :: Array{T, 2}
end
length(e :: EachRow)      = size(e.matrix, 1)
start(e :: EachRow)       = 1
next(e :: EachRow, state) = (vec(e.matrix[state, :]), state + 1)
done(e :: EachRow, state) = state > length(e) ? true : false
eachrow(m) = EachRow(m)

fvs = { 
       [ 1 1 1 ],
       [ 1 0 1 ],
       [ 1 0 0 ],
       [ 0 1 1 ],
       [ 0 0 1 ],
       [ 0 1 0 ]
       }
truth = Int32[ 2, 1, 1, 2, 1, 1 ]
questions = [ Question("fv[1] == 1?", fv -> fv[1] == 1), Question("fv[3] == 1?", fv -> fv[3] == 1), Question("fv[2] == 1?", fv -> fv[2] == 1) ]
res = dt_train(fvs, truth, questions)
@debug res
println("TREE:")
print_tree(res)
scores = [ score(res, fv) for fv in fvs ]
@debug scores

res = rf_train(fvs, truth, questions)
for (i, t) in enumerate(res)
  println("TREES[$i]:")
  print_tree(t, print_sets = false)
end

scores = [ score(res, fv) for fv in fvs ]
@debug scores

@debug tile_bounds([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# -------------------------------------------------------------------------------------------------------------------------
# Boston property values
# -------------------------------------------------------------------------------------------------------------------------
raw         = dataset("MASS", "boston")
data        = float32(array(raw))
order       = shuffle([1:506])
train       = data[order[1:400], 1:(end-1)]
train_tgts  = vec(data[order[1:400], end])
lbs, ubs    = tile_bounds(train_tgts, N = 4)
train_truth = quantize(train_tgts, lbs, ubs)
test        = data[order[401:506], 1:(end-1)]
test_truth  = quantize(vec(data[order[401:506], end]), lbs, ubs)
cnames      = names(raw)[1:end-1]
questions   = vcat([ tile_questions(train[:, i], i, cnames[i], N = 10) for i = 1:length(cnames) ]...)

x = vec(data[order[401:506], end])
@debug "data bounds, min: $(minimum(data[1:400, end])) max: $(maximum(data[1:400, end]))"
for (i, (l, u)) in enumerate(zip(lbs, ubs))
  @debug "q$i -- $l $u $(x[map((x, y) -> x && y, x .>= l, x .< u)])"
end

res = dt_train(eachrow(train), train_truth, questions)
#print_tree(res, print_sets = false)
confmat = DefaultDict(Int32, DefaultDict{Int32, Int32}, () -> DefaultDict(Int32, Int32, 0)) 
hyp = [ best(res, fv)[1] for fv in eachrow(test) ]
for (h, t) in zip(hyp, test_truth)
  confmat[t][h] += 1
end
print_confusion_matrix(confmat)

confmat = DefaultDict(Int32, DefaultDict{Int32, Int32}, () -> DefaultDict(Int32, Int32, 0)) 
res = rf_train(collect(eachrow(train)), train_truth, questions, bags = 50)
hyp = [ best(res, fv)[1] for fv in eachrow(test) ]
for (h, t) in zip(hyp, test_truth)
  confmat[t][h] += 1
end
print_confusion_matrix(confmat)

# -------------------------------------------------------------------------------------------------------------------------
# Iris
# -------------------------------------------------------------------------------------------------------------------------
raw         = dataset("datasets", "iris")
data        = array(raw)
order       = shuffle([1:size(data, 1)])
train       = float32(data[order[1:75], 1:(end-1)])
train_truth = String[ string(s) for s in data[order[1:75], end] ]
test        = float32(data[order[76:150], 1:(end-1)])
test_truth  = String[ string(s) for s in data[order[76:150], end] ]
cnames      = names(raw)[1:end-1]
questions   = vcat([ tile_questions(train[:, i], i, cnames[i], N = 5) for i = 1:length(cnames) ]...)

@debug train_truth
res = dt_train(eachrow(train), train_truth, questions)
confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0)) 
hyp = [ best(res, fv)[1] for fv in eachrow(test) ]
for (h, t) in zip(hyp, test_truth)
  confmat[t][h] += 1
end
print_confusion_matrix(confmat)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0)) 
res = rf_train(collect(eachrow(train)), train_truth, questions, bags = 50)
hyp = [ best(res, fv)[1] for fv in eachrow(test) ]
for (h, t) in zip(hyp, test_truth)
  confmat[t][h] += 1
end
print_confusion_matrix(confmat)
