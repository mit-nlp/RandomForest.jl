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

# -------------------------------------------------------------------------------------------------------------------------
# autocrt
# -------------------------------------------------------------------------------------------------------------------------
function to_ilr(i :: Real) 
  base = int(floor(i))
  if i - base > 0.35
    return string(base) * "+"
  else
    return string(base)
  end
end

const first_datum = 6
const truth_index = 4

dlidf = readtable("/Users/swade/Desktop/autocrt-for-wade/class-DLI-R.txt", separator = '\t')
budf  = readtable("/Users/swade/Desktop/autocrt-for-wade/class-BU-Rt.txt", separator = '\t')
rawdf = append!(dlidf, budf)

# clean
above_na_thres = trues(size(rawdf, 1))
for i = 1:size(rawdf, 1)
  xxx = count(f -> isna(f), array(rawdf[i, :]))
  if xxx > 4
    @debug "Dropping subject $i ($(rawdf[i, 1])), too many nas ($xxx)"
    above_na_thres[i] = false
  end
end

for i = first_datum:size(rawdf, 2)
  m = mean(dropna(rawdf[i]))
  rawdf[i] = array(rawdf[i], 0.0)
end

for i = 1:size(rawdf, 1)
  x = find(f -> f < 0, array(rawdf[i, first_datum:end]))
  if length(x) > 0
    @debug "FOUND neg feature: $(rawdf[i, x])"
  end
end

idxs    = Bool[ (!isna(x) && x != -1) for x in rawdf[:, truth_index] ]
cleandf = rawdf[idxs & above_na_thres, :]

order      = shuffle([1:size(cleandf, 1)])
crt_data   = cleandf[order, :]
train_mask = trues(size(cleandf, 1))
test_mask  = falses(size(cleandf, 1))
for l in 0.0:0.5:1.0
  idxs = find(x -> x == l, crt_data[:, truth_index])
  @debug @sprintf("%3s -- %3d", l, length(idxs))
  for i = 1:(length(idxs) * 0.33)
    test_mask[idxs[i]]  = true
    train_mask[idxs[i]] = false
  end
end

train       = float32(array(crt_data[train_mask, first_datum:end]))
train_truth = String[ to_ilr(x) for x in array(crt_data[train_mask, truth_index]) ]
test        = float32(array(crt_data[test_mask, first_datum:end]))
test_truth  = String[ to_ilr(x) for x in array(crt_data[test_mask, truth_index]) ]
cnames      = names(crt_data)[first_datum:end]
questions   = vcat([ tile_questions(train[:, i], i, cnames[i], N = 4) for i = 1:length(cnames) ]...)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0)) 
res = rf_train(collect(eachrow(train)), train_truth, questions, bags = 100)
hyp = [ best(res, fv)[1] for fv in eachrow(test) ]
for (h, t) in zip(hyp, test_truth)
  confmat[t][h] += 1
end
print_confusion_matrix(confmat)

confmat = DefaultDict(String, DefaultDict{String, Int32}, () -> DefaultDict(String, Int32, 0)) 
model = train_svm(eachrow(train), train_truth, C = 0.1, iterations = 100)
scr = test_classification(model, eachrow(test), test_truth, record = (t, h) -> confmat[t][h] += 1) * 100.0
print_confusion_matrix(confmat)
