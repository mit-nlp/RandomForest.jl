module RandomForest

using Stage, DataStructures
import Base: -, &, +, /, show, max
export Node, Question, dt_train, rf_train, print_tree, score, tile_questions, tile_bounds, quantize, best

# -------------------------------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------------------------------
(-){T}(x :: Set{T}, y :: Set{T}) = setdiff(x, y)
(+){T}(x :: Set{T}, y :: T) = push!(x, y)

(-){T}(x :: Associative{T, Set{Int32}}, y :: Associative{T, Set{Int32}}) = [ k => x[k] - y[k] for k in keys(x) ]
(&){T}(x :: Associative{T, Set{Int32}}, y :: Associative{T, Set{Int32}}) = [ k => x[k] ∩ y[k] for k in intersect(keys(x), keys(y)) ]

(+){T}(x :: DefaultDict{T, Float32}, y :: DefaultDict{T, Float32}) = DefaultDict(0.0f0, [ k => x[k] + y[k] for k in keys(x) ∪ keys(y) ])
(/){T}(x :: Associative{T, Float32}, y :: Float32) = [ k => x[k] / y for k in keys(x) ]

function max{T}(x :: Associative{T, Float32})
  m_k = nothing #zero(T)
  m_v = -Inf
  for (k, v) in x
    if v > m_v
      m_k = k
      m_v = v
    end
  end
  return m_k, m_v
end

function total{T}(set :: Associative{T, Set{Int32}})
  ret = 0
  for k in keys(set)
    ret += length(set[k])
  end
  return ret
end


# -------------------------------------------------------------------------------------------------------------------------
# Types
# -------------------------------------------------------------------------------------------------------------------------
type Question
  name    :: String
  p       :: Function
end

abstract Node

type Terminal{T} <: Node
  set  :: Associative{T, Set{Int32}}
  p    :: DefaultDict{T, Float32}
end

type NonTerminal{T} <: Node
  question :: Question
  set      :: Associative{T, Set{Int32}}
  yes      :: Node
  no       :: Node
  function NonTerminal(q, set)
    ret          = new()
    ret.question = q
    ret.set      = set
    return ret
  end
end

# -------------------------------------------------------------------------------------------------------------------------
# Methods
# -------------------------------------------------------------------------------------------------------------------------
NonTerminal{T}(q, set :: Associative{T, Set{Int32}}) = NonTerminal{T}(q, set)

show(io :: IO, q :: Question) = print(io, "Q(\"$(q.name)\")")
show(io :: IO, nt :: NonTerminal) = print(io, "N($(nt.question), $(nt.yes), $(nt.no))")

function Terminal{T}(set :: Associative{T, Set{Int32}})
  tot = total(set)
  return Terminal(set, DefaultDict(0.0f0, [ t => float32(length(set[t]) / tot) for t in keys(set) ]))
end
show(io :: IO, t :: Terminal) = print(io, "T(N = $(total(t.set)), $(t.p))")

function compact_set{T}(set :: Associative{T, Set{Int32}})
  out = String[]
  for t in keys(set)
    for e in set[t]
      push!(out, "$e => $t")
    end
  end
  return "[ " * join(out, ", ") * " ]"
end
    
function print_tree(tree :: Node; prefix = "", print_sets = true)
  if prefix != ""
    println(prefix * "|")
  end
  
  if isa(tree, Terminal)
    println(prefix * "+- $(tree.p)" * (print_sets ? " :: " * compact_set(tree.set) : ""))
  else
    println(prefix * "+- $(tree.question)" * (print_sets ? " :: " * compact_set(tree.set) : ""))
    
    addpre = prefix == "" ? "   " : "|  "
    print_tree(tree.yes, prefix = prefix * addpre, print_sets = print_sets)
    print_tree(tree.no, prefix = prefix * addpre, print_sets = print_sets)
  end
end

# -------------------------------------------------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------------------------------------------------
function h(root)
  tot = total(root)
  p   = [ length(root[t]) / tot for t in keys(root) ]
  hx  = 0.0

  for i = 1:length(p)
    if p[i] > 0
      hx -= p[i] * log2(p[i])
    end
  end
  
  return hx
end

function hq(root, qset)
  tot = total(root)
  p_q = total(qset & root) / tot
  return p_q * h(qset & root) + (1.0 - p_q) * h(root - qset)
end

function split_set(root, sets, questions)
  h_root  = h(root)
  gains   = [ h_root - hq(root, sets[q]) for q = 1:length(sets) ]
  s_gains = sortperm(gains, rev = true)
  opt_q   = s_gains[1]

  return NonTerminal(questions[opt_q], root), gains[s_gains[1]], opt_q
end

function build_tree(root, sets, questions)
  n, h, opt_q = split_set(root, sets, questions)
  #@debug "h = $h root = $root, q = $(questions[opt_q])"
  if total(root) == 0 || length(root) <= 1 || h <= 0.0
    return Terminal(root)
  else
    node     = n
    node.yes = build_tree(sets[opt_q] & root, sets, questions)
    node.no  = build_tree(root - sets[opt_q], sets, questions)
    #@debug "node = $node, $(root) - $(sets[opt_q]) = $(root - sets[opt_q])"
    return node
  end
end

function dt_train{T}(fvs, truth :: Vector{T}, questions :: Vector{Question})
  sets = [ DefaultDict(T, Set{Int32}, () -> Set{Int32}()) for q = 1:length(questions) ]
  root = DefaultDict(T, Set{Int32}, () -> Set{Int32}())
  for (i, (fv, t)) in enumerate(zip(fvs, truth))
    for q = 1:length(questions)
      if questions[q].p(fv)
        sets[q][t] += int32(i)
      end
    end
    root[t] += int32(i)
  end

  return build_tree(root, sets, questions)
end

function rf_train{T}(fvs :: Vector, truth :: Vector{T}, questions :: Vector{Question}; bags = 50, subset_size = 0.8, feature_subset_size = 0.8, seed = 0)
  srand(seed)
    
  trees = Array(Node, bags)
  for b = 1:bags
    feature_subset = rand!(1:length(fvs[1]), zeros(Int32, int(feature_subset_size * length(fvs[1]))))
    fv_subset      = rand!(1:length(fvs), zeros(Int32, int(subset_size * length(fvs))))
    trees[b]       = dt_train(fvs[fv_subset], truth[fv_subset], questions)
  end
  return trees
end

# TODO scoring functions
function score(tree :: Node, fv)
  current = tree
  while isa(current, NonTerminal)
    truth = current.question.p(fv)
    if truth
      current = current.yes
    else
      current = current.no
    end
  end
  return current.p
end

function best(tree :: Node, fv)
  dist = score(tree, fv)
  max(dist)
end

function score(trees :: Vector{Node}, fv)
  scores = sum([ score(t, fv) for t in trees ])
  return scores / sum(values(scores))
end

function best(trees :: Vector{Node}, fv)
  dist = score(trees, fv)
  max(dist)
end

# TODO Question generation
function tile_bounds(values :: Vector; N = 4)
  lbounds   = Array(Float32, N)
  ubounds   = Array(Float32, N)
  tile_size = round(length(values) / N)
  sorted    = sort(values)
  for i = 1:N
    lbounds[i] = i == 1 ? -Inf : sorted[(i-1) * tile_size]
    ubounds[i] = i == N ?  Inf : sorted[i * tile_size]
  end
  return lbounds, ubounds
end

function quantize(values, lbounds, ubounds)
  ret = Int32[]
  N   = length(ubounds)
  for v in values
    q = 1
    while q < N
      if v >= lbounds[q] && v < ubounds[q]
        break
      end
      q += 1
    end
    push!(ret, q)
  end
  return ret
end

function tile_questions(values :: Vector, index, name; N = 4)
  lbs, ubs = tile_bounds(values, N = N)
  ret = Array(Question, N)
  for i = 1:N
    ret[i] = Question("$(lbs[i]) >= $name < $(ubs[i])", fv-> fv[index] >= lbs[i] && fv[index] < ubs[i])
  end
  return ret
end

end # module
