"""
    FieldArray(
        name::String, values::Array;
        dims::Union{Nothing, Vector{<:AbstractString}, Vector{<:Pair}}=nothing,
        coords::Vector{<:FieldArray}=FieldArray[],
        attributes=nothing,
    )

A generic [xarray](https://xarray.pydata.org/en/stable/index.html)-like or 
[IRIS](https://scitools-iris.readthedocs.io/en/latest/)-like 
Array with named dimensions and optional coordinates.

NB: this aims to be simple and generic, not efficient !!! Intended for representing model output,
not for numerically-intensive calculations.

# Examples

2-dimensional FieldArray with no supplied dimension names or coordinates:

    julia> fa = PALEOmodel.FieldArray("vec", [1 2 3; 4 5 6])
    FieldArray(name="vec", eltype=Int64)
      dims: (NamedDimension(name=dim_1, size=2, coords=String[]), NamedDimension(name=dim_2, size=3, coords=String[]))

2-dimensional FieldArray with supplied dimension names, no coordinates:

    julia> fa = PALEOmodel.FieldArray("vec", [1 2 3; 4 5 6]; dims=["x", "y"])
    FieldArray(name="vec", eltype=Int64)
      dims: (NamedDimension(name=x, size=2, coords=String[]), NamedDimension(name=y, size=3, coords=String[]))

2-dimensional FieldArray with supplied dimension names and coordinates:

    julia> fa = PALEOmodel.FieldArray(
                "vec", [1 2 3; 4 5 6]; 
                dims=[
                    "y"=>("ymid",), 
                    "x"=>("xmid",)
                ], 
                coords=[
                    PALEOmodel.FieldArray("xmid", [0.5, 1.5, 2.5], dims=["x"]), 
                    PALEOmodel.FieldArray("ymid", [0.5, 1.5], dims=["y"])
                ],
            )

    FieldArray(name="vec", eltype=Int64)
      dims: (NamedDimension(name=y, size=2, coords=["ymid"]), NamedDimension(name=x, size=3, coords=["xmid"]))
      coords:
        FieldArray(name="xmid", eltype=Float64, dims=(NamedDimension(name=x, size=3, coords=String[]),))
        FieldArray(name="ymid", eltype=Float64, dims=(NamedDimension(name=y, size=2, coords=String[]),))

"""
struct FieldArray{T, N}
    name::String
    values::T
    dims::NTuple{N, PB.NamedDimension}
    coords::Vector{FieldArray}
    attributes::Union{Dict, Nothing}
end


function FieldArray(
    name::String, values::Array;
    dims::Union{Nothing, Vector{<:AbstractString}, Vector{<:Pair}}=nothing,
    coords::Vector{<:FieldArray}=FieldArray[],
    attributes=nothing,
)
    if isnothing(dims) 
        dims = [PB.NamedDimension("dim_$i", s, []) for (i, s) in enumerate(size(values))]
        
    elseif dims isa Vector{<:AbstractString}
        tmpdims = []
        for (d, s) in PB.IteratorUtils.zipstrict(dims, size(values); 
                errmsg="values has $(length(size(values))) dimensions but $(length(dims)) dims supplied")
            push!(tmpdims, PB.NamedDimension(d, s, []))
        end
        dims=tmpdims
    elseif dims isa Vector{<:Pair}
        tmpdims = []
        for ((dname, dcoords), s) in PB.IteratorUtils.zipstrict(dims, size(values); 
            errmsg="values has $(length(size(values))) dimensions but $(length(dims)) dims supplied")
            push!(tmpdims, PB.NamedDimension(dname, s, [dc for dc in dcoords]))
        end
        dims=tmpdims
    else
        error("invalid dims: ", dims)
    end

    # check supplied coords have same named dimensions of same size
    tmpcoords = FieldArray[] # narrow type
    for co in coords
        for codim in co.dims
            arraydimidx = findfirst(d->d.name == codim.name, dims)
            !isnothing(arraydimidx) || error("coord '$(co.name)' dimension '$(codim.name)' not present in array dimensions $([d.name for d in dims])")
            arraydim = dims[arraydimidx]
            codim.size == arraydim.size || error("coord '$(co.name)' dimension '$(codim.name)': size $(codim.size) != array dimension size $(arraydim.size)")
        end
        push!(tmpcoords, co)
    end

    return FieldArray(name, values, Tuple(dims), tmpcoords, attributes)
end


function Base.show(io::IO, fa::FieldArray)
    print(io, 
        "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values), ", dims=", fa.dims, ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", fa::FieldArray)
    println(io, "FieldArray(name=\"", fa.name, "\", eltype=", eltype(fa.values),")")
    println(io, "  dims: ", fa.dims)
   
    if !isempty(fa.coords)
        println(io, "  coords:")
        for co in fa.coords
            println(io, "    ", co)
        end
    end
    # println(io, "  attributes: ", fa.attributes)
end

# basic arithmetic operations
function Base.:*(fa_in::FieldArray, a::Real)
    fa_out = FieldArray("$a*"*fa_in.name, a.*fa_in.values, fa_in.dims, copy(fa_in.attributes))
    return fa_out
end
Base.:*(a::Real, fa_in::FieldArray) = fa_in*a

# default name from attributes
default_fieldarray_name(attributes::Nothing) = ""

function default_fieldarray_name(attributes::Dict)
    name = get(attributes, :domain_name, "")
    name *= isempty(name) ? "" : "."
    name *= get(attributes, :var_name, "")

    selectargs_records = get(attributes, :filter_records, NamedTuple())
    selectargs_region = get(attributes, :filter_region, NamedTuple())
    if !isempty(selectargs_region) || !isempty(selectargs_records)
        name *= "(" * join(["$k=$v" for (k, v) in Dict(pairs(merge(selectargs_records, selectargs_region)))], ", ") * ")"
    end
    
    return name
end

"""
    get_dimension(fa::FieldArray, dimname::AbstractString) -> nd::NamedDimension

Get dimension `dimname` (error if not present)
"""
function get_dimension(fa::FieldArray, dimname::AbstractString)
    arraydimidx = findfirst(d->d.name == dimname, fa.dims)
    !isnothing(arraydimidx) || error("dimension '$dimname' not present")
    return dims[arraydimidx]
end

"""
    get_coord_arrays(fa::FieldArray, dimname::AbstractString) -> coord_arrays::Vector{FieldArray}

Get coordinates for dimension `dimname` (empty Vector if no coordinates defined)
"""
function get_coord_arrays(fa::FieldArray, dimname::AbstractString)
    
    nd = get_dimension(fa, dimname)
    coord_names = nd.coords
    coord_arrays = FieldArray[]
    for cname in coord_names
        cnidx = findfirst(c->c.name == cname, fa.coords)
        !isnothing(cnidx) || error("dimension '$dimname' coordinate '$cname' not present")
        push!(coord_arrays, fa.coords[cnidx])
    end

    return coord_arrays
end

"""
    update_coordinates(varray::FieldArray, vec_coord_arrays::AbstractVector) -> FieldArray

Replace or add coordinates `vec_coord_arrays` to `varray`.

`new_coord_arrays` is a Vector of Pairs of "dim_name"=>(var1::FieldArray, var2::FieldArray, ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>(zmid::FieldArray, zlower::FieldArray, atm.zupper::FieldArray)]
"""
function update_coordinates(varray::FieldArray, vec_coord_arrays::AbstractVector)

    check_coords_argument(vec_coord_arrays) || 
        error("argument vec_coords_arrays should be a Vector of Pairs of \"dim_name\"=>(var1::FieldArray, var2::FieldArray, ...), eg: [\"z\"=>(zmid::FieldArray, zlower::FieldArray, atm.zupper::FieldArray)]")

    # generate Vector of NamedDimensions to use as new coordinates
    named_dimensions = PB.NamedDimension[]
    for (coord_name, coord_arrays) in vec_coord_arrays
        fixed_coords = []
        for coord_array in coord_arrays
            push!(fixed_coords, PB.FixedCoord(get(coord_array.attributes, :var_name, ""), coord_array.values, coord_array.attributes))
        end
        push!(
            named_dimensions, PB.NamedDimension(
                coord_name,
                length(first(fixed_coords).values),
                fixed_coords,
            )
        )
    end

    # replace coordinates
    varray_newcoords = FieldArray(varray.name, varray.values, Tuple(named_dimensions), varray.attributes)

    return varray_newcoords
end

# check 'coords' of form [] or ["z"=>[ ... ], ] or ["z"=>(...),]
check_coords_argument(coords) =
    isa(coords, AbstractVector) && (
        isempty(coords) || (
            isa(coords, AbstractVector{<:Pair}) &&
            isa(first(first(coords)), AbstractString) &&
            isa(last(first(coords)), Union{AbstractVector, Tuple})
        )
    )

#############################################################
# Create from PALEO objects
#############################################################

"""
    get_array(obj, ...) -> FieldArray

Get FieldArray from PALEO object `obj`
"""
function get_array end

"""
    get_array(f::Field [, selectargs::NamedTuple]; [attributes=nothing]) -> FieldArray

Return a [`FieldArray`](@ref) containing `f::Field` data values and
any attached coordinates, for the spatial region defined by `selectargs`.

Available `selectargs` depend on the grid `f.mesh`, and 
are passed to `PB.Grids.get_region`.

`attributes` (if present) are added to `FieldArray`
"""
function get_array(
    f::PB.Field{D, PB.ScalarSpace, V, N, M}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing,
) where {D, V, N, M}
    isempty(selectargs) ||
        error("get_array on Field f defined on ScalarSpace with non-empty selectargs=$selectargs")

    return FieldArray(default_fieldarray_name(attributes), f.values, f.data_dims, attributes)
end

function get_array(
    f::PB.Field{D, PB.CellSpace, V, 0, M}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing,    
) where {D, V, M}

    values, dims = PB.Grids.get_region(f.mesh, f.values; selectargs...)

    if !isnothing(attributes) && !isempty(selectargs)
        attributes = copy(attributes)
        attributes[:filter_region] = selectargs
    end

    return FieldArray(default_fieldarray_name(attributes), values, dims, attributes)
end

# single data dimension
# TODO generalize this to arbitrary data dimensions
function get_array(
    f::PB.Field{D, PB.CellSpace, V, 1, M}, selectargs::NamedTuple=NamedTuple();
    attributes=nothing,
) where {D, V, M}

    f_space_dims_colons = ntuple(i->Colon(), ndims(f.values) - 1)
    f_size_datadim = size(f.values)[end]

    dvalues, dims = PB.Grids.get_region(f.mesh, f.values[f_space_dims_colons..., 1]; selectargs...)
   
    d = (size(dvalues)..., f_size_datadim)
    values = Array{eltype(dvalues), length(d)}(undef, d...)
  
    if length(d) == 1
        # single cell - space dimension squeezed out
        for i in 1:f_size_datadim
            values[i], dims = PB.Grids.get_region(f.mesh, f.values[f_space_dims_colons..., i]; selectargs...)
        end
    else
        dvalues_colons = ntuple(i->Colon(), ndims(dvalues))
        for i in 1:f_size_datadim
            dvalues, dims = PB.Grids.get_region(f.mesh, f.values[f_space_dims_colons..., i]; selectargs...)
            values[dvalues_colons..., i] .= dvalues
        end
    end

    if !isnothing(attributes) && !isempty(selectargs)
        attributes = copy(attributes)
        attributes[:filter_region] = selectargs
    end

    return FieldArray(default_fieldarray_name(attributes), values, (dims..., f.data_dims...), attributes)
end



"""
    get_array(modeldata, varnamefull [, selectargs::NamedTuple] [; coords::AbstractVector]) -> FieldArray
   
Get [`FieldArray`](@ref) by Variable name, for spatial region defined by `selectargs`
(which are passed to `PB.Grids.get_region`).

Optional argument `coords` can be used to supply plot coordinates from Variable in output.
Format is a Vector of Pairs of "coord_name"=>("var_name1", "var_name2", ...)

Example: to replace a 1D column default pressure coordinate with a z coordinate:
 
    coords=["z"=>("atm.zmid", "atm.zlower", "atm.zupper")]

NB: the coordinates will be generated by applying `selectargs`,
so the supplied coordinate Variables must have the same dimensionality as `vars`.
"""
function get_array(
    modeldata::PB.AbstractModelData, varnamefull::AbstractString, selectargs::NamedTuple=NamedTuple();
    coords=nothing,
)
    var = PB.get_variable(modeldata.model, varnamefull)
    !isnothing(var) ||
        throw(ArgumentError("Variable $varnamefull not found"))
    f = PB.get_field(var, modeldata)
    attributes = copy(var.attributes)
    attributes[:var_name] = var.name
    attributes[:domain_name] = var.domain.name    

    varray = get_array(f, selectargs; attributes=attributes)

    if isnothing(coords)
        # keep original coords (if any)
    else
        check_coords_argument(coords) ||
            error("argument coords should be a Vector of Pairs of \"coord_name\"=>(\"var_name1\", \"var_name2\", ...), eg: [\"z\"=>(\"atm.zmid\", \"atm.zlower\", \"atm.zupper\"), ...]")

        vec_coords_arrays = [
            coord_name => Tuple(get_array(modeldata, cvn, selectargs) for cvn in coord_varnames) 
            for (coord_name, coord_varnames) in coords
        ]

        varray = update_coordinates(varray, vec_coords_arrays) 
    end

    return varray
end





#########################################
# get_region moved from PALEOboxes.jl
#########################################

"""
    get_region(grid::Union{PB.AbstractMesh, Nothing}, values; selectargs...) -> values_subset, (dim_subset::NamedDimension, ...)

Return the subset of `values` given by `selectargs` (Grid-specific keywords eg cell=, column=, ...)
and corresponding dimensions (with attached coordinates).
"""
function get_region(grid::Union{PB.AbstractMesh, Nothing}, values) end

"""
    get_region(grid::Nothing, values) -> values[]

Fallback for Domain with no grid, assumed 1 cell
"""
function get_region(grid::Nothing, values)
    length(values) == 1 ||
        throw(ArgumentError("grid==Nothing and length(values) != 1"))
    return values[], ()
end
   
"""
    get_region(grid::UnstructuredVectorGrid, values; cell) -> 
        values_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `cell::Union{Int, Symbol}`: an Int, or a Symbol to look up in `cellnames`
"""
function get_region(grid::PB.Grids.UnstructuredVectorGrid, values; cell::Union{Int, Symbol})
    if cell isa Int
        idx = cell
    else
        idx = get(grid.cellnames, cell, nothing)
        !isnothing(idx) ||
            throw(ArgumentError("cell ':$cell' not present in  grid.cellnames=$(grid.cellnames)"))
    end

    return (
        values[idx],
        (),  # no dimensions (ie squeeze out a dimension length 1 for single cell)
    )
end

"""
    get_region(grid::UnstructuredColumnGrid, values; column, [cell=nothing]) -> 
        values_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `column::Union{Int, Symbol}`: (may be an Int, or a Symbol to look up in `columnames`)
- `cell::Int`: optional cell index within `column`, highest cell is cell 1
"""
function get_region(grid::PB.Grids.UnstructuredColumnGrid, values; column, cell::Union{Nothing, Int}=nothing)

    if column isa Int
        column in 1:length(grid.Icolumns) ||
            throw(ArgumentError("column index $column out of range"))
        colidx = column
    else
        colidx = findfirst(isequal(column), grid.columnnames)
        !isnothing(colidx) || 
            throw(ArgumentError("columnname '$column' not present in  grid.columnnames=$(grid.columnnames)"))
    end

    if isnothing(cell)
        indices = grid.Icolumns[colidx]
        return (
            values[indices],
            (PB.NamedDimension("z", length(indices), PB.get_region(grid.z_coords, indices)), ),
        )
    else
        # squeeze out z dimension
        idx = grid.Icolumns[colidx][cell]
        return (
            values[idx],
            (),  # no dimensions (ie squeeze out a dimension length 1 for single cell)
        )
    end
    
end

"""
    get_region(grid::Union{CartesianLinearGrid{2}, CartesianArrayGrid{2}} , internalvalues; [i=i_idx], [j=j_idx]) ->
        arrayvalues_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `i::Int`: optional, slice along first dimension
- `j::Int`: optional, slice along second dimension

`internalvalues` are transformed if needed from internal Field representation as a Vector length `ncells`, to
an Array (2D if neither i, j arguments present, 1D if i or j present, 0D ie one cell if both present)
"""
function get_region(
    grid::Union{PB.Grids.CartesianLinearGrid{2}, PB.Grids.CartesianArrayGrid{2}}, internalvalues; 
    i::Union{Integer, Colon}=Colon(), j::Union{Integer, Colon}=Colon()
)
    return _get_region(grid, internalvalues, [i, j])
end

"""
    get_region(grid::Union{CartesianLinearGrid{3}, CartesianArrayGrid{3}}, internalvalues; [i=i_idx], [j=j_idx]) ->
        arrayvalues_subset, (dim_subset::NamedDimension, ...)

# Keywords for region selection:
- `i::Int`: optional, slice along first dimension
- `j::Int`: optional, slice along second dimension
- `k::Int`: optional, slice along third dimension

`internalvalues` are transformed if needed from internal Field representation as a Vector length `ncells`, to
an Array (3D if neither i, j, k arguments present, 2D if one of i, j or k present, 1D if two present,
0D ie one cell if i, j, k all specified).
"""
function get_region(
    grid::Union{PB.Grids.CartesianLinearGrid{3}, PB.Grids.CartesianArrayGrid{3}}, internalvalues;
    i::Union{Integer, Colon}=Colon(), j::Union{Integer, Colon}=Colon(), k::Union{Integer, Colon}=Colon()
)
    return _get_region(grid, internalvalues, [i, j, k])
end

function _get_region(
    grid::Union{PB.Grids.CartesianLinearGrid, PB.Grids.CartesianArrayGrid}, internalvalues, indices
)
    if !isempty(grid.coords) && !isempty(grid.coords_edges)
        dims = [
            PB.NamedDimension(grid.dimnames[idx], grid.coords[idx], grid.coords_edges[idx])
            for (idx, ind) in enumerate(indices) if isa(ind, Colon)
        ]
    elseif !isempty(grid.coords)
        dims = [
            PB.NamedDimension(grid.dimnames[idx], grid.coords[idx])
            for (idx, ind) in enumerate(indices) if isa(ind, Colon)
        ]
    else
        dims = [
            PB.NamedDimension(grid.dimnames[idx])
            for (idx, ind) in enumerate(indices) if isa(ind, Colon)
        ]
    end

    values = internal_to_cartesian(grid, internalvalues)
    if !all(isequal(Colon()), indices)
        values = values[indices...]
    end

    return values, Tuple(dims)    
end
