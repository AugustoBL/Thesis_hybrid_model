# Model and table input and output for training, evaluation and experiments.

"""
Write named records as a tab-separated table.

Args:
    path: Destination file path. Existing output is overwritten.
    rows: Nonempty collection of records sharing named columns.

Returns:
    Output file path.
"""
function save_table(path, rows; delimiter='\t')
    mkpath(dirname(path))
    open(path, "w") do output
        columns = propertynames(first(rows))
        println(output, join(string.(columns), delimiter))
        for row in rows
            println(output, join((getproperty(row, column) for column in columns), delimiter))
        end
    end
    return path
end

"""Read a numeric/symbol TSV written by save_table."""
function load_table(path)
    lines = readlines(path)
    columns = Tuple(Symbol.(split(first(lines), '\t')))
    parse_field = value -> value == "true" ? true : value == "false" ? false :
        something(tryparse(Int, value), tryparse(Float64, value), Symbol(value))
    return [NamedTuple{columns}(Tuple(parse_field.(split(line, '\t')))) for line in lines[2:end]]
end

"""
Load a model from BSON.

Args:
    file path: BSON file path with the model_cpu entry.

Returns:
    Saved model.
"""
function load_model(filepath)
    BSON.@load filepath model_cpu
    return model_cpu
end

"""
Save a CPU copy of a model to BSON.

Args:
    file path: Output file path in an existing directory.
    model: Model to save. Model will be named model_cpu

Returns:
    Destination file path.
"""
function save_model(filepath, model)
    model_cpu = Flux.cpu(model)
    BSON.@save filepath model_cpu
    return filepath
end
